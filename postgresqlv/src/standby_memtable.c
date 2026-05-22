#include "postgres.h"
#include "utils/elog.h"

#include "standby_memtable.h"
#include "lsmindex.h"

/* -----------------------------------------------------------------------
 * Private helpers
 * ----------------------------------------------------------------------- */

/*
 * find_loaded_slot — return the LSMIndexBufferSlot for indexRelId if it is
 * currently writable (recovery done; FlushedSegmentPool may or may not be
 * initialized — standby redo only needs the memtable, not the pool).
 * Returns NULL otherwise.
 */
static LSMIndexBufferSlot *
find_loaded_slot(Oid indexRelId)
{
    int slot_idx = lookup_lsm_index_idx(indexRelId);
    LSMIndexBufferSlot *slot;

    if (slot_idx < 0)
        return NULL;

    slot = &SharedLSMIndexBuffer->slots[slot_idx];
    if (!is_writable(pg_atomic_read_u32(&slot->valid)))
        return NULL;

    return slot;
}

/*
 * find_memtable_by_sid — linear search over all memtable slots known to the
 * given LSMIndex, looking for one whose memtable_id equals sid.
 * Returns NULL if not found.
 */
static ConcurrentMemTable
find_memtable_by_sid(LSMIndexBufferSlot *slot, SegmentId sid)
{
    LSMIndex lsm = &slot->lsmIndex;
    ConcurrentMemTable mt;

    /* Check the growing memtable */
    if (lsm->growing_memtable_idx != MT_IDX_INVALID &&
        lsm->growing_memtable_idx != MT_IDX_ROTATING)
    {
        mt = MT_FROM_SLOTIDX(lsm->growing_memtable_idx);
        if (mt->memtable_id == sid)
            return mt;
    }

    // FIXME: Impossible to reach here?
    /* Check sealed memtables */
    for (int i = 0; i < (int) lsm->memtable_count; i++)
    {
        mt = MT_FROM_SLOTIDX(lsm->memtable_idxs[i]);
        if (mt->memtable_id == sid)
            return mt;
    }

    return NULL;
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

/*
 * dpv_standby_register_memtable — allocate (or find) a memtable slot for sid
 * and wire it into the LSMIndex on the standby.
 */
void
dpv_standby_register_memtable(Oid indexRelId, SegmentId sid)
{
    LSMIndexBufferSlot *slot = find_loaded_slot(indexRelId);
    LSMIndex lsm;
    int mt_idx;
    ConcurrentMemTable mt;

    if (slot == NULL)
        return;

    lsm = &slot->lsmIndex;

    /*
     * Hold mt_lock exclusively for the entire dedup + allocate + wire
     * sequence so concurrent hot-standby query backends observe a
     * consistent set of LSMIndex fields.
     */
    LWLockAcquire(lsm->mt_lock, LW_EXCLUSIVE);

    /* Idempotency: if the memtable for this sid already exists, skip. */
    if (find_memtable_by_sid(slot, sid) != NULL)
    {
        LWLockRelease(lsm->mt_lock);
        return;
    }

    /*
     * register_and_set_memtable's `index` arg is only used in the
     * !is_recovery branch (to call RegisterStatusMemtable).  Passing NULL
     * with is_recovery=true is safe.
     */
    mt_idx = register_and_set_memtable(lsm, NULL, true, sid);
    if (mt_idx < 0)
    {
        LWLockRelease(lsm->mt_lock);
        elog(WARNING,
             "[dpv_standby_register_memtable] no free memtable slot for sid=%u",
             sid);
        return;
    }

    mt = MT_FROM_SLOTIDX(mt_idx);

    /*
     * Wire into the LSMIndex: treat as the growing memtable if its sid is
     * newer than the current growing one; otherwise enqueue as sealed.
     */
    if (sid > lsm->growing_memtable_id ||
        lsm->growing_memtable_idx == MT_IDX_INVALID)
    {
        /* Seal the previous growing memtable, if any. */
        if (lsm->growing_memtable_idx != MT_IDX_INVALID &&
            lsm->growing_memtable_idx != MT_IDX_ROTATING) // impossible to be MT_IDX_ROTATING
        {
            ConcurrentMemTable prev = MT_FROM_SLOTIDX(lsm->growing_memtable_idx);
            pg_atomic_write_u32(&prev->sealed, 1);
            if (lsm->memtable_count < MEMTABLE_NUM)
                lsm->memtable_idxs[lsm->memtable_count++] =
                    lsm->growing_memtable_idx;
        }
        lsm->growing_memtable_idx = mt_idx;
        lsm->growing_memtable_id  = sid;
    }
    else
    {
        /* This sid is older — treat as a sealed memtable. */
        pg_atomic_write_u32(&mt->sealed, 1);
        if (lsm->memtable_count < MEMTABLE_NUM)
            lsm->memtable_idxs[lsm->memtable_count++] = mt_idx;
    }

    LWLockRelease(lsm->mt_lock);
}

/*
 * dpv_standby_add_to_memtable — apply one inserted-vector WAL record to the
 * in-memory memtable on the standby.  The vector data is passed inline
 * (extracted from the WAL record by the redo callback).
 */
void
dpv_standby_add_to_memtable(Oid indexRelId, SegmentId sid,
                             uint32 slot_index, ItemPointer tid,
                             const void *vector, uint32 vector_bytes)
{
    LSMIndexBufferSlot *slot = find_loaded_slot(indexRelId);
    LSMIndex lsm;
    ConcurrentMemTable mt;

    if (slot == NULL)
        return;

    lsm = &slot->lsmIndex;

    /*
     * Take a shared lock to safely read the LSMIndex routing fields
     * (growing_memtable_idx, memtable_idxs, memtable_count) and obtain a
     * stable mt pointer.  Release before touching per-memtable fields,
     * which have their own atomic/barrier-based protection.
     */
    LWLockAcquire(lsm->mt_lock, LW_SHARED);
    mt = find_memtable_by_sid(slot, sid);
    LWLockRelease(lsm->mt_lock);
    if (mt == NULL)
        return;  /* Register hasn't arrived or already adopted */

    if (slot_index >= mt->capacity)
    {
        elog(WARNING,
             "[dpv_standby_add_to_memtable] slot_index %u >= capacity %u",
             slot_index, mt->capacity);
        return;
    }
    if (vector_bytes != VEC_BYTES_PER_ROW(mt))
    {
        elog(WARNING,
             "[dpv_standby_add_to_memtable] vector_bytes %u != expected %zu",
             vector_bytes, (size_t) VEC_BYTES_PER_ROW(mt));
        return;
    }

    /* Apply: tid + vector data */
    mt->tids[slot_index] = ItemPointerToInt64(tid);
    memcpy(VEC_PTR_AT(mt, slot_index), vector, vector_bytes);

    /* Advance current_size so queries see the new entry. */
    {
        uint32 new_size = slot_index + 1;
        uint32 cur;
        do {
            cur = pg_atomic_read_u32(&mt->current_size);
            if (cur >= new_size)
                break;
        } while (!pg_atomic_compare_exchange_u32(&mt->current_size, &cur, new_size));
    }

    /* Mark slot ready so queries can find it. */
    publish_slot_release(mt, slot_index);
}

/*
 * dpv_standby_update_max_sid — no-op: the status meta page is kept current
 * on the standby by the page-level GenericXLog FPI; no in-memory work needed.
 */
void
dpv_standby_update_max_sid(Oid indexRelId, SegmentId sid)
{
    (void) indexRelId;
    (void) sid;
}

/*
 * dpv_standby_release_memtable — intentional no-op for SharedMemtableBuffer
 * (spec §10).  Adoption drops the memtable slot; the page-level GenericXLog
 * has already cleaned the status page.
 */
void
dpv_standby_release_memtable(Oid indexRelId, SegmentId sid)
{
    (void) indexRelId;
    (void) sid;
}
