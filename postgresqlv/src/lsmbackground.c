#include "postgres.h"
#include "postmaster/postmaster.h"
#include "postmaster/bgworker.h"
#include "tcop/tcopprot.h"
#include "storage/shmem.h"
#include "storage/lwlock.h"
#include "storage/pmsignal.h"
#include "miscadmin.h"

#include "lsmindex.h"
#include "lsmbackground.h"
#include "vectorindeximpl.hpp"
#include "tasksend.h"
#include "lsm_merge_worker.h"

// FIXME: the order to flush? check all lsm index? Or scan buffer? Or use message to wake?

// claim the head sealed memtable for flushing(guarantee concurrency between background workers), but do not remove it yet
static int
claim_immutable_for_flush(LSMIndex lsm, bool wait)
{
    for (;;)
    {
        int mt_idx = -1;

        LWLockAcquire(lsm->mt_lock, LW_SHARED);

        uint32_t count = lsm->memtable_count;
        // find first unclaimed sealed memtable
        for (uint32_t i = 0; i < count; i++)
        {
            int cand = lsm->memtable_idxs[i];
            ConcurrentMemTable mt = MT_FROM_SLOTIDX(cand);

            uint32_t expect = 0;
            if (pg_atomic_compare_exchange_u32(&mt->flush_claimed, &expect, 1))
            {
                mt_idx = cand;
                break;
            }
            // if claimed by another background worker; keep scanning
        }

        LWLockRelease(lsm->mt_lock);

        if (mt_idx >= 0)
        {
            return mt_idx;
        }
        if (!wait)
        {
            return -1;
        }

        pg_usleep(1000);
    }
}

/* Wait until all reserved slots are published */
static inline void
wait_for_publish(ConcurrentMemTable t)
{
    for (;;)
    {
        if (pg_atomic_read_u32(&t->sealed) == 1)
        {
            uint32 cur_size = pg_atomic_read_u32(&t->current_size);
            uint32_t valid_size = Min(cur_size, t->capacity);
            uint32 rdy = pg_atomic_read_u32(&t->ready_cnt);
            if (rdy > valid_size)
            {
                elog(ERROR, "[wait_for_publish] rdy > valid_size");
            }
            else if (rdy == valid_size)
            {
                break;
            }
        }
        else
        {
            elog(ERROR, "[wait_for_publish] the memtable is not sealed");
        }
        pg_usleep(100); /* microseconds; tune as needed */
    }
}

// prepare everything needed to publish, but don't publish yet
// this heavy work is done outside locks
static void
prepare_for_flushing(LSMIndex lsm, int slot_idx, ConcurrentMemTable mt, PrepareFlushMeta prep)
{
    // wait until all reserved slots are published
    wait_for_publish(mt);

    uint32_t cur = pg_atomic_read_u32(&mt->current_size);
    uint32_t valid = (cur > mt->capacity)? mt->capacity: cur;
    
    prep->start_sid = mt->memtable_id;
    prep->end_sid = mt->memtable_id;
    prep->valid_rows = valid;
    prep->index_type = FLAT;

    // build hnsw type index on the segment
    // TODO: tune the parameters
    void *vector_index;
    IndexBuild(prep->index_type, mt, valid, &vector_index, 16, 100, (int)sqrt(MEMTABLE_MAX_CAPACITY));

    // serialize and flush the flat vector index
    IndexSerialize(vector_index, &prep->index_bin);
    IndexFree(vector_index);

    // build DSM mapping
    prep->map_size = sizeof(int64_t) * (Size)valid;
    prep->map_ptr = mt->tids;
    
    // build DSM bitmap (all ones; trim tail)
    prep->bitmap_size = GET_BITMAP_SIZE(valid);
    prep->bitmap_ptr = mt->bitmap;
}

// atomically publish segment and remove memtable from queue
static void
handoff_unlink(LSMIndex lsm, int mt_idx, PrepareFlushMeta prep)
{
    // global order to acquire locks: mt_lock -> seg_lock
    LWLockAcquire(lsm->mt_lock, LW_EXCLUSIVE);

    // find the claimed memtable in the queue
    int pos = -1;
    for (uint32_t i = 0; i < lsm->memtable_count; i++)
    {
        if (lsm->memtable_idxs[i] == mt_idx)
        {
            pos = i;
            break;
        }
    }

    if (pos < 0)
    {
        LWLockRelease(lsm->mt_lock);
        elog(ERROR, "[handoff] memtable idx %d not found in queue", mt_idx);
    }
    
    // remove the memtable from queue (compact)
    for (uint32_t i = (uint32_t)pos + 1; i < lsm->memtable_count; i++)
    {
        lsm->memtable_idxs[i - 1] = lsm->memtable_idxs[i];
    }
    --lsm->memtable_count;

    LWLockRelease(lsm->mt_lock);
}

static inline void
release_memtable_slot(int mt_idx)
{
    // decrease the reference count by 1
    // howeever, the memtable may still be in use by concurrent searches, so the reference count may not reach 0
    // this slot will not be reused until the reference count reaches 0
    pg_atomic_add_fetch_u32(&SharedMemtableBuffer->slots[mt_idx].ref_count, -1);
}

static bool 
lsm_flush_one_pending(LSMIndex lsm, int slot_idx, bool wait)
{
    // step 1. claim
    int mt_idx = claim_immutable_for_flush(lsm, wait);
    if (mt_idx < 0)
    {
        return false;
    }

    ConcurrentMemTable mt = MT_FROM_SLOTIDX(mt_idx);
    if (pg_atomic_read_u32(&mt->sealed) == 0)
    {
        elog(ERROR, "[lsm_flush_one_pending] claimed memtable not sealed");
    }

    // acquire vacuum lock
    LWLockAcquire(&mt->vacuum_lock, LW_SHARED);

    // step 2. prepare for flushing
    PrepareFlushMetaData prep;
    prepare_for_flushing(lsm, slot_idx, mt, &prep);

    // step 3. flush the segment to disk
    flush_segment_to_disk(lsm->indexRelId, &prep);

    // step 4. notify the vector index worker to load the segment from disk
    segment_update_blocking(slot_idx, lsm->indexRelId, SEGMENT_UPDATE_REGULAR, prep.start_sid, prep.end_sid);

    // step 5. atomic handoff
    handoff_unlink(lsm, mt_idx, &prep);

    // step 6. free the memtable slot
    release_memtable_slot(mt_idx);

    // step 7. update the segment array
    add_to_segment_array(slot_idx, lsm->indexRelId, prep.start_sid, prep.end_sid, prep.valid_rows, prep.index_type, 0.0f);
    
    // release vacuum lock
    LWLockRelease(&mt->vacuum_lock);
    
    return true;
}

static volatile sig_atomic_t got_sigterm = false;

static inline bool
OurPostmasterIsAlive(void)
{
	if (likely(!postmaster_possibly_dead))
		return true;
	return PostmasterIsAliveInternal();
}

void lsm_index_bgworker_main(Datum main_arg)
{
    // gracefully shut down if told to exit
    pqsignal(SIGTERM, die);
    // enable signals
    BackgroundWorkerUnblockSignals();

    // Connect to a database if needed
    // BackgroundWorkerInitializeConnection("your_db", NULL, 0);

    while (!got_sigterm)
    {
        CHECK_FOR_INTERRUPTS();

        if (!OurPostmasterIsAlive())
        {
            elog(LOG, "[lsm_index_bgworker_main] postmaster is dead, exiting.");
            proc_exit(1);
        }

        for (int slot = 0; slot < INDEX_BUF_SIZE; slot++)
        {
            if (!pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[slot].valid))
            {
                continue;
            }
            lsm_flush_one_pending(&SharedLSMIndexBuffer->slots[slot].lsmIndex, slot, false);
        }

        pg_usleep(100000L); // sleep 0.1 second
    }
}