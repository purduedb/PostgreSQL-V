#include "postgres.h"
#include "access/xlog.h"
#include "access/xloginsert.h"
#include "access/xlogreader.h"
#include "access/xlogutils.h"
#include "access/xlog_internal.h"
#include "miscadmin.h"
#include "utils/elog.h"

#include "replication_rmgr.h"
#include "lsmindex.h"
#include "lsmindex_io.h"
#include "standby_memtable.h"
#include "pending_fetch_queue.h"
#include "segment_vacuum_redo.h"

/* Forward decls for redo callbacks (defined below). */
static void redo_register_memtable(XLogReaderState *r);
static void redo_add_to_memtable(XLogReaderState *r);
static void redo_update_max_sid(XLogReaderState *r);
static void redo_release_memtable(XLogReaderState *r);
static void redo_segment_created(XLogReaderState *r);
static void redo_segment_replaced(XLogReaderState *r);
static void redo_index_create(XLogReaderState *r);
static void redo_vacuum_tombstones(XLogReaderState *r);

static void
dpv_replication_redo(XLogReaderState *record)
{
    uint8 info = XLogRecGetInfo(record) & ~XLR_INFO_MASK;

    switch (info)
    {
        case XLOG_DPV_REGISTER_MEMTABLE:  redo_register_memtable(record);  break;
        case XLOG_DPV_ADD_TO_MEMTABLE:    redo_add_to_memtable(record);    break;
        case XLOG_DPV_UPDATE_MAX_SID:     redo_update_max_sid(record);     break;
        case XLOG_DPV_RELEASE_MEMTABLE:   redo_release_memtable(record);   break;
        case XLOG_DPV_SEGMENT_CREATED:    redo_segment_created(record);    break;
        case XLOG_DPV_SEGMENT_REPLACED:   redo_segment_replaced(record);   break;
        case XLOG_DPV_INDEX_CREATE:       redo_index_create(record);       break;
        case XLOG_DPV_VACUUM_TOMBSTONES:   redo_vacuum_tombstones(record);   break;
        default:
            elog(PANIC, "decoupled_pgvector_redo: unknown op code %u", info);
    }
}

/* Optional descriptor for pg_waldump; minimal v1 stub is fine. */
static void
dpv_replication_desc(StringInfo buf, XLogReaderState *record)
{
    uint8 info = XLogRecGetInfo(record) & ~XLR_INFO_MASK;
    appendStringInfo(buf, "dpv op=0x%02x", info);
}

static const char *
dpv_replication_identify(uint8 info)
{
    switch (info & ~XLR_INFO_MASK)
    {
        case XLOG_DPV_REGISTER_MEMTABLE:  return "REGISTER_MEMTABLE";
        case XLOG_DPV_ADD_TO_MEMTABLE:    return "ADD_TO_MEMTABLE";
        case XLOG_DPV_UPDATE_MAX_SID:     return "UPDATE_MAX_SID";
        case XLOG_DPV_RELEASE_MEMTABLE:   return "RELEASE_MEMTABLE";
        case XLOG_DPV_SEGMENT_CREATED:    return "SEGMENT_CREATED";
        case XLOG_DPV_SEGMENT_REPLACED:   return "SEGMENT_REPLACED";
        case XLOG_DPV_INDEX_CREATE:       return "INDEX_CREATE";
        case XLOG_DPV_VACUUM_TOMBSTONES:   return "VACUUM_TOMBSTONES";
        default:                          return NULL;
    }
}

static const RmgrData dpv_replication_rmgr = {
    .rm_name      = RM_DPV_REPLICATION_NAME,
    .rm_redo      = dpv_replication_redo,
    .rm_desc      = dpv_replication_desc,
    .rm_identify  = dpv_replication_identify,
    .rm_startup   = NULL,
    .rm_cleanup   = NULL,
    .rm_mask      = NULL,
    .rm_decode    = NULL,
};

void
vector_replication_rmgr_register(void)
{
    RegisterCustomRmgr(RM_DPV_REPLICATION_ID, &dpv_replication_rmgr);
}

/* ----- Redo callbacks (implemented in Phase 5). ----- */

static void
redo_register_memtable(XLogReaderState *r)
{
    xl_dpv_register_memtable *rec =
        (xl_dpv_register_memtable *) XLogRecGetData(r);

    elog(DEBUG1, "[dpv rmgr] redo REGISTER_MEMTABLE: InHotStandby=%d indexRelId=%u sid=%u lsn=%X/%X",
         InHotStandby ? 1 : 0, rec->indexRelId, (uint32) rec->sid,
         LSN_FORMAT_ARGS(r->ReadRecPtr));

    if (!InHotStandby)
        return;  /* spec §7: skip during primary crash-recovery */
    PG_TRY();
    {
        (void) get_lsm_index_idx(rec->indexRelId, /* for_redo */ true, rec->dbOid);
    }
    PG_CATCH();
    {
        FlushErrorState();
        return;  /* recovery failed; nothing to replay against */
    }
    PG_END_TRY();
    dpv_standby_register_memtable(rec->indexRelId, rec->sid);
}

static void
redo_add_to_memtable(XLogReaderState *r)
{
    char                   *data   = XLogRecGetData(r);
    xl_dpv_add_to_memtable *rec    = (xl_dpv_add_to_memtable *) data;
    const void             *vector = data + sizeof(*rec);

    if (!InHotStandby)
        return;
    PG_TRY();
    {
        (void) get_lsm_index_idx(rec->indexRelId, /* for_redo */ true, rec->dbOid);
    }
    PG_CATCH();
    {
        FlushErrorState();
        return;  /* recovery failed; nothing to replay against */
    }
    PG_END_TRY();
    dpv_standby_add_to_memtable(rec->indexRelId, rec->sid, rec->slot_index,
                                 &rec->tid, vector, rec->vector_bytes);
}

static void
redo_update_max_sid(XLogReaderState *r)
{
    xl_dpv_update_max_sid *rec =
        (xl_dpv_update_max_sid *) XLogRecGetData(r);

    if (!InHotStandby)
        return;
    PG_TRY();
    {
        (void) get_lsm_index_idx(rec->indexRelId, /* for_redo */ true, rec->dbOid);
    }
    PG_CATCH();
    {
        FlushErrorState();
        return;  /* recovery failed; nothing to replay against */
    }
    PG_END_TRY();
    dpv_standby_update_max_sid(rec->indexRelId, rec->sid);
}

static void
redo_release_memtable(XLogReaderState *r)
{
    xl_dpv_release_memtable *rec =
        (xl_dpv_release_memtable *) XLogRecGetData(r);

    elog(DEBUG1, "[dpv rmgr] redo RELEASE_MEMTABLE: InHotStandby=%d indexRelId=%u sid=%u lsn=%X/%X",
         InHotStandby ? 1 : 0, rec->indexRelId, (uint32) rec->sid,
         LSN_FORMAT_ARGS(r->ReadRecPtr));

    if (!InHotStandby)
        return;
    PG_TRY();
    {
        (void) get_lsm_index_idx(rec->indexRelId, /* for_redo */ true, rec->dbOid);
    }
    PG_CATCH();
    {
        FlushErrorState();
        return;  /* recovery failed; nothing to replay against */
    }
    PG_END_TRY();
    dpv_standby_release_memtable(rec->indexRelId, rec->sid);
}

static void
redo_segment_created(XLogReaderState *r)
{
    xl_dpv_segment_created *rec = (xl_dpv_segment_created *) XLogRecGetData(r);
    DpvFetchEntryHeader hdr;

    elog(DEBUG1, "[dpv rmgr] redo SEGMENT_CREATED: InHotStandby=%d indexRelId=%u [%u,%u] v=%u lsn=%X/%X",
         InHotStandby ? 1 : 0, rec->indexRelId,
         (uint32) rec->start_sid, (uint32) rec->end_sid, rec->version,
         LSN_FORMAT_ARGS(r->ReadRecPtr));

    if (!InHotStandby)
        return;  /* primary crash recovery — recover_lsm_index_internal handles it */

    PG_TRY();
    {
        (void) get_lsm_index_idx(rec->indexRelId, /* for_redo */ true, rec->dbOid);
    }
    PG_CATCH();
    {
        FlushErrorState();
        return;  /* recovery failed; nothing to enqueue against */
    }
    PG_END_TRY();

    hdr = (DpvFetchEntryHeader) {
        .indexRelId = rec->indexRelId, .start_sid = rec->start_sid,
        .end_sid    = rec->end_sid,    .version   = rec->version,
        .kind       = DPV_FETCH_KIND_CREATED,
        .source_lsn = r->ReadRecPtr,
        .status     = DPV_FETCH_PENDING,
    };
    (void) dpv_queue_enqueue(&hdr, NULL, 0);
}

static void
redo_segment_replaced(XLogReaderState *r)
{
    char *data = XLogRecGetData(r);
    xl_dpv_segment_replaced *rechdr = (xl_dpv_segment_replaced *) data;
    DpvFetchEntryHeader qhdr;
    Size trailer_size;

    elog(DEBUG1, "[dpv rmgr] redo SEGMENT_REPLACED: InHotStandby=%d indexRelId=%u new=[%u,%u] v=%u lsn=%X/%X",
         InHotStandby ? 1 : 0, rechdr->indexRelId,
         (uint32) rechdr->new_start_sid, (uint32) rechdr->new_end_sid,
         rechdr->new_version, LSN_FORMAT_ARGS(r->ReadRecPtr));

    if (!InHotStandby)
        return;

    PG_TRY();
    {
        (void) get_lsm_index_idx(rechdr->indexRelId, /* for_redo */ true, rechdr->dbOid);
    }
    PG_CATCH();
    {
        FlushErrorState();
        return;  /* recovery failed; nothing to enqueue against */
    }
    PG_END_TRY();

    trailer_size = (Size) rechdr->old_count * sizeof(xl_dpv_seg_range)
                 + (Size) rechdr->offset_count * sizeof(xl_dpv_seg_offset);

    qhdr = (DpvFetchEntryHeader) {
        .indexRelId = rechdr->indexRelId,
        .start_sid  = rechdr->new_start_sid,
        .end_sid    = rechdr->new_end_sid,
        .version    = rechdr->new_version,
        .kind       = DPV_FETCH_KIND_REPLACED,
        .source_lsn = r->ReadRecPtr,
        .status     = DPV_FETCH_PENDING,
    };
    {
        Size expected_total = sizeof(*rechdr) + trailer_size;
        if (XLogRecGetDataLen(r) < expected_total)
            elog(PANIC, "redo_segment_replaced: WAL record too short: "
                        "got %u bytes, expected at least %zu "
                        "(old_count=%u offset_count=%u)",
                        XLogRecGetDataLen(r), (size_t) expected_total,
                        rechdr->old_count, rechdr->offset_count);
    }
    (void) dpv_queue_enqueue(&qhdr, data + sizeof(*rechdr), trailer_size);
}

/*
 * redo_index_create — create the per-index storage directory and write
 * index_metadata locally, then directly claim a LSMIndexBufferSlot in
 * LSM_SLOT_WRITABLE state so subsequent redo callbacks for this index
 * (REGISTER_MEMTABLE, ADD_TO_MEMTABLE, SEGMENT_CREATED, ...) hit
 * get_lsm_index_idx's lock-free fast path and avoid invoking the
 * IndexRecoveryWorker.
 *
 * Why direct claim instead of pre-triggering recovery via
 * get_lsm_index_idx(for_redo=true): the recovery worker would call
 * index_open(AccessShareLock), which blocks on the AccessExclusiveLock
 * primary's CREATE INDEX holds (propagated to the standby via
 * STANDBY_LOCK WAL records). The startup process driving WAL replay is
 * the one that would release that lock — by replaying CREATE INDEX's
 * commit WAL — but it is itself blocked waiting on the recovery worker.
 * Deadlock.
 *
 * Direct claim is sound here because there is nothing to recover for a
 * brand-new index: status pages were just initialized to empty by
 * CreateStatusMetaPage / InitializeStatusMemtableArray (whose WAL
 * replayed just before this record), no segments exist on disk yet, no
 * memtables exist in heap to reconstruct. The slot is populated from
 * fields the WAL record carries. See claim_lsm_index_slot_for_create_redo
 * in lsmindex.c.
 */
static void
redo_index_create(XLogReaderState *r)
{
    xl_dpv_index_create *rec = (xl_dpv_index_create *) XLogRecGetData(r);
    char dir_path[MAXPGPATH];
    LSMIndexData fake_lsm;

    if (!InHotStandby)
        return;  /* primary crash recovery — local files are already present */

    /*
     * Create the index storage directory and write index_metadata locally.
     * write_lsm_index_metadata writes to <storage>/<indexRelId>/index_metadata;
     * we synthesize a temporary LSMIndexData carrying just the fields it
     * needs (indexRelId, index_type, dim, elem_size).
     */
    GetLsmDirPath(dir_path, sizeof(dir_path), rec->indexRelId);
    ensure_dir_exists(dir_path);

    memset(&fake_lsm, 0, sizeof(fake_lsm));
    fake_lsm.indexRelId = rec->indexRelId;
    fake_lsm.index_type = (IndexType) rec->index_type;
    fake_lsm.dim        = rec->dim;
    fake_lsm.elem_size  = rec->elem_size;
    write_lsm_index_metadata(&fake_lsm);

    /*
     * Directly claim the slot in LSM_SLOT_WRITABLE — no recovery worker,
     * no index_open, no deadlock. Errors (e.g., no free slot) are logged
     * inside the helper but not propagated; subsequent redo callbacks will
     * skip their in-memory effect via find_loaded_slot's NULL return, and
     * disk-side state (status pages, segment files) is kept current
     * independently.
     */
    (void) claim_lsm_index_slot_for_create_redo(rec->dbOid,
                                                rec->indexRelId,
                                                rec->index_type,
                                                rec->dim,
                                                rec->elem_size);
}

static void
redo_vacuum_tombstones(XLogReaderState *r)
{
    char                     *data    = XLogRecGetData(r);
    xl_dpv_vacuum_tombstones *hdr     = (xl_dpv_vacuum_tombstones *) data;
    xl_dpv_vacuum_entry      *entries =
        (xl_dpv_vacuum_entry *) (data + sizeof(*hdr));

    if (!InHotStandby)
        return;

    PG_TRY();
    {
        (void) get_lsm_index_idx(hdr->indexRelId, /* for_redo */ true, hdr->dbOid);
    }
    PG_CATCH();
    {
        FlushErrorState();
        return;
    }
    PG_END_TRY();

    dpv_apply_vacuum_tombstones(hdr, entries);
}

/* ----- Emit helpers (primary-side WAL emission). Called from statuspage.c
 *       after all GenericXLog work for the operation is complete.
 *       These records carry ONLY semantic payload; GenericXLog handles all
 *       page-level WAL.  No buffer registration; no PageSetLSN. ----- */

XLogRecPtr
dpv_emit_register_memtable(Oid indexRelId, SegmentId sid)
{
    xl_dpv_register_memtable xlrec = {
        .dbOid      = MyDatabaseId,
        .indexRelId = indexRelId,
        .sid        = sid,
    };

    XLogBeginInsert();
    XLogRegisterData((char *) &xlrec, sizeof(xlrec));
    elog(DEBUG1, "[dpv rmgr] emit REGISTER_MEMTABLE: indexRelId=%u sid=%u",
         indexRelId, (uint32) sid);
    return XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_REGISTER_MEMTABLE);
}

XLogRecPtr
dpv_emit_add_to_memtable(Oid indexRelId, SegmentId sid,
                         uint32 slot_index, ItemPointer tid,
                         const void *vector, uint32 vector_bytes)
{
    xl_dpv_add_to_memtable xlrec = {
        .dbOid        = MyDatabaseId,
        .indexRelId   = indexRelId,
        .sid          = sid,
        .slot_index   = slot_index,
        .tid          = *tid,
        .vector_bytes = vector_bytes,
    };

    XLogBeginInsert();
    XLogRegisterData((char *) &xlrec, sizeof(xlrec));
    if (vector_bytes > 0)
        XLogRegisterData((char *) vector, vector_bytes);
    return XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_ADD_TO_MEMTABLE);
}

XLogRecPtr
dpv_emit_update_max_sid(Oid indexRelId, SegmentId sid)
{
    xl_dpv_update_max_sid xlrec = {
        .dbOid      = MyDatabaseId,
        .indexRelId = indexRelId,
        .sid        = sid,
    };

    XLogBeginInsert();
    XLogRegisterData((char *) &xlrec, sizeof(xlrec));
    return XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_UPDATE_MAX_SID);
}

XLogRecPtr
dpv_emit_release_memtable(Oid indexRelId, SegmentId sid)
{
    xl_dpv_release_memtable xlrec = {
        .dbOid      = MyDatabaseId,
        .indexRelId = indexRelId,
        .sid        = sid,
    };

    XLogBeginInsert();
    XLogRegisterData((char *) &xlrec, sizeof(xlrec));
    elog(DEBUG1, "[dpv rmgr] emit RELEASE_MEMTABLE: indexRelId=%u sid=%u",
         indexRelId, (uint32) sid);
    return XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_RELEASE_MEMTABLE);
}

XLogRecPtr
dpv_emit_segment_created(Oid indexRelId, SegmentId start_sid, SegmentId end_sid,
                         uint32 version)
{
    xl_dpv_segment_created xlrec = {
        .dbOid      = MyDatabaseId,
        .indexRelId = indexRelId, .start_sid = start_sid,
        .end_sid    = end_sid,    .version   = version,
    };

    XLogBeginInsert();
    XLogRegisterData((char *) &xlrec, sizeof(xlrec));
    elog(DEBUG1, "[dpv rmgr] emit SEGMENT_CREATED: indexRelId=%u [%u,%u] v=%u",
         indexRelId, (uint32) start_sid, (uint32) end_sid, version);
    return XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_SEGMENT_CREATED);
}

XLogRecPtr
dpv_emit_segment_replaced(Oid indexRelId,
                          const xl_dpv_seg_range *old_ranges, int old_count,
                          SegmentId new_start_sid, SegmentId new_end_sid,
                          uint32 new_version,
                          const xl_dpv_seg_offset *offsets, int offset_count)
{
    xl_dpv_segment_replaced hdr;

    Assert(old_count    >= 0 && old_count    <= UINT16_MAX);
    Assert(offset_count >= 0 && offset_count <= UINT16_MAX);

    hdr = (xl_dpv_segment_replaced) {
        .dbOid         = MyDatabaseId,
        .indexRelId    = indexRelId,
        .new_start_sid = new_start_sid,
        .new_end_sid   = new_end_sid,
        .new_version   = new_version,
        .old_count     = (uint16) old_count,
        .offset_count  = (uint16) offset_count,
    };

    XLogBeginInsert();
    XLogRegisterData((char *) &hdr, sizeof(hdr));
    if (old_count > 0)
        XLogRegisterData((char *) old_ranges, (Size) old_count * sizeof(xl_dpv_seg_range));
    if (offset_count > 0)
        XLogRegisterData((char *) offsets, (Size) offset_count * sizeof(xl_dpv_seg_offset));
    elog(DEBUG1,
         "[dpv rmgr] emit SEGMENT_REPLACED: indexRelId=%u new=[%u,%u] v=%u old_count=%d offset_count=%d",
         indexRelId, (uint32) new_start_sid, (uint32) new_end_sid,
         new_version, old_count, offset_count);
    return XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_SEGMENT_REPLACED);
}

XLogRecPtr
dpv_emit_index_create(Oid indexRelId, uint32 index_type,
                      uint32 dim, uint32 elem_size)
{
    xl_dpv_index_create xlrec = {
        .dbOid      = MyDatabaseId,
        .indexRelId = indexRelId,
        .index_type = index_type,
        .dim        = dim,
        .elem_size  = elem_size,
    };

    XLogBeginInsert();
    XLogRegisterData((char *) &xlrec, sizeof(xlrec));
    elog(DEBUG1, "[dpv rmgr] emit INDEX_CREATE: indexRelId=%u type=%u dim=%u elem_size=%u",
         indexRelId, index_type, dim, elem_size);
    return XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_INDEX_CREATE);
}

XLogRecPtr
dpv_emit_vacuum_tombstones(Oid indexRelId,
                           SegmentId sid, uint32 owner_version, uint32 subversion,
                           bool is_memtable_owner,
                           const xl_dpv_vacuum_entry *entries, int entry_count)
{
    xl_dpv_vacuum_tombstones hdr;

    Assert(entry_count >= 0);

    hdr = (xl_dpv_vacuum_tombstones) {
        .dbOid             = MyDatabaseId,
        .indexRelId        = indexRelId,
        .sid               = sid,
        .owner_version     = owner_version,
        .subversion        = subversion,
        .is_memtable_owner = is_memtable_owner ? 1u : 0u,
        .entry_count       = (uint32) entry_count,
        ._pad              = 0,
    };

    XLogBeginInsert();
    XLogRegisterData((char *) &hdr, sizeof(hdr));
    if (entry_count > 0)
        XLogRegisterData((char *) entries,
                         (Size) entry_count * sizeof(xl_dpv_vacuum_entry));
    elog(DEBUG1,
         "[dpv rmgr] emit VACUUM_TOMBSTONES: indexRelId=%u sid=%u owner_v=%u sub=%u memtable_owner=%d entries=%d",
         indexRelId, (uint32) sid, owner_version, subversion,
         is_memtable_owner ? 1 : 0, entry_count);
    return XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_VACUUM_TOMBSTONES);
}
