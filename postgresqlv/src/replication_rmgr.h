/*
 * replication_rmgr.h — custom resource manager for decoupled_pgvector
 *
 * Op codes 0x00–0x40: memtable lifecycle (Plan 1). 0x20 retired.
 * Op codes 0x50–0x60: segment lifecycle  (Plan 2).
 * Op code  0x70:      index create       (Plan 2).
 * Op code  0x80:      unified per-sid vacuum tombstones (Plan 3 refactor).
 *                     (Legacy 0x80 segment-vacuum and 0x20 memtable-tombstone
 *                      records are retired.)
 */
#ifndef DPV_REPLICATION_RMGR_H
#define DPV_REPLICATION_RMGR_H

#include "postgres.h"
#include "access/xlog.h"
#include "access/xlogreader.h"
#include "access/xlog_internal.h"
#include "storage/itemptr.h"
#include "lsm_segment.h"   /* SegmentId */

/*
 * Custom rmgr ID. PG reserves [RM_EXPERIMENTAL_ID, RM_MAX_ID] = [128, 255] for
 * extension use. We pick 137 — outside the upstream contrib rmgr ID range
 * documented in src/include/access/rmgrlist.h.
 */
#define RM_DPV_REPLICATION_ID  137
#define RM_DPV_REPLICATION_NAME "decoupled_pgvector"

/*
 * Info bits live in the upper nibble of xl_info. The low nibble (XLR_INFO_MASK)
 * is reserved for PG core. We dispatch on `info & ~XLR_INFO_MASK`.
 */
#define XLOG_DPV_REGISTER_MEMTABLE     0x00
#define XLOG_DPV_ADD_TO_MEMTABLE       0x10
#define XLOG_DPV_UPDATE_MAX_SID        0x30
#define XLOG_DPV_RELEASE_MEMTABLE      0x40
#define XLOG_DPV_SEGMENT_CREATED       0x50
#define XLOG_DPV_SEGMENT_REPLACED      0x60
#define XLOG_DPV_INDEX_CREATE          0x70
/* Unified per-sid vacuum record (Plan 3 refactor). One record per sid touched. */
#define XLOG_DPV_VACUUM_TOMBSTONES         0x80

/*
 * Per-entry payload for the unified vacuum record. Entries are emitted in
 * ascending sid_local_idx order — i.e. physical offset order within the
 * sid's range in the primary's map_ptr / mt->tids.
 */
typedef struct
{
    uint32  sid_local_idx;
    int64_t tid;
} xl_dpv_vacuum_entry;

/*
 * Variable-length record. One record per sid touched by a vacuum batch.
 * Payload layout on the wire:
 *   header (32 B, 8-aligned)
 *   xl_dpv_vacuum_entry entries[entry_count]
 */
typedef struct
{
    Oid       dbOid;                /*  4 */
    Oid       indexRelId;           /*  4 */
    SegmentId sid;                  /*  4 — the exact sid this batch vacuums. */
    uint32    owner_version;        /*  4 — primary's segment version at emit
                                     *     time; 0 if owner is a memtable.   */
    uint32    subversion;           /*  4 — new subversion for the bitmap
                                     *     file; UINT32_MAX if memtable owner. */
    uint32    is_memtable_owner;    /*  4 — 1 = owner is memtable on primary;
                                     *     0 = owner is a flushed segment.   */
    uint32    entry_count;          /*  4 */
    uint32    _pad;                 /*  4 — keeps total 32 B / 8-aligned.    */
    /* followed by xl_dpv_vacuum_entry entries[entry_count] */
} xl_dpv_vacuum_tombstones;

StaticAssertDecl(sizeof(xl_dpv_vacuum_tombstones) == 32,
                 "xl_dpv_vacuum_tombstones must be exactly 32 bytes");
StaticAssertDecl(sizeof(xl_dpv_vacuum_tombstones) % 8 == 0,
                 "xl_dpv_vacuum_tombstones must be 8-aligned so the entry "
                 "trailer's int64_t tid is naturally aligned");
StaticAssertDecl(sizeof(xl_dpv_vacuum_entry) == 16,
                 "xl_dpv_vacuum_entry must be exactly 16 bytes");
/* 0x90..0xF0 reserved for plan 3 */

/* ---- Record payload structs (the "main data" portion of each record) ----
 * Buffer references (the status page being modified) are attached separately
 * via XLogRegisterBuffer; only the *semantic* fields live here.
 */
typedef struct
{
    Oid       dbOid;       /* MyDatabaseId of the emitter; set by the producer
                            * so the standby's redo callback can drive
                            * per-DB recovery. */
    Oid       indexRelId;
    SegmentId sid;
} xl_dpv_register_memtable;

typedef struct
{
    Oid             dbOid;         /* MyDatabaseId of the emitter; set by the producer
                                    * so the standby's redo callback can drive
                                    * per-DB recovery. */
    Oid             indexRelId;
    SegmentId       sid;
    uint32          slot_index;
    ItemPointerData tid;
    uint32          vector_bytes;  /* size of the vector payload that follows */
    /* followed by `vector_bytes` raw bytes of vector data */
} xl_dpv_add_to_memtable;

typedef struct
{
    Oid       dbOid;       /* MyDatabaseId of the emitter; set by the producer
                            * so the standby's redo callback can drive
                            * per-DB recovery. */
    Oid       indexRelId;
    SegmentId sid;
} xl_dpv_update_max_sid;

typedef struct
{
    Oid       dbOid;       /* MyDatabaseId of the emitter; set by the producer
                            * so the standby's redo callback can drive
                            * per-DB recovery. */
    Oid       indexRelId;
    SegmentId sid;
} xl_dpv_release_memtable;

typedef struct
{
    Oid       dbOid;       /* MyDatabaseId of the emitter; set by the producer
                            * so the standby's redo callback can drive
                            * per-DB recovery. */
    Oid       indexRelId;
    SegmentId start_sid;
    SegmentId end_sid;
    uint32    version;
} xl_dpv_segment_created;

/*
 * Variable-length record. After this header on the wire:
 *   xl_dpv_seg_range  old[old_count]
 *   xl_dpv_seg_offset offsets[offset_count]
 */
typedef struct
{
    SegmentId start_sid;
    SegmentId end_sid;
    uint32    version;
} xl_dpv_seg_range;

typedef struct
{
    SegmentId source_sid;
    uint32    start_offset;
} xl_dpv_seg_offset;

typedef struct
{
    Oid       dbOid;       /* MyDatabaseId of the emitter; set by the producer
                            * so the standby's redo callback can drive
                            * per-DB recovery. */
    Oid       indexRelId;
    SegmentId new_start_sid;
    SegmentId new_end_sid;
    uint32    new_version;
    uint16    old_count;
    uint16    offset_count;
    /* followed by: xl_dpv_seg_range  old[old_count]
     *              xl_dpv_seg_offset offsets[offset_count] */
} xl_dpv_segment_replaced;

/*
 * Emitted by build_lsm_index on the primary BEFORE xl_dpv_segment_created
 * for the initial segment. The standby's redo callback creates the index
 * storage directory and writes index_metadata locally so subsequent
 * redo_segment_created / redo_*_memtable callbacks can drive recovery
 * successfully.
 *
 * dbOid is required because the redo runs in the startup process which
 * has no MyDatabaseId; the IndexRecoveryWorker uses this to connect to
 * the right database.
 */
typedef struct
{
    Oid       dbOid;
    Oid       indexRelId;
    uint32    index_type;
    uint32    dim;
    uint32    elem_size;
} xl_dpv_index_create;

/* Register the rmgr with PG. Called from _PG_init. */
extern void vector_replication_rmgr_register(void);

/* Emit helpers (used on the primary). Each emits a semantic-payload-only WAL
 * record AFTER the corresponding GenericXLog record(s) have already been
 * committed.  No buffer registration; no PageSetLSN. */
extern XLogRecPtr dpv_emit_register_memtable(Oid indexRelId, SegmentId sid);
extern XLogRecPtr dpv_emit_add_to_memtable(Oid indexRelId, SegmentId sid,
                                           uint32 slot_index, ItemPointer tid,
                                           const void *vector, uint32 vector_bytes);
extern XLogRecPtr dpv_emit_update_max_sid(Oid indexRelId, SegmentId sid);
extern XLogRecPtr dpv_emit_release_memtable(Oid indexRelId, SegmentId sid);

extern XLogRecPtr dpv_emit_segment_created(Oid indexRelId,
                                            SegmentId start_sid, SegmentId end_sid,
                                            uint32 version);

extern XLogRecPtr dpv_emit_segment_replaced(Oid indexRelId,
                                             const xl_dpv_seg_range *old_ranges, int old_count,
                                             SegmentId new_start_sid, SegmentId new_end_sid,
                                             uint32 new_version,
                                             const xl_dpv_seg_offset *offsets, int offset_count);

extern XLogRecPtr dpv_emit_index_create(Oid indexRelId,
                                         uint32 index_type,
                                         uint32 dim,
                                         uint32 elem_size);

extern XLogRecPtr dpv_emit_vacuum_tombstones(
    Oid indexRelId,
    SegmentId sid, uint32 owner_version, uint32 subversion,
    bool is_memtable_owner,
    const xl_dpv_vacuum_entry *entries, int entry_count);

#endif
