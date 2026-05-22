#ifndef DPV_STANDBY_MEMTABLE_H
#define DPV_STANDBY_MEMTABLE_H

#include "postgres.h"
#include "storage/itemptr.h"
#include "lsm_segment.h"

/*
 * Standby-side effects for redo callbacks. Each returns silently if the
 * targeted index's LSMIndexBufferSlot is not writable (absent or recovery
 * failed). Callers gate on lookup_lsm_index_idx() before invoking these;
 * that lookup waits past LSM_SLOT_RECOVERING and returns -1 on absence or
 * recovery failure.
 */

/* Allocate / find the memtable slot for sid on the standby. */
extern void dpv_standby_register_memtable(Oid indexRelId, SegmentId sid);

/* Materialize one inserted vector from inline WAL data (no heap fetch needed). */
extern void dpv_standby_add_to_memtable(Oid indexRelId, SegmentId sid,
                                         uint32 slot_index, ItemPointer tid,
                                         const void *vector, uint32 vector_bytes);

/* Update the cached max_memtable_sid. */
extern void dpv_standby_update_max_sid(Oid indexRelId, SegmentId sid);

/*
 * `Release` on the standby is intentionally a no-op for SharedMemtableBuffer
 * (see spec §10 — memtable persists until adoption). Provided as a stub so
 * the dispatcher is uniform.
 */
extern void dpv_standby_release_memtable(Oid indexRelId, SegmentId sid);

#endif
