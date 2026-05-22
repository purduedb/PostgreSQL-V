#ifndef DPV_SEGMENT_VACUUM_REDO_H
#define DPV_SEGMENT_VACUUM_REDO_H

#include "postgres.h"
#include "replication_rmgr.h"

/*
 * Apply a unified per-sid VacuumTombstones record on the standby
 * (Plan 3 refactor). One record per sid touched by a vacuum batch on
 * the primary; the standby routes to either a segment in the pool or
 * the matching memtable.
 */
extern void dpv_apply_vacuum_tombstones(const xl_dpv_vacuum_tombstones *hdr,
                                        const xl_dpv_vacuum_entry *entries);

#endif
