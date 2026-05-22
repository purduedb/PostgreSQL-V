#ifndef TASKSEND_H
#define TASKSEND_H

#include "postgres.h"
#include "ringbuffer.h"
#include "segment_adoption.h"   /* DpvVacuumGroup */

void vector_search_send(Oid index_oid, float *query, int dim, Size elem_size, int topk, int efs_nprobe, LSMSnapshot lsm_snapshot);
VectorSearchResult vector_search_get_result(void);
void index_build_blocking(Oid index_relid, int lsm_index_idx);
void index_load_blocking(Oid index_relid, int lsm_index_idx);
int segment_update_blocking(int lsm_index_idx, Oid index_relid, int operation_type,
                             SegmentId start_sid, SegmentId end_sid,
                             uint32_t expected_version,
                             /* ADOPT-only; pass NULL/0 for other ops */
                             const SegmentId *memtable_cover,
                             int memtable_cover_count,
                             /* ADOPT-only (Plan 3 refactor); pass NULL/0 otherwise.
                              * Per-sid grouped deletion-tid payload from the
                              * fetcher's cover scan. Groups are packed into the
                              * DSM segment after SegmentUpdateTaskData via
                              * tasksend.c's group serializer. */
                             const DpvVacuumGroup *groups,
                             int n_groups);

/*
 * Wrapper for the fetcher: submit a SEGMENT_UPDATE_ADOPT task and wait for
 * the vector_index_worker to perform adoption. Returns the maint_status
 * code reported by the worker:
 *   0 = ADOPTED, 1 = STALE_DISCARD, 2 = INDEX_UNLOADED.
 *
 * groups / n_groups carries per-sid deletion bits across memtable_cover,
 * captured by the fetcher under mt_lock SHARED + vacuum_lock SHARED. May
 * be NULL/0 for the common no-deletions case.
 */
int dpv_send_adopt_task(int lsm_idx, Oid index_relid,
                        SegmentId start_sid, SegmentId end_sid,
                        uint32_t version,
                        const SegmentId *memtable_cover,
                        int memtable_cover_count,
                        const DpvVacuumGroup *groups,
                        int n_groups);

#endif
