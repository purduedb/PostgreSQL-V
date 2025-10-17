#ifndef TASKSEND_H
#define TASKSEND_H

#include "postgres.h"
#include "ringbuffer.h"

void vector_search_send(Oid index_oid, float *query, int dim, Size elem_size, int topk, int efs_nprobe, LSMSnapshot lsm_snapshot);
VectorSearchResult vector_search_get_result(void);
ResultSlot *index_build_blocking(Oid index_relid, int lsm_index_idx);
ResultSlot *index_load_blocking(Oid index_relid, int lsm_index_idx);
ResultSlot *segment_update_blocking(int lsm_index_idx, Oid index_relid, int operation_type, SegmentId start_sid, SegmentId end_sid);

#endif
