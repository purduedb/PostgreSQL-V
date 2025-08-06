#include "postgres.h"
#include "fmgr.h"                         // For PG_MODULE_MAGIC
#include "miscadmin.h"                    // For MyProcPid, MyDatabaseId, etc.
#include "storage/ipc.h"                  // For on_proc_exit
#include "storage/lwlock.h"               // For LWLock (if you use them)
#include "storage/shmem.h"                // For ShmemInitStruct
#include "storage/dsm.h"                  // For DSM functions (dsm_create, attach, etc.)
#include "storage/pg_shmem.h"             // For shared memory setup
#include "storage/proc.h"                 // For PGPROC and current backend info

#include "postmaster/bgworker.h"         // Required for background worker APIs
#include "postmaster/postmaster.h"       // For BackgroundWorkerStartTime
#include "tcop/tcopprot.h"               // For ReadyForQuery
#include "utils/builtins.h"              // For elog(), etc.
#include "utils/memutils.h"              // For MemoryContext
#include "utils/guc.h"                   // For custom GUCs if any
#include "storage/pmsignal.h"


#include "lsm_background_worker.h"
#include "lsm_index_struct.h"
#include "faiss_index.hpp"


// a global flag to track whether the background worker received a SIGTERM
static volatile sig_atomic_t got_sigterm = false;

static inline bool
OurPostmasterIsAlive(void)
{
	if (likely(!postmaster_possibly_dead))
		return true;
	return PostmasterIsAliveInternal();
}

void
lsm_index_bgworker_main(Datum main_arg)
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

        // pg_usleep(1000000L); // sleep 1 second
        pg_usleep(100000L); // sleep 0.1 second


        for (int slot = 0; slot < INDEX_BUF_SIZE; slot++)
        {
            // elog(DEBUG1, "[lsm_index_bgworker_main] lsm index slot = %d, valid = %d", slot, pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[slot].valid));
            if (!pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[slot].valid))
                continue;

            // elog(DEBUG1, "[lsm_index_bgworker_main] lsm_handle = %d", SharedLSMIndexBuffer->slots[slot].lsm_handle);
            dsm_handle lsm_handle = SharedLSMIndexBuffer->slots[slot].lsm_handle;
            // elog(DEBUG1, "[lsm_index_bgworker_main] got lsm_handle");
            dsm_segment *seg = dsm_find_mapping(lsm_handle);
            if (seg == NULL)
                seg = dsm_attach(lsm_handle);
            // elog(DEBUG1, "[lsm_index_bgworker_main] attached");
            LSMIndex lsm = dsm_segment_address(seg);
            // elog(DEBUG1, "[lsm_index_bgworker_main] got the pointer to the LSMIndexData");

            // TODO: need to add lock for choosing a memtable to flush if there are more than one background workers
            // the background worker processes sealed memtables in order of the segment ids
            uint32_t min_id_pos = -1;
            for (int i = 0; i < DEFAULT_SEALED_MEMTABLE_CAPACITY; i++)
            {
                // elog(DEBUG1, "[lsm_index_bgworker_main] check sealed memtable %d, id = %d", i, pg_atomic_read_u32(&lsm->sealedMemtableIds[i]));
                if (pg_atomic_read_u32(&lsm->sealedMemtableIds[i]) >= START_SEGMENT_ID)
                {
                    if (min_id_pos == -1 || pg_atomic_read_u32(&lsm->sealedMemtableIds[i]) < pg_atomic_read_u32(&lsm->sealedMemtableIds[min_id_pos]))
                        min_id_pos = i;
                }
            }

            if (min_id_pos != -1)
            {
                elog(DEBUG1, "[lsm_index_bgworker_main] sealed memtable slot = %d, min_id = %d, segment_id = %u", slot, min_id_pos, pg_atomic_read_u32(&lsm->sealedMemtableIds[min_id_pos]));
                dsm_segment *sealed_seg = dsm_find_mapping(lsm->sealedMemtables[min_id_pos]);
                if (sealed_seg == NULL)
                sealed_seg = dsm_attach(lsm->sealedMemtables[min_id_pos]);
                ConcurrentMemTable sealed = dsm_segment_address(sealed_seg);

                build_index_and_flush(SharedLSMIndexBuffer->slots[slot].indexRelId, lsm, sealed, min_id_pos);
                pg_memory_barrier();
                pg_atomic_write_u32(&lsm->sealedMemtableIds[min_id_pos], FLUSHED_SEGMENT_ID);
                dsm_detach(sealed_seg);
            }

            dsm_detach(seg);
        }
    }
}

void build_index_and_flush(Oid relId, LSMIndex lsm, ConcurrentMemTable sealed_mt, int sealed_idx)
{
    elog(DEBUG1, "enter build_index_and_flush");

    LWLockAcquire(lsm->lock, LW_EXCLUSIVE);
    
    SegmentData *segment = NULL;
    for (int i = 0; i < DEFAULT_SEGMENT_CAPACITY; i++)
    {
        // TODO: need a lock if there are multiple background writers
        if (!pg_atomic_read_u32(&lsm->segments[i].valid))
        {
            segment = &lsm->segments[i];
            elog(DEBUG1, "[build_index_and_flush] select segment[%d]", i);
            break;
        }
    }
    if (segment == NULL)
    {
        elog(ERROR, "[build_index_and_flush] Segment buffer full, cannot flush");
        return;
    }
    
    // initialize the mapping
    uint32_t vector_count = pg_atomic_read_u32(&sealed_mt->current_size);
    segment->mapSize = sizeof(int64_t) * vector_count;
    dsm_segment *dsm_seg = dsm_create(segment->mapSize, 0);
    if (dsm_seg == NULL)
        elog(ERROR, "[build_index_and_flush] Failed to allocate dynamic shared memory segment");
    dsm_pin_mapping(dsm_seg);
    
    void * mapping_data = dsm_segment_address(dsm_seg);
    memcpy(mapping_data, sealed_mt->tids, segment->mapSize);
    segment->mapping = dsm_segment_handle(dsm_seg);
    segment->segment_mapping_cached_seg = NULL;
    elog(DEBUG1, "[build_index_and_flush] initialized the segment mapping");

    // initialize the bitmap
    segment->bitmapSize = MEMTABLE_BITMAP_SIZE;
    dsm_segment *bitmap_seg = dsm_create(segment->bitmapSize, 0);
    if (bitmap_seg == NULL)
        elog(ERROR, "[build_index_and_flush] Failed to allocate dynamic shared memory segment for bitmap");
    dsm_pin_mapping(bitmap_seg);

    void *bitmap_data = dsm_segment_address(bitmap_seg);
    memcpy(bitmap_data, sealed_mt->bitmap, segment->bitmapSize);
    segment->bitmap = dsm_segment_handle(bitmap_seg);
    segment->segment_bitmap_cached_seg = NULL;
    elog(DEBUG1, "[build_index_and_flush] initialized the segment bitmap");

    dsm_segment *index_seg = NULL; // for ivfflat
    switch (lsm->index_type)
    {
    case IVFFLAT:
    {
        int local_nlist = (int) sqrt(vector_count);
        segment->index = FaissIvfflatBuildAllocate(lsm->dim, local_nlist, get_lsm_memtable_pointer(0, sealed_mt), sealed_mt->start_vid, sealed_mt->start_vid + vector_count - 1, vector_count, &(segment->indexSize), &index_seg);
        segment->segment_index_cached_seg = NULL;
        elog(DEBUG1, "[build_index_and_flush] initialized the segment ivf index");
        break;
    }
    case HNSW:
    {
        int M = 32;
        int efC = 64;
        segment->index = FaissHnswIndexBuildAllocate(lsm->dim, M, efC, get_lsm_memtable_pointer(0, sealed_mt), sealed_mt->start_vid, sealed_mt->start_vid + vector_count - 1, vector_count, &(segment->indexSize), &index_seg);
        segment->segment_index_cached_seg = NULL;
        elog(DEBUG1, "[build_index_and_flush] initialized the segment hnsw index");
        break;
    }
    default:
        break;
    }
    
    // Assign metadata
    // pg_atomic_write_u32(&segment->loaded, 1);
    SegmentId seg_id = pg_atomic_read_u32(&lsm->sealedMemtableIds[sealed_idx]);
    segment->segmentId = seg_id;
    segment->lowestVid = sealed_mt->start_vid;
    segment->highestVid = sealed_mt->start_vid + pg_atomic_read_u32(&sealed_mt->current_size) - 1;
    pg_memory_barrier();
    elog(DEBUG1, "[build_index_and_flush] finish writing the segment, whose segmentId = %d, lowestVid = %d, highestVid = %d", segment->segmentId, segment->lowestVid, segment->highestVid);
    pg_atomic_write_u32(&segment->valid, 1);

    // write index files
    // step1: write the LSM index files
    write_segment_file(relId, seg_id, mapping_data, segment->mapSize, SEGMENT_MAPPING);
    write_segment_file(relId, seg_id, bitmap_data, segment->bitmapSize, SEGMENT_BITMAP);
    write_segment_file(relId, seg_id, dsm_segment_address(index_seg), segment->indexSize, SEGMENT_INDEX);
    // step2: write the LSM metafile
    write_lsm_index_metadata(relId, lsm);

    LWLockRelease(lsm->lock);
}