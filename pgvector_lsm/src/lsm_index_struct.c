#include "lsm_index_struct.h"
#include "index_storage_manager.h"
#include "storage/shmem.h"
#include "storage/dsm.h"
#include "storage/fd.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "access/heapam.h"

#include "utils/memutils.h"
#include "storage/bufmgr.h"
#include "visibility.h"

#ifndef O_DIRECTORY
#define O_DIRECTORY 0
#endif

LSMIndexBuffer *SharedLSMIndexBuffer = NULL;

static bool test_consistency(Relation heapRel, LSMIndex lsm_index, VectorId max_vid);
static void recover_lsm_index(Relation index, LSMIndex lsm_index, VectorId start_vid);

// get the pointer to the local memory based on the cached segment
static void*
get_pointer_from_cached_segment(dsm_handle handle, dsm_segment **seg)
{
    if (*seg == NULL)
    {
        *seg = dsm_attach(handle);
        dsm_pin_mapping(*seg);
    }
    return dsm_segment_address(*seg); 
}

// convert between ItemPointer and Int64
int64_t
ItemPointerToInt64(const ItemPointer tid)
{
    BlockNumber blkno = BlockIdGetBlockNumber(&(tid->ip_blkid));
    OffsetNumber posid = tid->ip_posid;

    // Combine into a single 64-bit signed integer
    return ((int64_t) blkno << 16) | (uint16_t) posid;
}

ItemPointerData
Int64ToItemPointer(int64_t encoded)
{
    ItemPointerData tid;

    BlockNumber blkno = (BlockNumber) ((encoded >> 16) & 0xFFFFFFFF);
    OffsetNumber posid = (OffsetNumber) (encoded & 0xFFFF);

    BlockIdSet(&tid.ip_blkid, blkno);
    tid.ip_posid = posid;

    return tid;
}


static void
ensure_segment_directory_exists(Oid indexRelId)
{
    char path[MAXPGPATH];
    get_lsm_dir_path(path, sizeof(path), indexRelId); // You might already have a helper for this

    struct stat st;
    if (stat(path, &st) != 0)
    {
        if (mkdir(path, S_IRWXU) != 0)
        {
            elog(ERROR, "Failed to create segment directory: %s", path);
        }
    }
    else if (!S_ISDIR(st.st_mode))
    {
        elog(ERROR, "Path exists but is not a directory: %s", path);
    }
}

static size_t estimate_deserialized_index_size(size_t serialized_size_bytes)
{
    // Conservative overhead factor (e.g., 1.8x the serialized size)
    const double overhead_factor = 1.8;

    // Multiply and round up
    size_t estimated = (size_t)ceil(serialized_size_bytes * overhead_factor);

    // Round up to next multiple of 8 (for 8-byte alignment)
    estimated = (estimated + 7) & ~(size_t)7;

    return estimated;
}

// write and load files
#define MAX_FILE_SIZE (1L * 1024L * 1024L * 1024L) // 2GB (file size limitation)

void
write_segment_file(Oid indexRelId, SegmentId segmentId,
                   const void *data, Size size, SegmentFileKind kind)
{
    elog(DEBUG1, "enter write_segment_file, kind = %d, size = %ld", kind, size);
    ensure_segment_directory_exists(indexRelId);

    char path[MAXPGPATH];
    const char *type;

    /* Determine file type string */
    switch (kind)
    {
        case SEGMENT_INDEX:
            type = "index";
            break;
        case SEGMENT_BITMAP:
            type = "bitmap";
            break;
        case SEGMENT_MAPPING:
            type = "mapping";
            break;
        default:
            elog(ERROR, "Unknown SegmentFileKind: %d", (int) kind);
    }

    /* Generate base file path */
    get_lsm_segment_path(path, sizeof(path), indexRelId, segmentId, type);

    /* Write the data in chunks */
    Size written_size = 0;
    int chunk_index = 0;
    while (written_size < size) {
        char tmp_path[MAXPGPATH];
        snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d", path, chunk_index);

        int fd = OpenTransientFile(tmp_path, O_CREAT | O_WRONLY | O_TRUNC);
        if (fd < 0)
            elog(ERROR, "Failed to create temporary file: %s", tmp_path);

        Size chunk_size = (size - written_size > MAX_FILE_SIZE) ? MAX_FILE_SIZE : size - written_size;

        Size write_size;
        if ((write_size = write(fd, (char *)data + written_size, chunk_size)) != chunk_size)
            elog(ERROR, "Failed to write full content to temporary file: %s, should write %ld but wrote %ld instead", tmp_path, chunk_size, write_size);

        if (fsync(fd) != 0)
            elog(ERROR, "fsync failed on temporary file: %s", tmp_path);

        CloseTransientFile(fd);
        written_size += chunk_size;
        chunk_index++;
    }

    /* Rename the temporary files to final destination */
    for (int i = 0; i < chunk_index; i++) {
        char tmp_path[MAXPGPATH], final_path[MAXPGPATH];
        snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d", path, i);
        snprintf(final_path, sizeof(final_path), "%s.%d", path, i);

        if (rename(tmp_path, final_path) != 0)
            elog(ERROR, "Failed to rename %s to %s", tmp_path, final_path);
    }

    /* Fsync the directory to ensure durability */
    char dir_path[MAXPGPATH];
    strlcpy(dir_path, path, sizeof(dir_path));
    get_parent_directory(dir_path);

    int dirfd = open(dir_path, O_RDONLY | O_DIRECTORY);
    if (dirfd >= 0)
    {
        fsync(dirfd);
        close(dirfd);
    }
}

void
load_segment_file(Oid indexRelId, SegmentData *segment, uint32_t segment_id, SegmentFileKind kind)
{
    char path[MAXPGPATH];
    int fd;
    struct stat st;
    Size size;
    dsm_segment *dsm_seg;
    void *dest;

    /* Generate base file path based on kind */
    switch (kind)
    {
        case SEGMENT_INDEX:
            get_lsm_segment_index_path(path, sizeof(path), indexRelId, segment_id);
            break;
        case SEGMENT_BITMAP:
            get_lsm_segment_bitmap_path(path, sizeof(path), indexRelId, segment_id);
            break;
        case SEGMENT_MAPPING:
            get_lsm_segment_mapping_path(path, sizeof(path), indexRelId, segment_id);
            break;
        default:
            elog(ERROR, "Unknown SegmentFileKind: %d", (int)kind);
    }

    /* Initialize total_size to 0 */
    Size total_size = 0;
    int chunk_index = 0;

    /* First, iterate all chunk files and calculate total size */
    while (true) {
        char chunk_path[MAXPGPATH];
        snprintf(chunk_path, sizeof(chunk_path), "%s.%d", path, chunk_index);

        fd = OpenTransientFile(chunk_path, O_RDONLY | PG_BINARY);
        if (fd < 0)
        {
            if (chunk_index == 0)
                elog(ERROR, "Could not open FAISS segment file: %s", path);
            break; // No more chunks to read
        }

        /* Stat file */
        if (fstat(fd, &st) < 0)
            elog(ERROR, "Could not stat FAISS segment file: %s", chunk_path);

        size = st.st_size;
        total_size += size;

        CloseTransientFile(fd);
        chunk_index++;
    }


    /* Now, allocate a large memory block to hold all the chunks */
    dsm_seg = dsm_create(total_size, 0);
    if (!dsm_seg)
        elog(ERROR, "Failed to allocate dynamic shared memory segment");

    /* Pin the segment to ensure it's mapped in memory */
    dsm_pin_mapping(dsm_seg);
    dest = dsm_segment_address(dsm_seg);
    
    int chunk_num = chunk_index;
    Size offset = 0;
    for (int chunk_index = 0; chunk_index < chunk_num; chunk_index++)
    {
        char chunk_path[MAXPGPATH];
        snprintf(chunk_path, sizeof(chunk_path), "%s.%d", path, chunk_index);

        fd = OpenTransientFile(chunk_path, O_RDONLY | PG_BINARY);
        if (fd < 0)
        {
            elog(ERROR, "Could not open FAISS segment file: %s", path);
        }

        if (fstat(fd, &st) < 0)
            elog(ERROR, "Could not stat FAISS segment file: %s", chunk_path);

        size = st.st_size;

        /* Read the chunk into the allocated memory at the appropriate offset */
        if (read(fd, (char *)dest + offset, size) != size)
            elog(ERROR, "Failed to read complete segment file into memory: %s", chunk_path);

        offset += size; // Update the offset for the next chunk

        CloseTransientFile(fd);
    }
    
    switch (kind)
    {
        case SEGMENT_INDEX:
            segment->indexSize = total_size;
            segment->index = dsm_segment_handle(dsm_seg);
            segment->segment_index_cached_seg = dsm_seg;
            break;
        case SEGMENT_BITMAP:
            segment->bitmapSize = total_size;
            segment->bitmap = dsm_segment_handle(dsm_seg);
            segment->segment_bitmap_cached_seg = dsm_seg;
            break;
        case SEGMENT_MAPPING:
            segment->mapSize = total_size;
            segment->mapping = dsm_segment_handle(dsm_seg);
            segment->segment_mapping_cached_seg = dsm_seg;
            break;
        default:
            elog(ERROR, "Unknown SegmentFileKind (should not happen)");
    }
}

// ------------------------------------------------------------------------------------------------
 
// intialize the shared lsm index buffer
void 
lsm_index_buffer_shmem_initialize()
{
    elog(DEBUG1, "enter lsm_index_buffer_shmem_initialize");

    bool found;

    SharedLSMIndexBuffer = (LSMIndexBuffer *)
        ShmemInitStruct("LSM Index Buffer",
                        sizeof(LSMIndexBuffer),
                        &found);

    if (!found)
    {
        for (int i = 0; i < INDEX_BUF_SIZE; i++)
        {
            pg_atomic_write_u32(&SharedLSMIndexBuffer->slots[i].valid, 0);
            SharedLSMIndexBuffer->slots[i].indexRelId = InvalidOid;
            SharedLSMIndexBuffer->slots[i].lsm_handle = 0;
            SharedLSMIndexBuffer->slots[i].lsm_index_cached_seg = NULL;
        }
    }
}

void
allocate_new_memtable(LSMIndex lsm_index, VectorId start_vid, uint32 dim, uint32 elem_size)
{
    elog(DEBUG1, "enter allocate_new_memtable");
    elog(DEBUG1, "[allocate_new_memtable] start_vid = %d", start_vid);

    Size vector_data_size = dim * elem_size * MEMTABLE_MAX_SIZE;
    Size total_size = MAXALIGN(sizeof(ConcurrentMemTableData)) + MAXALIGN(vector_data_size);

    dsm_segment *seg = dsm_create(total_size, 0);
    if (seg == NULL)
        elog(ERROR, "Failed to allocate dynamic shared memory segment");
    dsm_pin_mapping(seg);  // ensure it stays mapped in this backend (We should not call this if we want to allow multiple backends write the memtable)

    ConcurrentMemTable memtable = (ConcurrentMemTable) dsm_segment_address(seg);

    MemSet(memtable, 0, sizeof(ConcurrentMemTableData));

    lsm_index->memtableID = pg_atomic_add_fetch_u32(&lsm_index->currentSegmentId, 1);
    
    memtable->start_vid = start_vid;
    memtable->max_size = MEMTABLE_MAX_SIZE;
    pg_atomic_init_u32(&memtable->current_size, 0);
    // initialize the vector metadata
    memtable->dim = dim;
    memtable->elem_size = elem_size;
    
    // for (uint32 i = 0; i < MEMTABLE_MAX_SIZE; i++)
    // {
    //     pg_atomic_init_u32(&memtable->written[i], 0);
    // }
    MemSet(memtable->tids, 0, sizeof(uint64_t) * MEMTABLE_MAX_SIZE);
    MemSet(memtable->bitmap, 0, sizeof(uint64_t) * MEMTABLE_BITMAP_SIZE);
    
    dsm_handle handle = dsm_segment_handle(seg);
    lsm_index->memtable = handle;

    lsm_index->memtableCachedSeg = seg;
}

static void
load_lsm_index(Relation index, uint32_t slot_num)
{
    Relation relId = index->rd_id;
    
    elog(DEBUG1, "enter load_lsm_index: relId = %d, slot_num = %d", relId, slot_num);

    // in this prototype, we treat every shutdown as a crash
    bool need_recovery = true;

    SharedLSMIndexBuffer->slots[slot_num].indexRelId = relId;
    // initialize LSMIndexData
    dsm_segment *lsm_index_data_dsm_seg = dsm_create(sizeof(LSMIndexData), 0);
    if (lsm_index_data_dsm_seg == NULL)
        elog(ERROR, "Failed to allocate dynamic shared memory segment");
    
    dsm_pin_mapping(lsm_index_data_dsm_seg);
    SharedLSMIndexBuffer->slots[slot_num].lsm_handle = dsm_segment_handle(lsm_index_data_dsm_seg);
    SharedLSMIndexBuffer->slots[slot_num].lsm_index_cached_seg = lsm_index_data_dsm_seg;

    LSMIndex lsm_index = (LSMIndex) dsm_segment_address(lsm_index_data_dsm_seg);
    MemSet(lsm_index, 0, sizeof(LSMIndexData));
    for (size_t i = 0; i < DEFAULT_SEGMENT_CAPACITY; i++)
    {
        pg_atomic_write_u32(&lsm_index->segments[i].valid, 0);
    }

    bool found;
    // lsm_index->memtable_lock = (LWLock*) ShmemInitStruct("LSM Memtable Lock", sizeof(LWLock), &found);
    // if (!found) LWLockInitialize(lsm_index->memtable_lock, LWLockNewTrancheId());

    // construct full path to metadata file
    char metadata_path[MAXPGPATH];
    get_lsm_metadata_path(metadata_path, sizeof(metadata_path), relId);
    elog(LOG, "Metadata file path: %s", metadata_path);

    // load the lsm index metadata from disk
    FILE *metafile = fopen(metadata_path, "rb");
    if (!metafile)
    {
        elog(ERROR, "Metadata file path: %s cannot open", metadata_path);
    }
    fread(&lsm_index->index_type, sizeof(IndexType), 1, metafile);
    fread(&lsm_index->dim, sizeof(uint32_t), 1, metafile);
    fread(&lsm_index->elem_size, sizeof(uint32_t), 1, metafile);
    uint32_t segmentNum;
    fread(&segmentNum, sizeof(uint32_t), 1, metafile);
    // fread(&lsm_index->currentSegmentId, sizeof(pg_atomic_uint32), 1, metafile);
    uint32_t max_segment_id = 0;
    int64_t max_vid = -1;
    for (uint32_t i = 0; i < segmentNum; i++)
    {
        fread(&lsm_index->segments[i].segmentId, sizeof(uint32_t), 1, metafile);
        fread(&lsm_index->segments[i].lowestVid, sizeof(int64_t), 1, metafile);
        fread(&lsm_index->segments[i].highestVid, sizeof(int64_t), 1, metafile);
        load_segment_file(relId, &(lsm_index->segments[i]), lsm_index->segments[i].segmentId, SEGMENT_INDEX);
        load_segment_file(relId, &(lsm_index->segments[i]), lsm_index->segments[i].segmentId, SEGMENT_MAPPING);
        if (!need_recovery)
        {
            load_segment_file(relId, &(lsm_index->segments[i]), lsm_index->segments[i].segmentId, SEGMENT_BITMAP);
        }
        else
        {
            // allocate a new bitmap
            Size bm_size = GET_BITMAP_SIZE(lsm_index->segments[i].highestVid - lsm_index->segments[i].lowestVid + 1);
            dsm_segment *bm_seg = dsm_create(bm_size, 0);
            if (!bm_seg)
                elog(ERROR, "Failed to allocate dynamic shared memory segment");
            dsm_pin_mapping(bm_seg);
            uint8 *bitmap = dsm_segment_address(bm_seg);
            MemSet(bitmap, 0, bm_size);
            lsm_index->segments[i].bitmapSize = bm_size;
            lsm_index->segments[i].bitmap = dsm_segment_handle(bm_seg);
            lsm_index->segments[i].segment_bitmap_cached_seg = bm_seg;
        }
        pg_atomic_write_u32(&lsm_index->segments[i].valid, 1);
        // pg_atomic_write_u32(&lsm_index->segments[i].loaded, 0);
        max_segment_id = Max(max_segment_id, lsm_index->segments[i].segmentId);
        max_vid = Max(max_vid, lsm_index->segments[i].highestVid);
    }
    pg_atomic_write_u32(&lsm_index->currentSegmentId, max_segment_id);
    fclose(metafile);

    // initialize sealed memtable slots and the memtable
    for (size_t i = 0; i < DEFAULT_SEALED_MEMTABLE_CAPACITY; i++)
    {
        pg_atomic_write_u32(&lsm_index->sealedMemtableIds[i], INVALID_SEGMENT_ID);
        lsm_index->sealedMemtables[i] = 0;
    }
    
    // initialize the lock
    lsm_index->lock = &GetNamedLWLockTranche(LSM_LOCK_TRANCHE_NAME)[slot_num];

    if (need_recovery)
    {
        recover_lsm_index(index, lsm_index, max_vid + 1);
    }
    if (lsm_index->memtable == 0)
    {
        allocate_new_memtable(lsm_index, max_vid + 1, lsm_index->dim, lsm_index->elem_size);
    }

    // mark the slot as valid
    pg_atomic_write_u32(&SharedLSMIndexBuffer->slots[slot_num].valid, 1);
}


static LSMIndex 
get_lsm_index(Relation index)
{
    Oid indexRelId = index->rd_id;

    // check if it's already in the buffer
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        if (pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid) && SharedLSMIndexBuffer->slots[i].indexRelId == indexRelId)
        {
            return (LSMIndex) get_pointer_from_cached_segment(SharedLSMIndexBuffer->slots[i].lsm_handle, &(SharedLSMIndexBuffer->slots[i].lsm_index_cached_seg));
        }
    }

    elog(DEBUG1, "[get_lsm_index] request lsm_index is not in the buffer");
    // find an empty slot
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        if (!pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid))
        {
            elog(DEBUG1, "[get_lsm_index] loading the requested lsm_index to slot[%d]", i);
            load_lsm_index(index, i);
            return (LSMIndex) get_pointer_from_cached_segment(SharedLSMIndexBuffer->slots[i].lsm_handle, &(SharedLSMIndexBuffer->slots[i].lsm_index_cached_seg));
        }
    }
    
    // no free slot - randomly evict one
    elog(ERROR, "[get_lsm_index] no free slot in the buffer");
    return NULL;
}

// for background worker
static dsm_handle 
get_lsm_index_background(Relation index)
{
    // elog(DEBUG1, "enter get_lsm_index: indexRelId = %d", indexRelId);

    Oid indexRelId = index->rd_id;

    // check if it's already in the buffer
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        if (pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid) && SharedLSMIndexBuffer->slots[i].indexRelId == indexRelId)
        {
            // elog(DEBUG1, "[get_lsm_index] request lsm_index is already in the buffer");
            return SharedLSMIndexBuffer->slots[i].lsm_handle;
        }
    }

    elog(ERROR, "[get_lsm_index_background] request lsm_index is not in the buffer");
    return NULL;
}

Pointer 
get_lsm_memtable_pointer(uint32_t pos, ConcurrentMemTable mt)
{
    Pointer p = (Pointer)mt + MAXALIGN(sizeof(ConcurrentMemTableData)) + (Size) pos * mt->dim * mt->elem_size;
    return p;
}

VectorId
insert_lsm_index(Relation index, const void *vector, int64_t tid)
{
    // elog(DEBUG1, "enter insert_lsm_index");

    VectorId vid;

    LSMIndex lsm_index = get_lsm_index(index);

    if (lsm_index->memtable == 0)
    {
        elog(ERROR, "no memtable to insert");
    }

    ConcurrentMemTable mt = (ConcurrentMemTable) get_pointer_from_cached_segment(lsm_index->memtable, &(lsm_index->memtableCachedSeg));

    uint32_t dim, elem_size;
    dim = mt->dim;
    elem_size = mt->elem_size;

    // FIXME: there is potential race on current_size if we plan to support multiple backends
    if (pg_atomic_read_u32(&mt->current_size) < mt->max_size)
    {
        uint32 pos = pg_atomic_fetch_add_u32(&mt->current_size, 1);
        vid = mt->start_vid + pos;

        // if (pos >= mt->max_size)
        //     goto retry;

        memcpy(get_lsm_memtable_pointer(pos, mt), vector, dim * elem_size);
        mt->tids[pos] = tid;
        SET_SLOT(mt->bitmap, pos);
        pg_write_barrier();
        // pg_atomic_write_u32(&mt->written[pos], 1);
        // elog(DEBUG1, "[insert_lsm_index] inserted the vector to the memtable");
    }

    // memtable is full, must seal and swap
    // LWLockAcquire(&mt->lock, LW_EXCLUSIVE);
    if (pg_atomic_read_u32(&mt->current_size) == mt->max_size )
    {
        elog(DEBUG1, "[insert_lsm_index] the current growing memtable is full");
        VectorId next_vid = mt->max_size + mt->start_vid;
        elog(DEBUG1, "[insert_lsm_index] mt->max_size = %d, mt->start_vid = %d, next_vid = %d", mt->max_size, mt->start_vid, next_vid);

        // LWLockAcquire(lsm_index->memtable_lock, LW_EXCLUSIVE);
        // append to sealed list
        // wait until there's a free slot in sealed_memtables[]
        bool inserted = false;
        while (!inserted)
        {
            for (int i = 0; i < DEFAULT_SEALED_MEMTABLE_CAPACITY; i++)
            {
                elog(DEBUG1, "[insert_lsm_index] check sealedMemtableIds[%d]", i);
                if (pg_atomic_read_u32(&lsm_index->sealedMemtableIds[i]) <= FLUSHED_SEGMENT_ID)
                {
                    elog(DEBUG1, "[insert_lsm_index] selected sealedMemtableIds[%d]", i);
                    if (pg_atomic_read_u32(&lsm_index->sealedMemtableIds[i]) == FLUSHED_SEGMENT_ID)
                    {
                        elog(DEBUG1, "[insert_lsm_index] about to clean sealedMemtableIds[%d]", i);
                        // TODO: do we need to explicitly call dsm_destroy? Also, need to ensure concurrency if we plan to support multiple backends
                        dsm_detach(lsm_index->sealedCachedSegs[i]);
                        lsm_index->sealedCachedSegs[i] = NULL;
                        lsm_index->sealedMemtables[i] = 0;
                        pg_atomic_write_u32(&lsm_index->sealedMemtableIds[i], INVALID_SEGMENT_ID);
                        elog(DEBUG1, "[insert_lsm_index] freed sealedMemtableIds[%d]", i);
                    }
                    
                    lsm_index->sealedMemtables[i] = lsm_index->memtable;
                    lsm_index->sealedCachedSegs[i] = lsm_index->memtableCachedSeg;
                    pg_write_barrier();
                    pg_atomic_write_u32(&lsm_index->sealedMemtableIds[i], lsm_index->memtableID);
                    inserted = true;
                    elog(DEBUG1, "[insert_lsm_index] inserted the sealed memtable");
                    break;
                }
            }
            if (!inserted)
            {
                // Optionally release lock to avoid deadlock or blocking other threads
                // LWLockRelease(lsm_index->memtable_lock);
                pg_usleep(1000);  // sleep 1ms
                // LWLockAcquire(lsm_index->memtable_lock, LW_EXCLUSIVE);
            }
        }
        allocate_new_memtable(lsm_index, next_vid, dim, elem_size);
        elog(DEBUG1, "[insert_lsm_index] returned from allocate_new_memtable");
        // LWLockRelease(lsm_index->memtable_lock);
    }
    // LWLockRelease(&mt->lock);
    return vid;
}

/*
    Called in `build_lsm_index`
    Ensure Crash Consistency:
        - Write to a temporary file
        - Flush all buffers to disk
        - Atomically rename the file to replace the old metadata
        - Optionally fsnyc() the directory
*/
void
write_lsm_index_metadata(Oid relId, LSMIndex lsm_index)
{
    ensure_segment_directory_exists(relId);

    char tmp_path[MAXPGPATH];
    char final_path[MAXPGPATH];
    char dir_path[MAXPGPATH];
    get_lsm_metadata_path(final_path, sizeof(final_path), relId);
    get_lsm_metadata_tmp_path(tmp_path, sizeof(tmp_path), relId);
    get_lsm_dir_path(dir_path, sizeof(dir_path), relId);

    FILE *fp = fopen(tmp_path, "wb");
    if (!fp)
        elog(ERROR, "Cannot open temp metadata file for writing: %s", tmp_path);

    // Write metadata
    fwrite(&lsm_index->index_type, sizeof(IndexType), 1, fp);
    fwrite(&lsm_index->dim, sizeof(uint32_t), 1, fp);
    fwrite(&lsm_index->elem_size, sizeof(uint32_t), 1, fp);

    uint32_t segment_count = 0;
    for (uint32_t i = 0; i < DEFAULT_SEGMENT_CAPACITY; i++) {
        if (pg_atomic_read_u32(&lsm_index->segments[i].valid))
            segment_count++;
    }
    fwrite(&segment_count, sizeof(uint32_t), 1, fp);

    for (uint32_t i = 0; i < DEFAULT_SEGMENT_CAPACITY; i++) {
        if (!pg_atomic_read_u32(&lsm_index->segments[i].valid))
            continue;
        fwrite(&lsm_index->segments[i].segmentId, sizeof(uint32_t), 1, fp);
        fwrite(&lsm_index->segments[i].lowestVid, sizeof(uint64_t), 1, fp);
        fwrite(&lsm_index->segments[i].highestVid, sizeof(uint64_t), 1, fp);
    }

    // Flush file buffers to disk
    fflush(fp);
    int fd = fileno(fp);
    if (fd != -1)
        fsync(fd);
    fclose(fp);

    // Rename atomically
    if (rename(tmp_path, final_path) != 0)
        elog(ERROR, "Failed to rename temp metadata file to final location: %s → %s", tmp_path, final_path);

    // Optional: fsync the directory to ensure rename is persisted
    int dirfd = open(dir_path, O_DIRECTORY | O_RDONLY);
    if (dirfd != -1) {
        fsync(dirfd);
        close(dirfd);
    }

    elog(LOG, "Crash-safe LSM index metadata written to: %s", final_path);
}

int
build_lsm_index(IndexType type, Oid relId, void *vector_index, int64_t *tids, uint32_t dim, uint32_t elem_size, VectorId lowest_vid, VectorId highest_vid)
{
    elog(DEBUG1, "enter build_lsm_index, type = %d", type);

    // find an empty slot
    int slot_num = -1;
    for (int i = 0; i < INDEX_BUF_SIZE; i++)
    {
        if (!pg_atomic_read_u32(&SharedLSMIndexBuffer->slots[i].valid))
        {
            slot_num = i;
            break;
        }
    }
    if (slot_num == -1)
    {
        // no free slot - randomly evict one
        elog(ERROR, "[build_lsm_index] no free slot");
    }
    elog(DEBUG1, "[build_lsm_index] slot_num = %d", slot_num);

    SharedLSMIndexBuffer->slots[slot_num].indexRelId = relId;

    // initialize LSMIndexData
    dsm_segment *lsm_index_data_dsm_seg = dsm_create(sizeof(LSMIndexData), 0);
    if (lsm_index_data_dsm_seg == NULL)
        elog(ERROR, "[build_lsm_index] Failed to allocate dynamic shared memory segment");
    dsm_pin_mapping(lsm_index_data_dsm_seg);
    SharedLSMIndexBuffer->slots[slot_num].lsm_handle = dsm_segment_handle(lsm_index_data_dsm_seg);
    // cache the local memory segment pointer
    SharedLSMIndexBuffer->slots[slot_num].lsm_index_cached_seg = lsm_index_data_dsm_seg;
    elog(DEBUG1, "[build_lsm_index] initialized LSMIndexData");

    LSMIndex lsm_index = (LSMIndex) dsm_segment_address(lsm_index_data_dsm_seg);
    
    // initialize the metadata of the vector
    lsm_index->index_type = type;
    lsm_index->dim = dim;
    lsm_index->elem_size = elem_size;
    elog(DEBUG1, "[build_lsm_index] dim = %d, elem_size = %d", dim, elem_size);
    
    // initialize segment management data
    pg_atomic_write_u32(&lsm_index->currentSegmentId, START_SEGMENT_ID);
    // lsm_index->segmentNum = 1; // initialize with 1 index segment

    // initialize segment valid flags
    for (size_t i = 0; i < DEFAULT_SEGMENT_CAPACITY; i++)
    {
        pg_atomic_write_u32(&lsm_index->segments[i].valid, 0);
    }

    // initialize the first segment
    lsm_index->segments[0].segmentId = START_SEGMENT_ID;
    lsm_index->segments[0].lowestVid = lowest_vid;
    lsm_index->segments[0].highestVid = highest_vid;
    elog(DEBUG1, "[build_lsm_index] segmentId = %d, lowestVid = %ld, highestVid = %ld", lsm_index->segments[0].segmentId, lowest_vid, highest_vid);

    // initialize the segment index
    switch (type)
    {
    case IVFFLAT:
        lsm_index->segments[0].index = FaissIvfflatIndexStore(vector_index, &lsm_index->segments[0].indexSize, &(lsm_index->segments[0].segment_index_cached_seg));
        break;
    case HNSW:
        lsm_index->segments[0].index = FaissHnswIndexStore(vector_index, &lsm_index->segments[0].indexSize, &(lsm_index->segments[0].segment_index_cached_seg));
        break;
    default:
        break;
    }
    elog(DEBUG1, "[build_lsm_index] initialized the segment index, indexSize = %ld", lsm_index->segments[0].indexSize);

    // initialize the bitmap
    uint32_t vector_count = highest_vid - lowest_vid + 1;
    lsm_index->segments[0].bitmapSize = GET_BITMAP_SIZE(vector_count);
    elog(DEBUG1, "[build_lsm_index] vector_count = %d, bitmapSize = %d", vector_count, lsm_index->segments[0].bitmapSize);
    dsm_segment *bitmap_seg = dsm_create(lsm_index->segments[0].bitmapSize, 0);
    if (bitmap_seg == NULL)
        elog(ERROR, "[build_lsm_index] Failed to allocate dynamic shared memory segment for bitmap");
    dsm_pin_mapping(bitmap_seg);
    void *bitmap_data = dsm_segment_address(bitmap_seg);
    MemSet(bitmap_data, 0xFF, lsm_index->segments[0].bitmapSize); // initialize all bits to 1
    // If current_size is not a multiple of 8, mask off the extra bits in the last byte
    int remaining_bits = vector_count % 8;
    if (remaining_bits != 0)
    {
        uint8_t *last_byte = &((uint8_t *) bitmap_data)[lsm_index->segments[0].bitmapSize - 1];
        *last_byte &= (1 << remaining_bits) - 1;  // Mask upper bits to zero
    }
    lsm_index->segments[0].bitmap = dsm_segment_handle(bitmap_seg);
    lsm_index->segments[0].segment_bitmap_cached_seg = bitmap_seg;
    elog(DEBUG1, "[build_lsm_index] initialized the segment bitmap");

    // initialize the mapping
    lsm_index->segments[0].mapSize = sizeof(int64_t) * vector_count;
    dsm_segment *map_dsm_seg = dsm_create(lsm_index->segments[0].mapSize, 0);
    if (map_dsm_seg == NULL)
        elog(ERROR, "[build_lsm_index] Failed to allocate dynamic shared memory segment");
    dsm_pin_mapping(map_dsm_seg);
    void * mapping_data = dsm_segment_address(map_dsm_seg);
    memcpy(mapping_data, tids, lsm_index->segments[0].mapSize);
    lsm_index->segments[0].mapping = dsm_segment_handle(map_dsm_seg);
    lsm_index->segments[0].segment_mapping_cached_seg = map_dsm_seg;
    elog(DEBUG1, "[build_lsm_index] initialized the segment mapping");

    pg_write_barrier();
    // pg_atomic_write_u32(&lsm_index->segments[0].loaded, 1);
    pg_atomic_write_u32(&lsm_index->segments[0].valid, 1);

    // initialize memtables
    for (size_t i = 0; i < DEFAULT_SEALED_MEMTABLE_CAPACITY; i++)
    {
        pg_atomic_write_u32(&lsm_index->sealedMemtableIds[i], INVALID_SEGMENT_ID);
        lsm_index->sealedMemtables[i] = 0;
    }
    allocate_new_memtable(lsm_index, highest_vid + 1, lsm_index->dim, lsm_index->elem_size);
    elog(DEBUG1, "[build_lsm_index] allocated a new memtable");

    // initialize the lock
    // LWLockInitialize(&lsm_index->lock, LSM_TRANCHE_ID);
    lsm_index->lock = &GetNamedLWLockTranche(LSM_LOCK_TRANCHE_NAME)[slot_num];
    
    // mark the slot as valid
    pg_write_barrier();
    pg_atomic_write_u32(&SharedLSMIndexBuffer->slots[slot_num].valid, 1);
    elog(DEBUG1, "[build_lsm_index] mark the slot as valid");

    // write index files
    // step1: write the LSM index files
    SegmentId seg_id = lsm_index->segments[0].segmentId;
    write_segment_file(relId, seg_id, mapping_data, lsm_index->segments[0].mapSize, SEGMENT_MAPPING);
    write_segment_file(relId, seg_id, bitmap_data, lsm_index->segments[0].bitmapSize, SEGMENT_BITMAP);
    write_segment_file(relId, seg_id, get_pointer_from_cached_segment(lsm_index->segments[0].index, 
        &(lsm_index->segments[0].segment_index_cached_seg)), lsm_index->segments[0].indexSize, SEGMENT_INDEX);
    // step2: write the LSM metafile
    write_lsm_index_metadata(relId, lsm_index);
}

// Comparison function for sorting DistancePair by distance
static int 
compare_distance(const void *a, const void *b)
{
    DistancePair *pairA = (DistancePair*)a;
    DistancePair *pairB = (DistancePair*)b;

    if (pairA->distance < pairB->distance)
        return -1;
    else if (pairA->distance > pairB->distance)
        return 1;
    else
        return 0;
}

// pairs_1 and pairs_2 should be sorted, return_pairs should be pre-allocated
static int 
merge_top_k(DistancePair *pairs_1, DistancePair *pairs_2, int num_1, int num_2, int top_k, DistancePair *merge_pair)
{
    // elog(DEBUG1, "enter merge_top_k");

    int i = 0, j = 0, k = 0;

    while (k < top_k && (i < num_1 || j < num_2)) {
        if (i < num_1 && (j >= num_2 || pairs_1[i].distance <= pairs_2[j].distance)) {
            merge_pair[k++] = pairs_1[i++];
        } else if (j < num_2) {
            merge_pair[k++] = pairs_2[j++];
        }
    }

    return k;
}

// the ids in the returned pairs are tids
static int
search_memtable(ConcurrentMemTable mt, const float *query_vector, int top_k, DistancePair *top_k_pairs)
{
    // elog(DEBUG1, "enter search_memtable");

    // compute distances
    uint32_t vector_num = pg_atomic_read_u32(&mt->current_size);
    float *distances = palloc(sizeof(float) * vector_num);
    FaissComputeDistances(get_lsm_memtable_pointer(0, mt), vector_num, mt->dim, query_vector, distances);

    DistancePair *distance_pairs = palloc(sizeof(DistancePair) * vector_num);
    int valid_vector_count = 0;  // counter for valid (visible) vectors
    for (uint32_t i = 0; i < vector_num; i++) {
        // check if the vector is marked as visible (bitmap[i] is set)
        if (IS_SLOT_SET(mt->bitmap, i)) {
            distance_pairs[valid_vector_count].distance = distances[i];
            distance_pairs[valid_vector_count].id = mt->tids[i];

            valid_vector_count++;
        }
    }

    // Sort only the valid distance pairs based on distance (ascending order)
    qsort(distance_pairs, valid_vector_count, sizeof(DistancePair), compare_distance);

    // Store the top K closest vectors
    for (int i = 0; i < top_k && i < valid_vector_count; i++) {
        top_k_pairs[i].distance = distance_pairs[i].distance;
        top_k_pairs[i].id = distance_pairs[i].id;
    }

    // Free the temporary memory
    pfree(distance_pairs);
    pfree(distances);

    // Set the number of results
    int pair_num = (valid_vector_count < top_k) ? valid_vector_count : top_k;
    // elog(DEBUG1, "[search_memtable] return %d", pair_num);
    return pair_num;
}

TopKTuples
search_lsm_index(Relation index, const float *query_vector, int top_k, int nprobe_efs)
{
    // elog(DEBUG1, "enter search_lsm_index: relId = %d, top_k = %d", index->rd_id, top_k);

    LSMIndex lsm_index = get_lsm_index(index);

    // int tmp_top_k = 2 * top_k;
    int tmp_top_k = top_k;
    
    // allocate distance pairs for sorting
    // int all_valid_vector_count = 0;
    DistancePair *final_pairs, *pair_1;
    int num_1;

    // step 1: search the growing memtable
    // elog(DEBUG1, "[search_lsm_index]step 1: search the growing memtable");
    ConcurrentMemTable mt = (ConcurrentMemTable) get_pointer_from_cached_segment(lsm_index->memtable, &(lsm_index->memtableCachedSeg));
    // allocate the DistancePair structure
    DistancePair *mt_pairs = palloc(sizeof(DistancePair) * tmp_top_k);
    // conduct search
    num_1 = search_memtable(mt, query_vector, tmp_top_k, mt_pairs);
    pair_1 = mt_pairs;

    // step 2: search sealed memtables
    // elog(DEBUG1, "[search_lsm_index] step 2: search sealed memtables");
    // record the minimum sealed memtable ID
    SegmentId min_sealed_mt_id = INVALID_SEGMENT_ID;
    for (size_t i = 0; i < DEFAULT_SEALED_MEMTABLE_CAPACITY; i++)
    {
        uint32_t sealed_mt_id = pg_atomic_read_u32(&lsm_index->sealedMemtableIds[i]);
        if (sealed_mt_id >= START_SEGMENT_ID)
        {
            // elog(DEBUG1, "[search_lsm_index] search the sealed memtable whose id = %d", sealed_mt_id);
            // record the minimum segment ID
            if (min_sealed_mt_id == 0 || sealed_mt_id < min_sealed_mt_id)
            {
                min_sealed_mt_id = sealed_mt_id;
            }
            // search the sealed memtable
            ConcurrentMemTable sealed_mt = (ConcurrentMemTable) get_pointer_from_cached_segment(lsm_index->sealedMemtables[i], &(lsm_index->sealedCachedSegs[i]));
            DistancePair *sealed_mt_pairs = palloc(sizeof(DistancePair) * tmp_top_k);
            int sealed_mt_pair_num = search_memtable(sealed_mt, query_vector, tmp_top_k, sealed_mt_pairs);
            // merge
            final_pairs = palloc(sizeof(DistancePair) * tmp_top_k);
            int merge_num = merge_top_k(pair_1, sealed_mt_pairs, num_1, sealed_mt_pair_num, tmp_top_k, final_pairs);
            pfree(pair_1);
            pfree(sealed_mt_pairs);
            num_1 = merge_num;
            pair_1 = final_pairs;
        }
    }

    // int segment_top_k = 2 * top_k;
    int segment_top_k = top_k;
    // search flushed segments
    // elog(DEBUG1, "[search_lsm_index] step 3: search flushed segments");
    for (size_t i = 0; i < DEFAULT_SEGMENT_CAPACITY; i++)
    {
        if (pg_atomic_read_u32(&lsm_index->segments[i].valid) && (min_sealed_mt_id == INVALID_SEGMENT_ID || lsm_index->segments[i].segmentId < min_sealed_mt_id))
        {
            // elog(DEBUG1, "[search_lsm_index] search segment %d, index size = %ld", lsm_index->segments[i].segmentId, lsm_index->segments[i].indexSize);
            topKVector *segment_result;
            switch (lsm_index->index_type)
            {
            case IVFFLAT:
                segment_result = FaissIvfflatIndexSearch(get_pointer_from_cached_segment(lsm_index->segments[i].index, &(lsm_index->segments[i].segment_index_cached_seg)), 
                                                        lsm_index->segments[i].index, lsm_index->segments[i].indexSize, query_vector, segment_top_k, nprobe_efs);
                break;
            case HNSW:
            {
                segment_result = FaissHnswIndexSearch(get_pointer_from_cached_segment(lsm_index->segments[i].index, &(lsm_index->segments[i].segment_index_cached_seg)), 
                                                      lsm_index->segments[i].index, lsm_index->segments[i].indexSize, query_vector, segment_top_k, nprobe_efs);
                break;
            }
            default:
                break;
            }
            DistancePair *segment_pairs = palloc(sizeof(DistancePair) * tmp_top_k);
            int valid_num = 0;
            
            uint8 *bitmap = (uint8 *) get_pointer_from_cached_segment(lsm_index->segments[i].bitmap, &(lsm_index->segments[i].segment_bitmap_cached_seg));
            int64_t *mapping = (int64_t *) get_pointer_from_cached_segment(lsm_index->segments[i].mapping, &(lsm_index->segments[i].segment_mapping_cached_seg));
            // elog(DEBUG1, "[search_lsm_index] got pointers of the bitmap and the mapping");
            
            for (size_t j = 0; j < segment_result->num_results; j++)
            {
                int pos = segment_result->vids[j] - lsm_index->segments[i].lowestVid;
                // elog(DEBUG1, "[search_lsm_index] segment_result->vids[j] = %d, lsm_index->segments[i].lowestVid = %d, pos = %d", segment_result->vids[j], lsm_index->segments[i].lowestVid, pos);
                if (IS_SLOT_SET(bitmap, pos))
                {
                    segment_pairs[valid_num].distance = segment_result->distances[j];
                    segment_pairs[valid_num].id = mapping[pos];
                    ++valid_num;

                    // Stop if we've reached tmp_top_k
                    if (valid_num == tmp_top_k) {
                        break;
                    }
                }
            }
            // elog(DEBUG1, "[search_lsm_index] got the top_k tuples");

            // merge
            final_pairs = palloc(sizeof(DistancePair) * tmp_top_k);
            int merge_num = merge_top_k(pair_1, segment_pairs, num_1, valid_num, tmp_top_k, final_pairs);
            // elog(DEBUG1, "[search_lsm_index] returned from merge_top_k");
            pfree(pair_1);
            pfree(segment_pairs);
            num_1 = merge_num;
            pair_1 = final_pairs;

            // Free the memory for this segment result
            pfree(segment_result);
        }
    }

    TopKTuples top_k_tuples = {num_1, pair_1};
    elog(DEBUG1, "[search_lsm_index] num_1 = %d", num_1);
    return top_k_tuples;
}

static int
compare_vector_id(const void *a, const void *b)
{
    VectorId va = *(const VectorId *)a;
    VectorId vb = *(const VectorId *)b;

    if (va < vb)
        return -1;
    else if (va > vb)
        return 1;
    else
        return 0;
}

typedef struct sealed_mt_pair {
    SegmentId mt_id;
    int slot_id;
} sealed_mt_pair;

// Comparator: sort by mt_id (ascending)
int
compare_sealed_mt_pair(const void *a, const void *b)
{
    const sealed_mt_pair *pa = (const sealed_mt_pair *)a;
    const sealed_mt_pair *pb = (const sealed_mt_pair *)b;
    return (pa->mt_id > pb->mt_id) - (pa->mt_id < pb->mt_id);
}

void bulk_delete_lsm_index(Relation index, VectorId *vec_ids, int delete_num)
{
    // sort vec_ids
    qsort(vec_ids, delete_num, sizeof(VectorId), compare_vector_id);

    dsm_handle lsm_hdl = get_lsm_index_background(index);
    dsm_segment *lsm_seg = dsm_find_mapping(lsm_hdl);
    if (lsm_seg == NULL)
        lsm_seg = dsm_attach(lsm_hdl);
    LSMIndex lsm_index = dsm_segment_address(lsm_seg);

    // TODO: finer granularity of the lock
    LWLockAcquire(lsm_index->lock, LW_EXCLUSIVE);

    int i = 0; // index into vec_ids
    // iterate all flushed segments
    for (int seg_idx = 0; seg_idx < DEFAULT_SEGMENT_CAPACITY && i < delete_num; seg_idx++)
    {
        SegmentData *seg_data = &lsm_index->segments[seg_idx];
        
        // skip unused segment slots
        if (pg_atomic_read_u32(&seg_data->valid) == 0)
            continue;
        
        VectorId seg_lo = seg_data->lowestVid;
        VectorId seg_hi = seg_data->highestVid;

        if (i >= delete_num || vec_ids[i] > seg_hi)
            continue;
        
        dsm_segment *bitmap_seg = dsm_find_mapping(seg_data->bitmap);
        if (bitmap_seg == NULL)
            bitmap_seg = dsm_attach(seg_data->bitmap);
        uint8_t *bitmap = (uint8_t *) dsm_segment_address(bitmap_seg);

        // delete all vec_ids that fall in this segment
        while (i < delete_num && vec_ids[i] >= seg_lo && vec_ids[i] <= seg_hi)
        {
            int slot_idx = (int)(vec_ids[i] - seg_lo);
            CLEAR_SLOT(bitmap, slot_idx);
            ++i;
        }
        // dsm_detach(bitmap_seg);
    }
    
    // iterate all sealed memtables
    sealed_mt_pair sealed_list[DEFAULT_SEALED_MEMTABLE_CAPACITY];
    int sealed_count = 0;

    for (int j = 0; j < DEFAULT_SEALED_MEMTABLE_CAPACITY; j++)
    {
        SegmentId id = pg_atomic_read_u32(&lsm_index->sealedMemtableIds[j]);
        if (id != INVALID_SEGMENT_ID)
        {
            sealed_list[sealed_count].mt_id = id;
            sealed_list[sealed_count].slot_id = j;
            sealed_count++;
        }
    }

    // sort sealed memtables based on the VectorId
    qsort(sealed_list, sealed_count, sizeof(sealed_mt_pair), compare_sealed_mt_pair);

    for (int j = 0; j < sealed_count && i < delete_num; j++)
    {
        int idx = sealed_list[j].slot_id;
    
        dsm_segment *mt_seg = dsm_find_mapping(lsm_index->sealedMemtables[idx]);
        if (mt_seg == NULL)
            mt_seg = dsm_attach(lsm_index->sealedMemtables[idx]);
        ConcurrentMemTable mt = (ConcurrentMemTable) dsm_segment_address(mt_seg);

        VectorId mt_lo = mt->start_vid;
        VectorId mt_hi = mt->start_vid + pg_atomic_read_u32(&mt->current_size) - 1;

        while (i < delete_num && vec_ids[i] < mt_lo)
        {
            elog(ERROR, "[bulk_delete_lsm_index] unfound deleted vid");
            i++;
        }
        if (i >= delete_num || vec_ids[i] > mt_hi)
            continue;

        while (i < delete_num && vec_ids[i] >= mt_lo && vec_ids[i] <= mt_hi)
        {
            int slot_idx = (int)(vec_ids[i] - mt_lo);
            CLEAR_SLOT(mt->bitmap, slot_idx);
            i++;
        }
        // dsm_detach(mt_seg);
    }

    // iterate the growing memtable
    if (i < delete_num)
    {
        dsm_segment *mt_seg = dsm_find_mapping(lsm_index->memtable);
        if (mt_seg == NULL)
            mt_seg = dsm_attach(lsm_index->memtable);
        ConcurrentMemTable mt = (ConcurrentMemTable) dsm_segment_address(mt_seg);
        VectorId mt_lo = mt->start_vid;
        VectorId mt_hi = mt->start_vid + mt->max_size - 1;

        while (i < delete_num && vec_ids[i] >= mt_lo && vec_ids[i] <= mt_hi)
        {
            int slot_idx = (int)(vec_ids[i] - mt_lo);
            CLEAR_SLOT(mt->bitmap, slot_idx);
            i++;
        }
        // dsm_detach(mt_seg);
    }

    if (i < delete_num)
    {
        elog(ERROR, "[bulk_delete_lsm_index] bulk delete is not processed completely");
    }

    LWLockRelease(lsm_index->lock);
    // dsm_detach(lsm_seg);
}

typedef struct vid_hid_pair
{
    VectorId vid;
    ItemPointer hid;
} vid_hid_pair;

int compare_vid_hid_pair(const void *a, const void *b) {
    const vid_hid_pair *pair_a = (const vid_hid_pair *)a;
    const vid_hid_pair *pair_b = (const vid_hid_pair *)b;

    // Compare based on the vid field
    if (pair_a->vid < pair_b->vid) {
        return -1; // a < b
    } else if (pair_a->vid > pair_b->vid) {
        return 1;  // a > b
    } else {
        return 0;  // a == b
    }
}

static int get_largest_set_slot(uint8_t *bitmap, uint32_t max_size)
{
    int last_byte = (max_size - 1) / 8;

    for (int byte_index = last_byte; byte_index >= 0; byte_index--)
    {
        uint8_t byte = bitmap[byte_index];
        if (byte == 0)
            continue;

        // Find the most significant bit set in this byte
        for (int bit = 7; bit >= 0; bit--)
        {
            if (byte & (1 << bit))
            {
                int pos = (byte_index << 3) + bit;
                if ((uint32_t)pos < max_size)
                    return pos;
            }
        }
    }

    // No bit is set
    return -1;
}

/*
    the lsm_index should be preloaded before calling `recover_lsm_index`
    the bitmap of the flushed segment should not be set before calling this function
*/ 
static void 
recover_lsm_index(Relation index, LSMIndex lsm_index, VectorId start_vid)
{
    elog(DEBUG1, "enter recover_lsm_index");

    TimestampTz start_time = GetCurrentTimestamp();

    LWLockAcquire(lsm_index->lock, LW_EXCLUSIVE);
    dsm_handle mt_hdls[DEFAULT_SEALED_MEMTABLE_CAPACITY + 1];
    dsm_segment *mt_segs[DEFAULT_SEALED_MEMTABLE_CAPACITY + 1];
    for (int i = 0; i < DEFAULT_SEALED_MEMTABLE_CAPACITY + 1; i++)
    {
        mt_hdls[i] = 0;
        mt_segs[i] = NULL;
    }
    VectorId current_start_vid = start_vid;
    uint32_t dim = lsm_index->dim;
    uint32_t elem_size = lsm_index->elem_size;
    int mt_num = 0;
    
    BufferAccessStrategy bas = GetAccessStrategy(BAS_BULKREAD);
    BlockNumber searchPage = InvalidBlockNumber;
	GetVisibilityStartPage(index, &searchPage);
	BlockNumber insertPage = InvalidBlockNumber;
    Oid relId = index->rd_index->indrelid;
    AttrNumber attnum = index->rd_index->indkey.values[0];
    // TODO: double check the concurrency issue here
    Relation heapRel = table_open(relId, AccessShareLock);

    // create the index tuple descriptor
	TupleDesc vitupdesc = CreateTemplateTupleDesc(1);
	TupleDescInitEntry(vitupdesc, (AttrNumber) 1, "vid", INT8OID, -1, 0);
	Datum vid_datum;
	bool isnull;
    VectorId max_vid = start_vid;

    // iterate visibility files
    while (BlockNumberIsValid(searchPage))
	{
		Buffer buf;
		Page page;
		OffsetNumber maxoffno;
        VectorId segment_vids[MaxOffsetNumber];
        vid_hid_pair mt_pairs[MaxOffsetNumber];
        int mt_vnum = 0;
        int segment_vnum = 0;

        buf = ReadBufferExtended(index, MAIN_FORKNUM, searchPage, RBM_NORMAL, bas);
        LockBuffer(buf, BUFFER_LOCK_SHARE);
        page = BufferGetPage(buf);
        maxoffno = PageGetMaxOffsetNumber(page);

        for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			IndexTuple itup = (IndexTuple) PageGetItem(page, PageGetItemId(page, offno));
			vid_datum = index_getattr(itup, 1, vitupdesc, &isnull);
            VectorId vid = DatumGetInt64(vid_datum);
            
            // elog(DEBUG1, "[recover_lsm_index] live visibility vid = %ld", vid);
            if (vid < start_vid)
            {
                segment_vids[segment_vnum++] = vid;
            }
            else
            {
			    ItemPointer htup = &(itup->t_tid);
                mt_pairs[mt_vnum].vid = vid;
                mt_pairs[mt_vnum++].hid = htup;
            }
        }
        searchPage = IvfflatPageGetOpaque(page)->nextblkno;

        // FIXME: we can increase the batch to improve the performance
        // set bitmap for vectors in flushed segments
        if (segment_vnum > 0)
        {
            // sort vids in ascending order
            qsort(segment_vids, segment_vnum, sizeof(VectorId), compare_vector_id);
            int i = 0; // index into segment_vids
            for (int seg_idx = 0; seg_idx < DEFAULT_SEGMENT_CAPACITY && i < segment_vnum; seg_idx++)
            {
                SegmentData *seg_data = &lsm_index->segments[seg_idx];

                if (pg_atomic_read_u32(&seg_data->valid) == 0)
                    continue;

                VectorId seg_lo = seg_data->lowestVid;
                VectorId seg_hi = seg_data->highestVid;

                if (i >= segment_vnum || segment_vids[i] > seg_hi)
                    continue;
                
                uint8_t *bitmap = (uint8_t *) get_pointer_from_cached_segment(seg_data->bitmap, &(seg_data->segment_bitmap_cached_seg));

                // delete all vec_ids that fall in this segment
                while (i < segment_vnum && segment_vids[i] >= seg_lo && segment_vids[i] <= seg_hi)
                {
                    int slot_idx = (int)(segment_vids[i] - seg_lo);
                    SET_SLOT(bitmap, slot_idx);
                    ++i;
                }
            }
            // elog(DEBUG1, "[recover_lsm_index] processed flushed vectors");
        }

        // insert unflushed vectors to memtables
        TupleTableSlot *slot = table_slot_create(heapRel, NULL);
        TupleDesc tupDesc = RelationGetDescr(heapRel);
        // sort the memtable pairs
        qsort(mt_pairs, mt_vnum, sizeof(vid_hid_pair), compare_vid_hid_pair);
        // elog(DEBUG1, "[recover_lsm_index] unflushed vectors num = %d", mt_vnum);
        if (mt_vnum > 0)
        {
            int i = 0; // index into mt_pairs
            for (int mt_idx = 0; mt_idx < DEFAULT_SEALED_MEMTABLE_CAPACITY + 1 && i < mt_vnum; mt_idx++)
            {
                ConcurrentMemTable mt = NULL;

                // initialize the memtable
                if (mt_hdls[mt_idx] == 0)
                {
                    Size vector_data_size = dim * elem_size * MEMTABLE_MAX_SIZE;
                    Size total_size = MAXALIGN(sizeof(ConcurrentMemTableData)) + MAXALIGN(vector_data_size);

                    dsm_segment *new_mt_seg = dsm_create(total_size, 0);
                    if (new_mt_seg == NULL)
                        elog(ERROR, "Failed to allocate dynamic shared memory segment"); 
                    mt_segs[mt_idx] = new_mt_seg;
                    dsm_pin_mapping(mt_segs[mt_idx]);
                    mt_hdls[mt_idx] = dsm_segment_handle(new_mt_seg);

                    mt = (ConcurrentMemTable) dsm_segment_address(mt_segs[mt_idx]);
                    MemSet(mt, 0, sizeof(ConcurrentMemTableData));
                    
                    mt->start_vid = current_start_vid;
                    current_start_vid += current_start_vid + MEMTABLE_MAX_SIZE;
                    mt->max_size = MEMTABLE_MAX_SIZE; 
                    pg_atomic_init_u32(&mt->current_size, 0);
                    mt->dim = dim;
                    mt->elem_size = elem_size;

                    MemSet(mt->tids, 0, sizeof(uint64_t) * MEMTABLE_MAX_SIZE);
                    MemSet(mt->bitmap, 0, sizeof(uint64_t) * MEMTABLE_BITMAP_SIZE);
                    
                    mt_num = mt_idx + 1;

                    elog(DEBUG1, "[recover_lsm_index] initialized a new memtable");
                }

                if (mt == NULL)
                {
                    mt = get_pointer_from_cached_segment(mt_hdls[mt_idx], &mt_segs[mt_idx]);
                }

                VectorId mt_lo = mt->start_vid;
                VectorId mt_hi = mt->start_vid + mt->max_size - 1;
                
                if (i >= mt_vnum || mt_pairs[i].vid > mt_hi)
                    continue;
                
                while (i < mt_vnum && mt_pairs[i].vid >= mt_lo && mt_pairs[i].vid <= mt_hi)
                {
                    int slot_idx = (int)(mt_pairs[i].vid - mt_lo);
                    SET_SLOT(mt->bitmap, slot_idx);
                    mt->tids[slot_idx] = ItemPointerToInt64(mt_pairs[i].hid);
                    
                    // FIXME: optimize the performance
                    // fetch the corresponding vector data
                    if (!table_tuple_fetch_row_version(heapRel, mt_pairs[i].hid, SnapshotAny, slot))
                    {
                        // CLEAR_SLOT(mt->bitmap, slot_idx);
                        elog(DEBUG1, "failed to fetch tuple for LSM index recovery due to the heap pruning, vid = %ld", mt_pairs[i].vid); 
                    }
                    
                    // extract the vector attribute
                    bool isnull;
                    Datum vector_datum = slot_getattr(slot, attnum, &isnull);
                    if (isnull)
                        elog(ERROR, "failed to fetch the vector data in the LSM index recovery phase");
                      
                    // ensure we get a detoasted, safe-to-access value
                    Vector *vec = (Vector *) PG_DETOAST_DATUM_COPY(vector_datum);
                    // Now you can use vec->x or vec->dim, depending on pgvector's internal format
                    memcpy(get_lsm_memtable_pointer(slot_idx, mt), vec->x, dim * elem_size);

                    ++i;
                }
            }
            // elog(DEBUG1, "[recover_lsm_index] processed unflushed vectors");
        }

        ExecDropSingleTupleTableSlot(slot);
        UnlockReleaseBuffer(buf);
    }

    elog(DEBUG1, "[recover_lsm_index] mt_num = %d", mt_num);
    // assign memtables to the LSM index
    for (int mt_idx = 0; mt_idx < mt_num; mt_idx++)
    {
        if (mt_idx < mt_num - 1)
        {
            ConcurrentMemTable mt = get_pointer_from_cached_segment(mt_hdls[mt_idx], &mt_segs[mt_idx]);
            pg_atomic_write_u32(&mt->current_size, mt->max_size);
            lsm_index->sealedMemtables[mt_idx] = mt_hdls[mt_idx];
            SegmentId id_tmp = pg_atomic_add_fetch_u32(&lsm_index->currentSegmentId, 1);
            pg_atomic_write_u32(&lsm_index->sealedMemtableIds[mt_idx], id_tmp);
            lsm_index->sealedCachedSegs[mt_idx] = mt_segs[mt_idx];
        }
        else
        {
            ConcurrentMemTable mt = get_pointer_from_cached_segment(mt_hdls[mt_idx], &mt_segs[mt_idx]);
            pg_atomic_write_u32(&mt->current_size, get_largest_set_slot(mt->bitmap, mt->max_size) + 1);
            lsm_index->memtableID = pg_atomic_add_fetch_u32(&lsm_index->currentSegmentId, 1);
            lsm_index->memtable = mt_hdls[mt_idx];
            lsm_index->memtableCachedSeg = mt_segs[mt_idx];
            max_vid = mt->start_vid + pg_atomic_read_u32(&mt->current_size) - 1;
        }
    }
    elog(DEBUG1, "[recover_lsm_index] assigned memtables to the LSM index");

    TimestampTz end_time = GetCurrentTimestamp();
    long elapsed_ms = TimestampDifferenceMilliseconds(start_time, end_time);
    elog(DEBUG1, "[recover_lsm_index] complete the recovery in %ld ms", elapsed_ms);

    // check the consistency post recovery
    // if (test_consistency(heapRel, lsm_index, max_vid))
    // {
    //     elog(DEBUG1, "[recover_lsm_index] passed the consistency check");
    // }
    // else
    // {
    //     elog(ERROR, "[recover_lsm_index] failed to pass the consistency check");
    // }
    
    pg_atomic_write_u32(&lsm_index->recovered, 1);
    LWLockRelease(lsm_index->lock);
    table_close(heapRel, AccessShareLock);
}

static int64_t 
get_next_set_slot(const uint8_t *bitmap, int64_t pos, int64_t max_slots)
{
    int64_t i = pos + 1;
    for (; i < max_slots; i++)
    {
        if (IS_SLOT_SET(bitmap, i))
            return i;
    }
    elog(DEBUG1, "[get_next_set_slot] return -1, no set slot found");
    return -1;  // No set slot found
}

// called in the recover_lsm_index function (heapRel is already opened and the lock of the lsm_index is acquired)
static bool 
test_consistency(Relation heapRel, LSMIndex lsm_index, VectorId max_vid)
{
    elog(DEBUG1, "enter test_consistency, max_vid = %ld", max_vid);
    
    size_t bitmap_size = max_vid + 1;
    // dynamically allocate memory for the bitmaps instead of using VLAs
    uint8_t *bitmap_any = malloc(GET_BITMAP_SIZE(bitmap_size));
    uint8_t *bitmap_visible = malloc(GET_BITMAP_SIZE(bitmap_size));
    if (!bitmap_any || !bitmap_visible) {
        elog(ERROR, "Memory allocation failed for bitmaps");
        return false;
    }
    MemSet(bitmap_any, 0, GET_BITMAP_SIZE(bitmap_size));
    MemSet(bitmap_visible, 0, GET_BITMAP_SIZE(bitmap_size));
    int any_num = 0;
    int visible_num = 0;

    // get the current snapshot to check visibility
    Snapshot snapshot = GetActiveSnapshot();
    // return all physical tuples (including dead tuples)
    HeapScanDesc scan = (HeapScanDesc) heap_beginscan(heapRel, SnapshotAny, 0, NULL, NULL, 0);

    HeapTuple tuple;
    elog(DEBUG1, "[test_consistency] begin scanning the heap table");
    while ((tuple = heap_getnext(scan, ForwardScanDirection)) != NULL)
    {
        Datum datum;
        bool isnull;

        // check if tuple is alive under the active snapshot
        bool visible = HeapTupleSatisfiesVisibility(tuple, snapshot, scan->rs_cbuf);

        // get first column value
        datum = heap_getattr(tuple, 1, RelationGetDescr(heapRel), &isnull);
        if (isnull)
        {
            elog(ERROR, "[test_consistency] the first column of the heap tuple is NULL");
        }
        else
        {
            int32 id = DatumGetInt32(datum);
            // set bitmap_visible
            if (visible)
            {
                if (id - 1 > max_vid)
                {
                    elog(DEBUG1, "[test_consistency] inconsistency state: id > max_vid");
                    return false;
                }
                // PRIMARY KEY start from 1 but our vid start from 0
                SET_SLOT(bitmap_visible, id - 1);
                ++visible_num;
            }
            
            // set bitmap_any
            if (id - 1 <= max_vid)
            {
                // ignore dead tuples beyond the range of vids
                SET_SLOT(bitmap_any, id - 1);
                ++any_num;
            }
        }
    }
    elog(DEBUG1, "[test_consistency] set the bitmap_any and the bitmap_visible, any_num = %d, visible_num = %d", any_num, visible_num);

    // 1. all vectors in the LSM vector index can be found in the heap table (regardless of the visibility)
    // 2. all visible vectors in the heap table can be find in the LSM vector index
    VectorId bm_visible_vid = get_next_set_slot(bitmap_visible, -1, bitmap_size);
    // iterate the LSM vector index (the lock of the vector index is still held)
    for (size_t i = 0; i < DEFAULT_SEGMENT_CAPACITY; i++)
    {
        if (pg_atomic_read_u32(&lsm_index->segments[i].valid))
        {            
            // elog(DEBUG1, "[test_consistency], iterate segment %d", i);
            uint8 *seg_bitmap = (uint8 *) get_pointer_from_cached_segment(lsm_index->segments[i].bitmap, &(lsm_index->segments[i].segment_bitmap_cached_seg));
            int64_t *seg_mapping = (int64_t *) get_pointer_from_cached_segment(lsm_index->segments[i].mapping, &(lsm_index->segments[i].segment_mapping_cached_seg));
            VectorId lo_vid = lsm_index->segments[i].lowestVid;
            VectorId hi_vid = lsm_index->segments[i].highestVid;
            for (VectorId vid = lo_vid; vid <= hi_vid; vid++)
            {
                if (IS_SLOT_SET(seg_bitmap, vid - lo_vid))
                {
                    // check 1
                    if (!IS_SLOT_SET(bitmap_any, vid))
                    {
                        // FIXME: for heap pruning
                        // ItemPointerData tupleId = Int64ToItemPointer  
                        // Buffer buffer = ReadBuffer(heapRel, )
                        elog(DEBUG1, "[test_consistency] inconsistency state: a live vector in a flushed segment is not in the heap table, vid = %ld", vid);
                        return false;
                    }
                    // check 2
                    if (bm_visible_vid != -1 && vid == bm_visible_vid)
                    {
                        bm_visible_vid = get_next_set_slot(bitmap_visible, bm_visible_vid, bitmap_size);
                    }
                }
            }
        }
    }
    elog(DEBUG1, "[test_consistency] iterated all flushed segments");
    
    // iterate the sealed memtable
    for (size_t i = 0; i < DEFAULT_SEALED_MEMTABLE_CAPACITY; i++)
    {
        uint32_t sealed_mt_id = pg_atomic_read_u32(&lsm_index->sealedMemtableIds[i]);
        if (sealed_mt_id >= START_SEGMENT_ID)
        {
            elog(DEBUG1, "[test_consistency] iterate the sealed memtable %d, whose mt_id = %d", i, sealed_mt_id);
            ConcurrentMemTable sealed_mt = (ConcurrentMemTable) get_pointer_from_cached_segment(lsm_index->sealedMemtables[i], &(lsm_index->sealedCachedSegs[i]));
            VectorId lo_vid = sealed_mt->start_vid;
            uint32_t mt_size = pg_atomic_read_u32(&sealed_mt->current_size);
            for (VectorId vid = lo_vid; vid < lo_vid + mt_size; vid++)
            {
                if (IS_SLOT_SET(sealed_mt->bitmap, vid - lo_vid))
                {
                    // check 1
                    if (!IS_SLOT_SET(bitmap_any, vid))
                    {
                        elog(DEBUG1, "[test_consistency] inconsistency state: a live vector in a sealed memtable is not in the heap table");
                        return false;
                    }
                    // check 2
                    if (bm_visible_vid != -1 && vid == bm_visible_vid)
                    {
                        bm_visible_vid = get_next_set_slot(bitmap_visible, bm_visible_vid, bitmap_size);
                    }
                }
            }
        }
    }
    elog(DEBUG1, "[test_consistency] iterated all sealed memtables");

    // iterate the memtable
    if (lsm_index->memtable != 0)
    {
        ConcurrentMemTable mt = (ConcurrentMemTable) get_pointer_from_cached_segment(lsm_index->memtable, &(lsm_index->memtableCachedSeg));
        VectorId lo_vid = mt->start_vid;
        uint32_t mt_size = pg_atomic_read_u32(&mt->current_size);
        for (VectorId vid = lo_vid; vid < lo_vid + mt_size; vid++)
        {
            if (IS_SLOT_SET(mt->bitmap, vid - lo_vid))
            {
                // check 1
                if (!IS_SLOT_SET(bitmap_any, vid))
                {
                    elog(DEBUG1, "[test_consistency] inconsistency state: a live vector in the active memtable is not in the heap table");
                    return false;
                }
                // check 2
                if (bm_visible_vid != -1 && vid == bm_visible_vid)
                {
                    bm_visible_vid = get_next_set_slot(bitmap_visible, bm_visible_vid, bitmap_size);
                }
            }
        }
        elog(DEBUG1, "[test_consistency] iterated the growing memtable");
    }
    else
        elog(DEBUG1, "[test_consistency] no growing memtable need to be iterated");
    
    // check 2
    elog(DEBUG1, "[test_consistency] final bm_visible_vid = %ld", bm_visible_vid);
    if (bm_visible_vid != -1)
    {
        return false;
    }
    
    // pass the visibility check
    elog(DEBUG1, "[test_consistency] passed the visibility check");
    return true;
}