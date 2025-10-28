#include "c.h"
#include "lsmindex.h"
#include "vectorindeximpl.hpp"
#include "storage/fd.h"
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <string.h>
#include "lsm_segment.h"


#ifndef O_DIRECTORY
#define O_DIRECTORY 0
#endif


#define MAX_FILE_SIZE (1L * 1024L * 1024L * 1024L)

/* LSM segment directory path: /.../indexRelId/ */
static void GetLsmDirPath(char *buf, size_t buflen, Oid indexRelId)
{
    snprintf(buf, buflen, VECTOR_STORAGE_BASE_DIR "%u/", indexRelId);
}

static void GetLsmFilePath(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentIdStart, uint32_t segmentIdEnd, const char *type)
{
    snprintf(buf, buflen, VECTOR_STORAGE_BASE_DIR "%u/%s_%u_%u", indexRelId, type, segmentIdStart, segmentIdEnd);
}

static void GetLSMIndexFilePath(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentIdStart, uint32_t segmentIdEnd)
{
    GetLsmFilePath(buf, buflen, indexRelId, segmentIdStart, segmentIdEnd, "index");
}

static void GetLSMBitmapFilePath(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentIdStart, uint32_t segmentIdEnd)
{
    GetLsmFilePath(buf, buflen, indexRelId, segmentIdStart, segmentIdEnd, "bitmap");
}

static void GetLSMMappingFilePath(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentIdStart, uint32_t segmentIdEnd)
{
    GetLsmFilePath(buf, buflen, indexRelId, segmentIdStart, segmentIdEnd, "mapping");
}

static void GetLSMSegmentMetadataPath(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentIdStart, uint32_t segmentIdEnd)
{
    GetLsmFilePath(buf, buflen, indexRelId, segmentIdStart, segmentIdEnd, "metadata");
}

static void GetLSMSegmentMetadataTmpPath(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentIdStart, uint32_t segmentIdEnd)
{
    GetLsmFilePath(buf, buflen, indexRelId, segmentIdStart, segmentIdEnd, "metadata.tmp");
}

static void get_lsm_metadata_path(char *buf, size_t buflen, Oid indexRelId)
{
    snprintf(buf, buflen, VECTOR_STORAGE_BASE_DIR "%u/metadata", indexRelId);
}

static void get_lsm_metadata_tmp_path(char *buf, size_t buflen, Oid indexRelId)
{
    snprintf(buf, buflen, VECTOR_STORAGE_BASE_DIR "%u/metadata.tmp", indexRelId);
}

static void ensure_dir_exists(const char *path)
{
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

// write to disk
static void 
write_segment_file(const char *path, const void *data, Size size)
{
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

// We use alloc instead of palloc here to avoid autovacuum in Postgres (local memory)
// This function can only be called by the vector index worker
static void *
read_segment_file(const char *path)
{
    int fd;
    struct stat st;
    Size size;

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
    void *dest = malloc(total_size);
    if (!dest)
        elog(ERROR, "Failed to allocate memory");

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

    return dest;
}

static void 
write_lsm_segment_metadata(Oid indexRelId, PrepareFlushMeta prep)
{
    char tmp_path[MAXPGPATH];
    char final_path[MAXPGPATH];
    char dir_path[MAXPGPATH];

    GetLsmDirPath(dir_path, sizeof(dir_path), indexRelId);
    GetLSMSegmentMetadataTmpPath(tmp_path, sizeof(tmp_path), indexRelId, prep->start_sid, prep->end_sid);
    GetLSMSegmentMetadataPath(final_path, sizeof(final_path), indexRelId, prep->start_sid, prep->end_sid);

    FILE *fp = fopen(tmp_path, "wb");
    if (!fp)
        elog(ERROR, "[write_lsm_segment_metadata] Cannot open temp metadata file for writing: %s", tmp_path);

    // write segment metadata
    fwrite(&prep->start_sid, sizeof(SegmentId), 1, fp);
    fwrite(&prep->end_sid, sizeof(SegmentId), 1, fp);
    fwrite(&prep->valid_rows, sizeof(uint32_t), 1, fp);
    uint32_t index_type_u32 = (uint32_t) prep->index_type;
    fwrite(&index_type_u32, sizeof(uint32_t), 1, fp);

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

bool
read_lsm_segment_metadata(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, 
					  SegmentId *out_start_sid, SegmentId *out_end_sid, uint32_t *valid_rows, IndexType *index_type)
{
	char metadata_path[MAXPGPATH];
	GetLSMSegmentMetadataPath(metadata_path, sizeof(metadata_path), indexRelId, start_sid, end_sid);
	
	FILE *fp = fopen(metadata_path, "rb");
	if (!fp)
		return false;
	
	if (fread(out_start_sid, sizeof(SegmentId), 1, fp) != 1)
	{
		fclose(fp);
		return false;
	}
	
	if (fread(out_end_sid, sizeof(SegmentId), 1, fp) != 1)
	{
		fclose(fp);
		return false;
	}
	
	if (fread(valid_rows, sizeof(uint32_t), 1, fp) != 1)
	{
		fclose(fp);
		return false;
	}
	
	uint32_t index_type_u32;
	if (fread(&index_type_u32, sizeof(uint32_t), 1, fp) != 1)
	{
		fclose(fp);
		return false;
	}
	*index_type = (IndexType) index_type_u32;
	
	fclose(fp);
	return true;
}

static int
compare_segment_files(const void *a, const void *b)
{
    const SegmentFileInfo *file_a = (const SegmentFileInfo *)a;
    const SegmentFileInfo *file_b = (const SegmentFileInfo *)b;
    
    if (file_a->start_sid < file_b->start_sid)
        return -1;
    else if (file_a->start_sid > file_b->start_sid)
        return 1;
    else
        return 0;
}

int
scan_segment_metadata_files(Oid indexRelId, SegmentFileInfo *files, int max_files)
{
    char dir_path[MAXPGPATH];
    GetLsmDirPath(dir_path, sizeof(dir_path), indexRelId);
    
    DIR *dir = opendir(dir_path);
    if (!dir)
    {
        elog(DEBUG1, "[scan_segment_metadata_files] Cannot open directory: %s", dir_path);
        return 0;
    }
    
    int file_count = 0;
    struct dirent *entry;
    
    while ((entry = readdir(dir)) != NULL && file_count < max_files)
    {
        // Look for files matching pattern: metadata_<start_sid>_<end_sid>
        if (strncmp(entry->d_name, "metadata_", 9) == 0)
        {
            char *filename = entry->d_name + 9; // Skip "metadata_"
            char *underscore = strchr(filename, '_');
            if (underscore)
            {
                *underscore = '\0';
                SegmentId start_sid = (SegmentId)atoi(filename);
                SegmentId end_sid = (SegmentId)atoi(underscore + 1);
                
                files[file_count].start_sid = start_sid;
                files[file_count].end_sid = end_sid;
                snprintf(files[file_count].filename, sizeof(files[file_count].filename), 
                        "%s/%s", dir_path, entry->d_name);
                file_count++;
            }
        }
    }
    
    closedir(dir);
    
    // Sort files by start_sid
    qsort(files, file_count, sizeof(SegmentFileInfo), compare_segment_files);
    
    return file_count;
}

void
write_lsm_index_metadata(LSMIndex lsm)
{
	char dir_path[MAXPGPATH];
	char tmp_path[MAXPGPATH];
	char final_path[MAXPGPATH];

	GetLsmDirPath(dir_path, sizeof(dir_path), lsm->indexRelId);
	ensure_dir_exists(dir_path);
	get_lsm_metadata_tmp_path(tmp_path, sizeof(tmp_path), lsm->indexRelId);
	get_lsm_metadata_path(final_path, sizeof(final_path), lsm->indexRelId);

	FILE *fp = fopen(tmp_path, "wb");
	if (!fp)
		elog(ERROR, "[flush_lsm_index_metadata] Cannot open temp metadata file for writing: %s", tmp_path);

	/* Persist basic LSM index metadata */
	uint32_t index_type_u32 = (uint32_t) lsm->index_type;
	if (fwrite(&index_type_u32, sizeof(uint32_t), 1, fp) != 1)
		elog(ERROR, "[flush_lsm_index_metadata] Failed to write index_type to %s", tmp_path);
	if (fwrite(&lsm->dim, sizeof(uint32_t), 1, fp) != 1)
		elog(ERROR, "[flush_lsm_index_metadata] Failed to write dim to %s", tmp_path);
	if (fwrite(&lsm->elem_size, sizeof(uint32_t), 1, fp) != 1)
		elog(ERROR, "[flush_lsm_index_metadata] Failed to write elem_size to %s", tmp_path);

	/* Flush file buffers to disk */
	fflush(fp);
	int fd = fileno(fp);
	if (fd != -1)
		fsync(fd);
	fclose(fp);

	/* Atomic rename to final path */
	if (rename(tmp_path, final_path) != 0)
		elog(ERROR, "Failed to rename temp metadata file to final location: %s → %s", tmp_path, final_path);

	/* Fsync the directory for durability */
	int dirfd = open(dir_path, O_DIRECTORY | O_RDONLY);
	if (dirfd != -1)
	{
		fsync(dirfd);
		close(dirfd);
	}

	elog(LOG, "Crash-safe LSM metadata written to: %s", final_path);
}

bool
read_lsm_index_metadata(Oid indexRelId, IndexType *index_type, uint32_t *dim, uint32_t *elem_size)
{
	char metadata_path[MAXPGPATH];
	get_lsm_metadata_path(metadata_path, sizeof(metadata_path), indexRelId);
	
	FILE *fp = fopen(metadata_path, "rb");
	if (!fp)
		return false;
	
	uint32_t index_type_u32;
	if (fread(&index_type_u32, sizeof(uint32_t), 1, fp) != 1)
	{
		fclose(fp);
		return false;
	}
	*index_type = (IndexType) index_type_u32;
	
	if (fread(dim, sizeof(uint32_t), 1, fp) != 1)
	{
		fclose(fp);
		return false;
	}
	
	if (fread(elem_size, sizeof(uint32_t), 1, fp) != 1)
	{
		fclose(fp);
		return false;
	}
	
	fclose(fp);
	return true;
}

void 
flush_segment_to_disk(Oid indexRelId, PrepareFlushMeta prep)
{
    elog(DEBUG1, "[flush_segment_to_disk] segment whose start id = %u, end_id = %u", prep->start_sid, prep->end_sid);

    char file_path[MAXPGPATH];
    GetLsmDirPath(file_path, MAXPGPATH, indexRelId);
    ensure_dir_exists(file_path);

    // index file
    GetLSMIndexFilePath(file_path, sizeof(file_path), indexRelId, prep->start_sid, prep->end_sid);
    IndexBinarySetFlush(file_path, prep->index_bin);

    // bitmap file
    GetLSMBitmapFilePath(file_path, sizeof(file_path), indexRelId, prep->start_sid, prep->end_sid);
    write_segment_file(file_path, prep->bitmap_ptr, prep->bitmap_size);

    // mapping file
    GetLSMMappingFilePath(file_path, sizeof(file_path), indexRelId, prep->start_sid, prep->end_sid);
    write_segment_file(file_path, prep->map_ptr, prep->map_size);

    // metadata file
    write_lsm_segment_metadata(indexRelId, prep);
}

void 
load_bitmap_file(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint8_t **bitmap)
{
    char path[MAXPGPATH];
    GetLSMBitmapFilePath(path, sizeof(path), indexRelId, start_sid, end_sid);
    *bitmap = (uint8_t *) read_segment_file(path);
}

void 
load_mapping_file(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, int64_t **mapping)
{
    char path[MAXPGPATH];
    GetLSMMappingFilePath(path, sizeof(path), indexRelId, start_sid, end_sid);
    *mapping = (int64_t *) read_segment_file(path);
}

void 
load_index_file(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, IndexType index_type, void **index)
{
    char path[MAXPGPATH];
    GetLSMIndexFilePath(path, sizeof(path), indexRelId, start_sid, end_sid);
    IndexLoadAndSave(path, index_type, index);
}
