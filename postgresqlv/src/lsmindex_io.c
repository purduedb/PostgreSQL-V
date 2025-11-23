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

static void GetLsmFilePathWithVersion(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentIdStart, uint32_t segmentIdEnd, uint32_t version, const char *type)
{
    snprintf(buf, buflen, VECTOR_STORAGE_BASE_DIR "%u/%s_%u_%u_v%u", indexRelId, type, segmentIdStart, segmentIdEnd, version);
}

static void GetLSMIndexFilePathWithVersion(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentIdStart, uint32_t segmentIdEnd, uint32_t version)
{
    GetLsmFilePathWithVersion(buf, buflen, indexRelId, segmentIdStart, segmentIdEnd, version, "index");
}

static void GetLSMBitmapFilePathWithVersion(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentIdStart, uint32_t segmentIdEnd, uint32_t version)
{
    GetLsmFilePathWithVersion(buf, buflen, indexRelId, segmentIdStart, segmentIdEnd, version, "bitmap");
}

// Get bitmap file path with version and optional subversion
// If subversion is UINT32_MAX, no subversion suffix is added
static void GetLSMBitmapFilePathWithSubversion(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentIdStart, uint32_t segmentIdEnd, uint32_t version, uint32_t subversion)
{
    if (subversion == UINT32_MAX)
    {
        // No subversion - use standard format
        GetLSMBitmapFilePathWithVersion(buf, buflen, indexRelId, segmentIdStart, segmentIdEnd, version);
    }
    else
    {
        // Include subversion in filename: bitmap_<start>_<end>_v<version>_s<subversion>
        snprintf(buf, buflen, VECTOR_STORAGE_BASE_DIR "%u/bitmap_%u_%u_v%u_s%u", 
                 indexRelId, segmentIdStart, segmentIdEnd, version, subversion);
    }
}

static void GetLSMBitmapFilePathForMemtable(char *buf, size_t buflen, Oid indexRelId, SegmentId memtable_id)
{
    snprintf(buf, buflen, VECTOR_STORAGE_BASE_DIR "%u/bitmap_memtable_%u", indexRelId, memtable_id);
}

static void GetLSMMappingFilePathWithVersion(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentIdStart, uint32_t segmentIdEnd, uint32_t version)
{
    GetLsmFilePathWithVersion(buf, buflen, indexRelId, segmentIdStart, segmentIdEnd, version, "mapping");
}

static void GetLSMSegmentMetadataPathWithVersion(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentIdStart, uint32_t segmentIdEnd, uint32_t version)
{
    GetLsmFilePathWithVersion(buf, buflen, indexRelId, segmentIdStart, segmentIdEnd, version, "metadata");
}

static void GetLSMSegmentMetadataTmpPathWithVersion(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentIdStart, uint32_t segmentIdEnd, uint32_t version)
{
    GetLsmFilePathWithVersion(buf, buflen, indexRelId, segmentIdStart, segmentIdEnd, version, "metadata.tmp");
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
// If delete_count is not NULL, it will be written first (4 bytes) before the data
static void 
write_segment_file(const char *path, const void *data, Size size, const uint32_t *delete_count)
{
    Size written_size = 0;
    int chunk_index = 0;
    bool delete_count_written = false;
    
    while (written_size < size || (delete_count != NULL && !delete_count_written)) {
        char tmp_path[MAXPGPATH];
        snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d", path, chunk_index);

        int fd = OpenTransientFile(tmp_path, O_CREAT | O_WRONLY | O_TRUNC);
        if (fd < 0)
            elog(ERROR, "Failed to create temporary file: %s", tmp_path);

        Size total_to_write = 0;
        
        // Write delete_count first if this is the first chunk and delete_count is provided
        if (delete_count != NULL && !delete_count_written && chunk_index == 0)
        {
            Size write_size;
            if ((write_size = write(fd, delete_count, sizeof(uint32_t))) != sizeof(uint32_t))
                elog(ERROR, "Failed to write delete_count to temporary file: %s", tmp_path);
            total_to_write += sizeof(uint32_t);
            delete_count_written = true;
        }
        
        // Write data chunk
        if (written_size < size)
        {
            Size chunk_size = (size - written_size > MAX_FILE_SIZE - total_to_write) ? 
                              (MAX_FILE_SIZE - total_to_write) : size - written_size;

            Size write_size;
            if ((write_size = write(fd, (char *)data + written_size, chunk_size)) != chunk_size)
                elog(ERROR, "Failed to write full content to temporary file: %s, should write %ld but wrote %ld instead", tmp_path, chunk_size, write_size);

            written_size += chunk_size;
        }

        if (fsync(fd) != 0)
            elog(ERROR, "fsync failed on temporary file: %s", tmp_path);

        CloseTransientFile(fd);
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
// note that we use malloc instead of palloc here as this function will be called by the vector index worker
// If delete_count_out is not NULL, it will read delete_count first (4 bytes) from the file
static void *
read_segment_file(const char *path, bool pg_alloc, uint32_t *delete_count_out)
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
                fprintf(stderr, "[read_segment_file] ERROR: Could not open FAISS segment file: %s\n", path);
            break; // No more chunks to read
        }

        /* Stat file */
        if (fstat(fd, &st) < 0)
            fprintf(stderr, "[read_segment_file] ERROR: Could not stat FAISS segment file: %s\n", chunk_path);

        size = st.st_size;
        total_size += size;

        CloseTransientFile(fd);
        chunk_index++;
    }

    /* Adjust total_size if delete_count is present */
    Size data_size = total_size;
    if (delete_count_out != NULL && total_size >= sizeof(uint32_t))
    {
        data_size = total_size - sizeof(uint32_t);
    }

    /* Now, allocate a large memory block to hold all the chunks */
    void *dest = pg_alloc ? palloc(data_size) : malloc(data_size);
    if (!dest)
        fprintf(stderr, "[read_segment_file] ERROR: Failed to allocate memory\n");

    int chunk_num = chunk_index;
    Size offset = 0;
    bool delete_count_read = false;
    
    for (int chunk_index = 0; chunk_index < chunk_num; chunk_index++)
    {
        char chunk_path[MAXPGPATH];
        snprintf(chunk_path, sizeof(chunk_path), "%s.%d", path, chunk_index);

        fd = OpenTransientFile(chunk_path, O_RDONLY | PG_BINARY);
        if (fd < 0)
        {
            fprintf(stderr, "[read_segment_file] ERROR: Could not open FAISS segment file: %s\n", chunk_path);
            Assert(0);
        }

        /* Stat file */
        if (fstat(fd, &st) < 0)
        {
            fprintf(stderr, "[read_segment_file] ERROR: Could not stat FAISS segment file: %s\n", chunk_path);
            CloseTransientFile(fd);
            Assert(0);
        }

        size = st.st_size;
        
        // Read delete_count from first chunk if needed
        if (delete_count_out != NULL && !delete_count_read && chunk_index == 0 && size >= sizeof(uint32_t))
        {
            if (read(fd, delete_count_out, sizeof(uint32_t)) != sizeof(uint32_t))
            {
                fprintf(stderr, "[read_segment_file] ERROR: Failed to read delete_count from file: %s\n", chunk_path);
                CloseTransientFile(fd);
                continue;
            }
            size -= sizeof(uint32_t);  // Adjust size to skip delete_count
            delete_count_read = true;
        }
        
        // Read data chunk
        if (size > 0 && offset < data_size)
        {
            Size read_size = (offset + size > data_size) ? (data_size - offset) : size;
            if (read(fd, (char *)dest + offset, read_size) != read_size)
            {
                fprintf(stderr, "[read_segment_file] ERROR: Failed to read complete segment file into memory: %s\n", chunk_path);
                CloseTransientFile(fd);
                continue;
            }
            offset += read_size;
        }

        CloseTransientFile(fd);
    }

    return dest;
}

/* Find the latest version number for a segment by scanning directory */
uint32_t
find_latest_segment_version(Oid indexRelId, SegmentId start_sid, SegmentId end_sid)
{
    char dir_path[MAXPGPATH];
    GetLsmDirPath(dir_path, sizeof(dir_path), indexRelId);
    
    DIR *dir = opendir(dir_path);
    if (!dir)
    {
        // No directory or no existing files, start with version 0
        return 0;
    }
    
    uint32_t max_version = 0;
    char pattern_prefix[256];
    snprintf(pattern_prefix, sizeof(pattern_prefix), "metadata_%u_%u_v", start_sid, end_sid);
    size_t prefix_len = strlen(pattern_prefix);
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        // Look for files matching pattern: metadata_<start_sid>_<end_sid>_v<version>
        if (strncmp(entry->d_name, pattern_prefix, prefix_len) == 0)
        {
            char *version_str = entry->d_name + prefix_len;
            uint32_t version = (uint32_t)atoi(version_str);
            if (version > max_version)
            {
                max_version = version;
            }
        }
    }
    
    closedir(dir);
    return max_version;
}

/* Find the latest subversion for a bitmap file given a version */
uint32_t
find_latest_bitmap_subversion(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version)
{
    char dir_path[MAXPGPATH];
    GetLsmDirPath(dir_path, sizeof(dir_path), indexRelId);
    
    DIR *dir = opendir(dir_path);
    if (!dir)
    {
        // No directory - no subversion exists
        return UINT32_MAX;
    }
    
    // Pattern: bitmap_<start_sid>_<end_sid>_v<version>_s<subversion>
    char pattern_prefix[256];
    snprintf(pattern_prefix, sizeof(pattern_prefix), "bitmap_%u_%u_v%u_s", start_sid, end_sid, version);
    size_t prefix_len = strlen(pattern_prefix);
    
    uint32_t max_subversion = UINT32_MAX;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        // Look for files matching pattern with subversion
        if (strncmp(entry->d_name, pattern_prefix, prefix_len) == 0)
        {
            char *subversion_str = entry->d_name + prefix_len;
            uint32_t subversion = (uint32_t)atoi(subversion_str);
            if (max_subversion == UINT32_MAX || subversion > max_subversion)
            {
                max_subversion = subversion;
            }
        }
    }
    
    closedir(dir);
    return max_subversion;
}

static void 
write_lsm_segment_metadata(Oid indexRelId, PrepareFlushMeta prep, uint32_t version)
{
    char tmp_path[MAXPGPATH];
    char final_path[MAXPGPATH];
    char dir_path[MAXPGPATH];

    GetLsmDirPath(dir_path, sizeof(dir_path), indexRelId);
    GetLSMSegmentMetadataTmpPathWithVersion(tmp_path, sizeof(tmp_path), indexRelId, prep->start_sid, prep->end_sid, version);
    GetLSMSegmentMetadataPathWithVersion(final_path, sizeof(final_path), indexRelId, prep->start_sid, prep->end_sid, version);

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
read_lsm_segment_metadata(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version,
					  SegmentId *out_start_sid, SegmentId *out_end_sid, uint32_t *valid_rows, IndexType *index_type)
{
	char metadata_path[MAXPGPATH];
	
	// If version is UINT32_MAX, find latest version
	if (version == UINT32_MAX)
	{
		version = find_latest_segment_version(indexRelId, start_sid, end_sid);
	}

	GetLSMSegmentMetadataPathWithVersion(metadata_path, sizeof(metadata_path), indexRelId, start_sid, end_sid, version);
	
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

static int
compare_segment_versions(const void *a, const void *b)
{
    typedef struct {
        SegmentId start_sid;
        SegmentId end_sid;
        uint32_t max_version;
        bool found;
    } SegmentVersion;
    
    const SegmentVersion *seg_a = (const SegmentVersion *)a;
    const SegmentVersion *seg_b = (const SegmentVersion *)b;
    
    // First sort by start_sid
    if (seg_a->start_sid < seg_b->start_sid)
        return -1;
    else if (seg_a->start_sid > seg_b->start_sid)
        return 1;
    else
    {
        // For same start_sid, prefer larger scope (end_sid - start_sid)
        // This ensures merged segments (larger scope) are processed first
        if (seg_a->end_sid > seg_b->end_sid)
            return -1;
        else if (seg_a->end_sid < seg_b->end_sid)
            return 1;
        else
            return 0;
    }
}

int
scan_segment_metadata_files(Oid indexRelId, SegmentFileInfo *files, int max_files)
{
    char dir_path[MAXPGPATH];
    GetLsmDirPath(dir_path, sizeof(dir_path), indexRelId);
    
    DIR *dir = opendir(dir_path);
    if (!dir)
    {
        fprintf(stderr, "[scan_segment_metadata_files] Cannot open directory: %s\n", dir_path);
        return 0;
    }
    
    // Temporary storage to track latest version for each segment
    typedef struct {
        SegmentId start_sid;
        SegmentId end_sid;
        uint32_t max_version;
        bool found;
    } SegmentVersion;
    
    SegmentVersion segments[MAX_SEGMENTS_COUNT];
    int segment_count = 0;
    struct dirent *entry;
    
    while ((entry = readdir(dir)) != NULL)
    {
        // Look for files matching pattern: metadata_<start_sid>_<end_sid>_v<version>
        if (strncmp(entry->d_name, "metadata_", 9) == 0)
        {
            char *filename = entry->d_name + 9; // Skip "metadata_"
            char *underscore1 = strchr(filename, '_');
            if (underscore1)
            {
                *underscore1 = '\0';
                SegmentId start_sid = (SegmentId)atoi(filename);
                
                char *rest = underscore1 + 1;
                char *version_start = strstr(rest, "_v");
                if (!version_start)
                    continue; // Skip files without version number
                
                // Format: metadata_<start_sid>_<end_sid>_v<version>
                *version_start = '\0';
                SegmentId end_sid = (SegmentId)atoi(rest);
                uint32_t version = (uint32_t)atoi(version_start + 2);
                
                // Find or create entry for this segment
                int seg_idx = -1;
                for (int i = 0; i < segment_count; i++)
                {
                    if (segments[i].start_sid == start_sid && segments[i].end_sid == end_sid)
                    {
                        seg_idx = i;
                        break;
                    }
                }
                
                if (seg_idx == -1)
                {
                    // New segment
                    if (segment_count >= MAX_SEGMENTS_COUNT)
                        continue;
                    seg_idx = segment_count++;
                    segments[seg_idx].start_sid = start_sid;
                    segments[seg_idx].end_sid = end_sid;
                    segments[seg_idx].max_version = version;
                    segments[seg_idx].found = true;
                }
                else
                {
                    // Update max version
                    if (version > segments[seg_idx].max_version)
                    {
                        segments[seg_idx].max_version = version;
                    }
                }
            }
        }
    }
    
    closedir(dir);
    
    // Sort segments by start_sid, and for same start_sid, prefer larger scope
    qsort(segments, segment_count, sizeof(SegmentVersion), compare_segment_versions);
    
    // Sequentially scan segments, ensuring no gaps or overlaps
    // Track the current largest end_sid and only accept contiguous segments
    SegmentId largest_end_sid = (SegmentId)-1; // Initialize to -1 (no segments processed yet)
    int file_count = 0;
    
    for (int i = 0; i < segment_count && file_count < max_files; i++)
    {
        SegmentId start_sid = segments[i].start_sid;
        SegmentId end_sid = segments[i].end_sid;
        
        if (largest_end_sid == (SegmentId)-1)
        {
            // First segment - accept it and initialize largest_end_sid
            files[file_count].start_sid = start_sid;
            files[file_count].end_sid = end_sid;
            files[file_count].version = segments[i].max_version;
            largest_end_sid = end_sid;
            file_count++;
            // elog(DEBUG1, "[scan_segment_metadata_files] accepted first segment: %u-%u", start_sid, end_sid);
        }
        else if (start_sid <= largest_end_sid)
        {
            // Overlap detected - skip this segment
            // (we prefer the segment with larger scope, which should have been processed first after sorting)
            // elog(DEBUG1, "[scan_segment_metadata_files] skipped overlapping segment: %u-%u", start_sid, end_sid);
            continue;
        }
        else if (start_sid == largest_end_sid + 1)
        {
            // Contiguous segment - accept it
            files[file_count].start_sid = start_sid;
            files[file_count].end_sid = end_sid;
            files[file_count].version = segments[i].max_version;
            largest_end_sid = end_sid;
            file_count++;
            // elog(DEBUG1, "[scan_segment_metadata_files] accepted contiguous segment: %u-%u", start_sid, end_sid);
        }
        else // start_sid > largest_end_sid + 1
        {
            // Gap detected - trigger error
            elog(ERROR, "Gap detected in segment files: expected start_sid %u but found %u (largest_end_sid: %u)",
                 (unsigned int)(largest_end_sid + 1), (unsigned int)start_sid, (unsigned int)largest_end_sid);
        }
    }
    
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

// Note: the index binary set will be freed in this function
void 
flush_segment_to_disk(Oid indexRelId, PrepareFlushMeta prep)
{
    elog(DEBUG1, "[flush_segment_to_disk] segment whose start id = %u, end_id = %u", prep->start_sid, prep->end_sid);

    char file_path[MAXPGPATH];
    GetLsmDirPath(file_path, MAXPGPATH, indexRelId);
    ensure_dir_exists(file_path);

    // Find latest version and increment it for atomic writing
    uint32_t latest_version = find_latest_segment_version(indexRelId, prep->start_sid, prep->end_sid);
    uint32_t new_version = latest_version + 1;
    
    elog(DEBUG1, "[flush_segment_to_disk] Writing segment with version %u (previous: %u)", new_version, latest_version);

    // index file - write with new version
    GetLSMIndexFilePathWithVersion(file_path, sizeof(file_path), indexRelId, prep->start_sid, prep->end_sid, new_version);
    IndexBinarySetFlush(file_path, prep->index_bin);

    // bitmap file - write with new version (include delete_count)
    GetLSMBitmapFilePathWithVersion(file_path, sizeof(file_path), indexRelId, prep->start_sid, prep->end_sid, new_version);
    // TODO: for debugging
    elog(DEBUG1, "[flush_segment_to_disk] Writing bitmap file for segment %u-%u version %u, delete_count %u", 
         prep->start_sid, prep->end_sid, new_version, prep->delete_count);
    write_segment_file(file_path, prep->bitmap_ptr, prep->bitmap_size, &prep->delete_count);

    // mapping file - write with new version (no delete_count)
    GetLSMMappingFilePathWithVersion(file_path, sizeof(file_path), indexRelId, prep->start_sid, prep->end_sid, new_version);
    write_segment_file(file_path, prep->map_ptr, prep->map_size, NULL);

    // metadata file - write with new version (this is the last step, making it atomic)
    write_lsm_segment_metadata(indexRelId, prep, new_version);
    
    elog(DEBUG1, "[flush_segment_to_disk] Successfully wrote segment %u-%u with version %u", prep->start_sid, prep->end_sid, new_version);
}

// Read only the delete_count from bitmap file (more efficient than loading entire bitmap)
bool
read_bitmap_delete_count(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, uint32_t *delete_count_out)
{
    // If version is UINT32_MAX, find latest version
    if (version == UINT32_MAX)
    {
        version = find_latest_segment_version(indexRelId, start_sid, end_sid);
    }
    
    // Find the latest subversion for this version (if any)
    uint32_t subversion = find_latest_bitmap_subversion(indexRelId, start_sid, end_sid, version);
    
    char path[MAXPGPATH];
    GetLSMBitmapFilePathWithSubversion(path, sizeof(path), indexRelId, start_sid, end_sid, version, subversion);
    
    // Open the first chunk file (delete_count is always in the first chunk)
    char chunk_path[MAXPGPATH];
    snprintf(chunk_path, sizeof(chunk_path), "%s.0", path);
    
    int fd = OpenTransientFile(chunk_path, O_RDONLY | PG_BINARY);
    if (fd < 0)
    {
        return false;
    }
    
    // Read only the delete_count (first 4 bytes)
    uint32_t delete_count;
    if (read(fd, &delete_count, sizeof(uint32_t)) != sizeof(uint32_t))
    {
        CloseTransientFile(fd);
        return false;
    }
    
    CloseTransientFile(fd);
    *delete_count_out = delete_count;
    return true;
}

void 
load_bitmap_file(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, uint8_t **bitmap, bool pg_alloc, uint32_t *delete_count_out)
{
    // If version is UINT32_MAX, find latest version
    if (version == UINT32_MAX)
    {
        version = find_latest_segment_version(indexRelId, start_sid, end_sid);
    }
    
    // Find the latest subversion for this version (if any)
    uint32_t subversion = find_latest_bitmap_subversion(indexRelId, start_sid, end_sid, version);
    
    char path[MAXPGPATH];
    GetLSMBitmapFilePathWithSubversion(path, sizeof(path), indexRelId, start_sid, end_sid, version, subversion);
    *bitmap = (uint8_t *) read_segment_file(path, pg_alloc, delete_count_out);

    // TODO: for debugging
    elog(DEBUG1, "[load_bitmap_file] Loaded bitmap file for segment %u-%u version %u subversion %u, delete_count %u", 
         start_sid, end_sid, version, subversion, *delete_count_out);
}

// Write bitmap file with subversion (for vacuum operations)
// If subversion is UINT32_MAX, writes without subversion suffix
// delete_count will be written to the file
void 
write_bitmap_file_with_subversion(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, uint32_t subversion, const uint8_t *bitmap, Size bitmap_size, uint32_t delete_count)
{
    // elog(DEBUG1, "[write_bitmap_file_with_subversion] Writing bitmap file for segment %u-%u version %u subversion %u, delete_count %u", 
    //      start_sid, end_sid, version, subversion, delete_count);

    char file_path[MAXPGPATH];
    GetLsmDirPath(file_path, MAXPGPATH, indexRelId);
    ensure_dir_exists(file_path);
    
    GetLSMBitmapFilePathWithSubversion(file_path, sizeof(file_path), indexRelId, start_sid, end_sid, version, subversion);
    write_segment_file(file_path, bitmap, bitmap_size, &delete_count);
    
    if (subversion == UINT32_MAX)
    {
        elog(DEBUG1, "[write_bitmap_file_with_subversion] Successfully wrote bitmap file for segment %u-%u version %u", 
             start_sid, end_sid, version);
    }
    else
    {
        elog(DEBUG1, "[write_bitmap_file_with_subversion] Successfully wrote bitmap file for segment %u-%u version %u subversion %u", 
             start_sid, end_sid, version, subversion);
    }
}

void
write_bitmap_for_memtable(Oid indexRelId, SegmentId memtable_id, uint8_t *bitmap, Size bitmap_size, uint32_t delete_count)
{
    char file_path[MAXPGPATH];
    GetLsmDirPath(file_path, MAXPGPATH, indexRelId);
    ensure_dir_exists(file_path);

    GetLSMBitmapFilePathForMemtable(file_path, sizeof(file_path), indexRelId, memtable_id);
    write_segment_file(file_path, bitmap, bitmap_size, &delete_count);
    
    elog(DEBUG1, "[write_bitmap_for_memtable] Successfully wrote bitmap file for memtable %u", memtable_id);
}

void 
load_mapping_file(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, int64_t **mapping, bool pg_alloc)
{
    // If version is UINT32_MAX, find latest version
    if (version == UINT32_MAX)
    {
        version = find_latest_segment_version(indexRelId, start_sid, end_sid);
    }
    
    char path[MAXPGPATH];
    GetLSMMappingFilePathWithVersion(path, sizeof(path), indexRelId, start_sid, end_sid, version);
    *mapping = (int64_t *) read_segment_file(path, pg_alloc, NULL);
}

void 
load_index_file(Oid indexRelId, SegmentId start_sid, SegmentId end_sid, uint32_t version, IndexType index_type, void **index)
{
    // If version is UINT32_MAX, find latest version
    if (version == UINT32_MAX)
    {
        version = find_latest_segment_version(indexRelId, start_sid, end_sid);
    }

    char path[MAXPGPATH];
    GetLSMIndexFilePathWithVersion(path, sizeof(path), indexRelId, start_sid, end_sid, version);
    IndexLoadAndSave(path, index_type, index);
}
