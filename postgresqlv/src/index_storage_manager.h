#ifndef INDEX_STORAGE_MANAGER_H
#define INDEX_STORAGE_MANAGER_H

#include "postgres.h"
#include "utils/hsearch.h"
#include "storage/ipc.h"

// Metadata entry mapping a relation ID to its external storage path
typedef struct VectorIndexMetadataEntry
{
    Oid indexRelId;        // Index relation OID (unique identifier)
    char directoryPath[128];   // Full directory path where vector files are stored
} VectorIndexMetadataEntry;


// Hash table used to cache metadata mapping in memory
typedef struct VectorStorageMetadata
{
    HTAB *hashTable;       // Postgres hash table: Oid -> VectorIndexMetadataEntry
} VectorStorageMetadata;

extern VectorStorageMetadata *vector_storage_metadata;

void 
write_vector_metadata(void);

const char 
*get_vector_index_directory(Oid indexRelId);

void 
register_vector_index_metadata(Oid indexRelId);


// For chaining with other extensions
static shmem_startup_hook_type prev_shmem_startup_hook = NULL;



#define VECTOR_STORAGE_BASE_DIR "/ssd_root/liu4127/pg_vector_extension_indexes/"
#define VECTOR_METADATA_FILENAME "storage_meta.json"

/* LSM segment directory path: /.../indexRelId/ */
void get_lsm_dir_path(char *buf, size_t buflen, Oid indexRelId)
{
    snprintf(buf, buflen, VECTOR_STORAGE_BASE_DIR "%u/", indexRelId);
}

/* Generic LSM segment file path: /.../indexRelId/{type}_segmentId */
void get_lsm_segment_path(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentId, const char *type)
{
    snprintf(buf, buflen, VECTOR_STORAGE_BASE_DIR "%u/%s_%u", indexRelId, type, segmentId);
}

/* Specific helpers */
void get_lsm_segment_index_path(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentId)
{
    get_lsm_segment_path(buf, buflen, indexRelId, segmentId, "index");
}

void get_lsm_segment_bitmap_path(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentId)
{
    get_lsm_segment_path(buf, buflen, indexRelId, segmentId, "bitmap");
}

void get_lsm_segment_mapping_path(char *buf, size_t buflen, Oid indexRelId, uint32_t segmentId)
{
    get_lsm_segment_path(buf, buflen, indexRelId, segmentId, "mapping");
}

/* Metadata files: /.../indexRelId/metadata and metadata.tmp */
void get_lsm_metadata_path(char *buf, size_t buflen, Oid indexRelId)
{
    snprintf(buf, buflen, VECTOR_STORAGE_BASE_DIR "%u/metadata", indexRelId);
}

void get_lsm_metadata_tmp_path(char *buf, size_t buflen, Oid indexRelId)
{
    snprintf(buf, buflen, VECTOR_STORAGE_BASE_DIR "%u/metadata.tmp", indexRelId);
}



#endif  // INDEX_STORAGE_MANAGER_H