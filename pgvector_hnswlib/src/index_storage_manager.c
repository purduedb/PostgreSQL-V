#include "postgres.h"
#include "index_storage_manager.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"
#include "storage/lwlock.h"

VectorStorageMetadata *vector_storage_metadata = NULL;

static char *get_metadata_file_path(void)
{
    static char path[MAXPGPATH];

    // TODO: check if the file is stored in json?
    snprintf(path, sizeof(path), "%s%s", VECTOR_STORAGE_BASE_DIR, VECTOR_METADATA_FILENAME);
    return path;
}

// Initialize the hash table if not already done and also load the file to the metadata cache
static void initialize_metadata_cache(void)
{
    static LWLock *metadata_lock = NULL;

    bool found;
    
    if (metadata_lock == NULL)
    {
        // Acquire the lock to prevent race conditions during init
        metadata_lock = &(GetNamedLWLockTranche("vector_storage_metadata")->lock);
    }

    LWLockAcquire(metadata_lock, LW_EXCLUSIVE);

    // only the first backend will initialize the structure
    vector_storage_metadata = ShmemInitStruct("Vector Storage Metadata",
                                              sizeof(VectorStorageMetadata),
                                              &found);

    if (!found)
    {
        HASHCTL ctl;
        memset(&ctl, 0, sizeof(ctl));
        ctl.keysize = sizeof(Oid);
        ctl.entrysize = sizeof(VectorIndexMetadataEntry);
        // FIXME: currently, we only support 256 indexes at maximum
        vector_storage_metadata->hashTable = ShmemInitHash("Vector Storage Hash Table", 128, 256, &ctl, HASH_ELEM | HASH_SHARED_MEM | HASH_ALLOC);

        // load data
        FILE *fp = fopen(get_metadata_file_path(), "r");
        if (fp == NULL)
        {
            // No metadata file yet; not an error
            elog(NOTICE, "No vector storage metdata file yet");
            LWLockRelease(metadata_lock);
            return;
        }

        char line[256];
        while (fgets(line, sizeof(line), fp))
        {
            Oid relid;
            char path[128];

            if (sscanf(line, "%u %127s", &relid, path) == 2)
            {
                bool found;
                VectorIndexMetadataEntry *entry = (VectorIndexMetadataEntry *) hash_search(vector_storage_metadata->hashTable,
                                                                                            &relid, HASH_ENTER, &found);
                strncpy(entry->directoryPath, path, sizeof(entry->directoryPath) - 1);
                entry->directoryPath[sizeof(entry->directoryPath) - 1] = '\0';
            }
        }

        fclose(fp);
    }
    LWLockRelease(metadata_lock);
}


// Write from memory to metadata file
void write_vector_metadata(void)
{
    static LWLock *metadata_lock = NULL;
    if (metadata_lock == NULL)
        metadata_lock = &(GetNamedLWLockTranche("vector_storage_metadata")->lock);

    // Ensure metadata is initialized
    if (vector_storage_metadata == NULL)
        initialize_metadata_cache();

    LWLockAcquire(metadata_lock, LW_SHARED);  // allow concurrent reads

    FILE *fp = fopen(get_metadata_file_path(), "w");
    if (fp == NULL)
    {
        LWLockRelease(metadata_lock);
        elog(ERROR, "Failed to open metadata file for writing");
        return;
    }

    HASH_SEQ_STATUS status;
    hash_seq_init(&status, vector_storage_metadata->hashTable);

    VectorIndexMetadataEntry *entry;
    while ((entry = (VectorIndexMetadataEntry *) hash_seq_search(&status)) != NULL)
    {
        fprintf(fp, "%u %s\n", entry->indexRelId, entry->directoryPath);
    }

    fclose(fp);
    LWLockRelease(metadata_lock);
}

const char *get_vector_index_directory(Oid indexRelId)
{
    static LWLock *metadata_lock = NULL;
    if (metadata_lock == NULL)
        metadata_lock = &(GetNamedLWLockTranche("vector_storage_metadata")->lock);

    if (vector_storage_metadata == NULL)
        initialize_metadata_cache();

    LWLockAcquire(metadata_lock, LW_SHARED);

    VectorIndexMetadataEntry *entry;
    entry = (VectorIndexMetadataEntry *) hash_search(vector_storage_metadata->hashTable,
                                                     &indexRelId,
                                                     HASH_FIND,
                                                     NULL);

    if (entry == NULL)
    {
        LWLockRelease(metadata_lock);
        elog(ERROR, "Vector index metadata not found for relation ID %u", indexRelId);
        return NULL;
    }

    static char fullPath[MAXPGPATH];
    snprintf(fullPath, sizeof(fullPath), "%s%s", VECTOR_STORAGE_BASE_DIR, entry->directoryPath);

    LWLockRelease(metadata_lock);
    return fullPath;
}

void register_vector_index_metadata(Oid indexRelId)
{
    static LWLock *metadata_lock = NULL;
    if (metadata_lock == NULL)
        metadata_lock = &(GetNamedLWLockTranche("vector_storage_metadata")->lock);

    if (vector_storage_metadata == NULL)
        initialize_metadata_cache();

    LWLockAcquire(metadata_lock, LW_EXCLUSIVE);

    bool found;
    VectorIndexMetadataEntry *entry = (VectorIndexMetadataEntry *) hash_search(
        vector_storage_metadata->hashTable,
        &indexRelId,
        HASH_ENTER,
        &found);

    // Generate relative directory path: e.g., "12345/"
    snprintf(entry->directoryPath, sizeof(entry->directoryPath), "%u/", indexRelId);

    // Persist the updated hash table to file
    write_vector_metadata();

    LWLockRelease(metadata_lock);
}

