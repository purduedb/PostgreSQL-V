#ifndef VECTORINDEXIMPL_HPP
#define VECTORINDEXIMPL_HPP
#ifdef __cplusplus
extern "C" {
#endif

#include "postgres.h"
#include "lsmindex.h"
#include "utils.h"


// hnsw
int HnswIndexInit(int dimension, int M, int efConstruction, void** hnswIndexPtr);
int HnswIndexcreate(void* hnswIndexPtr, int M, int efConstruction, VectorArray vectors);

// ivfflat
int IvfflatTrain(VectorArray samples, int lists, void** ivfIndexPtr);
int IvfflatIndexCreate(void* ivfIndexPtr, VectorArray vectors);

// index build (general)
void IndexBuild(IndexType type, ConcurrentMemTable mt, uint32_t valid_rows, void** indexPtr, int M, int efConstruction, int lists);

// vector index search
topKVector* VectorIndexSearch(IndexType type, void *index_ptr, uint8_t *bitmap_ptr, uint32_t count, const float* query_vector, int k, int efs_nprobe);

// int IndexStore(void* indexPtr, const char *path);
// int HnswIndexLoad(void* hnswIndexPtr, knowhere::Json *search_conf, const char *path);
int IndexFree(void* indexPtr);
// void IndexDeserialize(void* index_string, void *indexPtr, void *confPtr);
// void IndexDeserializeAndSave(void* bin, IndexType index_type, int idx0, int idx1);
void IndexLoadAndSave(const char* path, IndexType index_type, void** indexPtr);
// void IndexSerializeAndFlush(void *indexPtr, Size *seg_size, const char* filename, dsm_segment **ret_seg);
// void IndexSerialize(void *indexPtr, Size *seg_size, dsm_segment **ret_seg, void **ret_bin_set);
void IndexSerialize(void *indexPtr, void **ret_bin_set);
void IndexBinarySetFlush(const char* filename, void *ret_bin_set);
Size BitmapSetLoadAndSave(const char* path, int idx0, int idx1, uint32_t count);

// brute force search
// void ComputeMultipleDistances(const void *vectors, uint32_t vector_num, uint32_t dim, const float *query_vector, float *distances);
float ComputeDistance(const float *a, const float *b, uint32_t dim);
topKVector* BruteForceSearch(const float* vectors, const float* query_vector, const uint8_t *bitmap, int count, int k, int dim);

// merge index
void MergeIndex(void *index_ptr, uint8_t *bitmap_ptr, int count, IndexType old_index_type, IndexType new_index_type, void **new_index_ptr, int *new_index_count, int M, int efConstruction, int lists);

// merge two indices
void MergeTwoIndices(void *index1_ptr, uint8_t *bitmap1_ptr, int count1, IndexType index1_type, float deletion_ratio1,
                     void *index2_ptr, uint8_t *bitmap2_ptr, int count2, IndexType index2_type, float deletion_ratio2,
                     void **merged_index_ptr, uint8_t **merged_bitmap_ptr, int *merged_count,
                     IndexType *merged_index_type, float *merged_deletion_ratio);


#ifdef __cplusplus
}
#endif

#endif