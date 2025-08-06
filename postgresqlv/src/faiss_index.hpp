#ifndef FAISS_INDEX_HPP
#define FAISS_INDEX_HPP

#ifdef __cplusplus
extern "C" {
#endif

#include "postgres.h"
#include "ivfflat.h"

int FaissIvfflatIndexCreate(void* ivf_index_ptr, int64_t lowest_vid, VectorArray vectors);
int FaissIvfflatTrain(Relation index, VectorArray samples, VectorArray centers, void** ivf_index_p);
int FaissIndexFree(void* index_p);
dsm_handle FaissIvfflatBuildAllocate(int dim, int nlist, void *vectors, void *bitmap, uint64_t lowest_vid, uint64_t highest_vid, int vector_num, size_t *index_size, dsm_segment **return_index_seg);
dsm_handle FaissIvfflatIndexStore(void* ivf_index_ptr, size_t *index_size, dsm_segment **return_index_seg);
topKVector* FaissIvfflatIndexSearch(void *serialized_index, uint32_t index_hdl, size_t index_size, const float *query_vector, int top_k, int nprobe);
void FaissComputeDistances(const void *vectors, uint32_t vector_num, uint32_t dim, const float *query_vector, float *distances);

int FaissHnswIndexInit(Relation index, int dimension, int M, int efConstruction, void** hnsw_index_ptr);
int FaissHnswIndexCreate(void* hnsw_index_ptr, int64_t lowest_vid, VectorArray vectors);
dsm_handle FaissHnswIndexBuildAllocate(int dim, int M, int efConstruction, void *vectors, void *bitmap, uint64_t lowest_vid, uint64_t highest_vid, int vector_num, size_t *index_size, dsm_segment **return_index_seg);
dsm_handle FaissHnswIndexStore(void* ivf_index_ptr, size_t *index_size, dsm_segment **return_index_seg);
topKVector* FaissHnswIndexSearch(void *serialized_index, uint32_t index_hdl, size_t index_size, const float *query_vector, int top_k, int efSearch);

#ifdef __cplusplus
}
#endif

#endif