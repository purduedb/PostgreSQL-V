#include "faiss_index.hpp"
#include "utils/memutils.h"
#include "storage/dsm.h"
#include "utils/timestamp.h"

#include <omp.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <algorithm>
#include <utility>
#include <queue>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/impl/io.h>
#include <faiss/utils/distances.h>
#include <faiss/Index.h>
#include <faiss/IndexIDMap.h>

#undef gettext
#undef dgettext
#undef ngettext
#undef dngettext

#include <immintrin.h> 
#include <fstream>
#include <string>
#include <sstream>
#include <iostream> 

// diskann .h files
#include "utils.h"
#include "filter_utils.h"
#include "index_factory.h"
#include "pq_flash_index.h"
#include "linux_aligned_file_reader.h"
#include "disk_utils.h"


using idx_t = faiss::idx_t;

#define DESERIALIZED_INDEX_CACHE_SIZE 100
faiss::Index *cached_deserialized_indexes[DESERIALIZED_INDEX_CACHE_SIZE];
uint32 cached_index_hdls[DESERIALIZED_INDEX_CACHE_SIZE];
int cached_num = 0;

static faiss::Index * 
get_cached_deserialized_index(uint32 hdl)
{
    for (int i = 0; i < DESERIALIZED_INDEX_CACHE_SIZE; i++)
    {
        if (cached_index_hdls[i] == hdl)
        {
            return cached_deserialized_indexes[i];
        }
    }
    return nullptr;    
}

/* 
 * Creates faiss index from vects
 *
 * Code taken from faiss/tutorial/cpp/2-IVFFlat.cpp
 */
// FIXME: is it okay to use faiss::IndexIVFFlat *  as the return value and the parameter?
extern "C" int
FaissIvfflatIndexCreate(void* ivf_index_ptr, int64_t lowest_vid, VectorArray vectors)
{
    elog(DEBUG1, "enter FaissIvfflatIndexCreate, lowest_vid = %d, vector_num = %d", lowest_vid, vectors->length);

    omp_set_num_threads(1);

    faiss::IndexIVFFlat* index = static_cast<faiss::IndexIVFFlat*>(ivf_index_ptr);
    assert(index->is_trained);

    std::vector<float> flat_data(vectors->length * vectors->dim);
    for (int64_t i = 0; i < vectors->length; ++i)
    {
        // Get the vector pointer
        Vector * vec_ptr = (Vector *) VectorArrayGet(vectors, i);

        // Copy into flat_data at the correct offset
        memcpy(&flat_data[i * vectors->dim], vec_ptr->x, sizeof(float) * vectors->dim);
    }
    
    idx_t* vids = new idx_t[vectors->length];
    for (size_t i = 0; i < vectors->length; i++)
    {
        vids[i] = (idx_t)lowest_vid + i;
    }

    index->add_with_ids(vectors->length, flat_data.data(), vids);

    delete[] vids;
    return 0;
}

extern "C" int
FaissIvfflatTrain(Relation index, VectorArray samples, VectorArray centers, void** ivf_index_p)
{
    elog(DEBUG1, "enter FaissIvfflatTrain");

    int dimension = centers->dim;
    int nlist = centers->maxlen;
    int numSamples = samples->length;

    elog(DEBUG1, "[FaissIvfflatTrain] dimension = %d, nlist = %d, numSamples = %d", dimension, nlist, numSamples);

    elog(NOTICE, "Init quantizer and faiss ivfflat indexes");
    // Initing indexes
    faiss::IndexFlatL2 *quantizer = new faiss::IndexFlatL2(dimension);
    faiss::IndexIVFFlat *ivf_index = new faiss::IndexIVFFlat(quantizer, dimension, nlist, faiss::METRIC_L2);

    // assert(!ivf_index->is_trained);

    // assert(samples->itemsize == sizeof(float) * dimension);
    elog(DEBUG1, "samples->itemsize = %d", samples->itemsize);
    float *data = (float *) samples->items;

    // flat the data
    std::vector<float> train_data(samples->length * samples->dim);
    for (int64_t i = 0; i < samples->length; ++i) {
        Vector* vec_ptr = (Vector *) VectorArrayGet(samples, i);
        memcpy(&train_data[i * samples->dim], vec_ptr->x, sizeof(float) * samples->dim);
    }

    elog(NOTICE, "Train faiss ivfflat index to create clusters");
    ivf_index->train(samples->length, train_data.data());
    assert(ivf_index->is_trained);

    *ivf_index_p = static_cast<void *>(ivf_index);
    return 0;
}


class StringWriter : public faiss::IOWriter {
public:
  std::vector<uint8_t> data;

  size_t operator()(const void *ptr, size_t size, size_t nitems) override
  {
    size_t bytes = size * nitems;
    data.insert(data.end(), (uint8_t *)ptr, (uint8_t *)ptr + bytes);
    return nitems;
  }

  int fileno() const
  {
    return -1; // Not applicable
  }
};

extern "C" int
FaissIndexFree(void* index_p)
{
    elog(DEBUG1, "enter FaissIndexFree");

    if (index_p != NULL) {
        elog(DEBUG1, "[FaissIndexFree] index_p = %p", index_p);

        try {
            delete static_cast<faiss::Index*>(index_p);
        } catch (const std::exception& e) {
            elog(ERROR, "[FaissIndexFree] exception during delete: %s", e.what());
        } catch (...) {
            elog(ERROR, "[FaissIndexFree] unknown crash during delete");
        }
    }

    return 0;
}

class StringReader : public faiss::IOReader {
public:
  std::string data;
  size_t position;

  StringReader(const std::string &data) : data(data), position(0) {}

  size_t operator()(void *ptr, size_t size, size_t nitems) override {
    size_t to_read = size * nitems;
    if (position + to_read > data.size()) {
      to_read = data.size() - position;
    }
    memcpy(ptr, data.data() + position, to_read);
    position += to_read;
    return to_read / size;
  }
};

/**
 * Serialize an index to a string.
 */
std::string 
serializeIndex(const faiss::Index *index)
{
  StringWriter writer;
  faiss::write_index(index, &writer);
  return std::string(writer.data.begin(), writer.data.end());
}

/**
 * Read an index from a string.
 */
faiss::Index *
deserializeIndex(const std::string &index_data)
{
  StringReader reader(index_data);
  faiss::Index *index = faiss::read_index(&reader);
  return index;
}

// TODO: in the current version, we still build index for vectors that are marked deleted in the bitmap
// return the fully serialized faiss index
extern "C" dsm_handle 
FaissIvfflatBuildAllocate(int dim, int nlist, void *vectors, uint64_t lowest_vid, uint64_t highest_vid, 
                          int vector_num, size_t *index_size, dsm_segment **return_index_seg)
{
    elog(DEBUG1, "enter FaissIvfflatBuildAllocate");
    faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(dim);
    faiss::IndexIVFFlat* index = new faiss::IndexIVFFlat(quantizer, dim, nlist);
    float * faiss_vectors = reinterpret_cast<float*>(vectors);
    Assert(!index->is_trained);
    index->train(vector_num, faiss_vectors);
    Assert(index->is_trained);

    idx_t* vids = new idx_t[vector_num];
    for (size_t i = 0; i < vector_num; i++)
    {
        vids[i] = lowest_vid + i;
    }

    index->add_with_ids(vector_num, faiss_vectors, vids);
    elog(DEBUG1, "[FaissIvfflatBuildAllocate] finish building index: ntotal = %ld", index->ntotal);
    free(vids);

    // convert and return the serialized index
    std::string index_data = serializeIndex(dynamic_cast<faiss::Index *> (index));
    dsm_segment *index_seg = dsm_create(index_data.size(), 0);
    if (index_seg == NULL)
        elog(ERROR, "Failed to allocate dynamic shared memory segment");
    void *dsm_addr = dsm_segment_address(index_seg);
    memcpy(dsm_addr, index_data.data(), index_data.size());
    dsm_pin_mapping(index_seg);
    *index_size = index_data.size();
    elog(DEBUG1, "finish storing the vector index");

    delete index;
    *return_index_seg = index_seg;
    return dsm_segment_handle(index_seg);
}

extern "C" dsm_handle
FaissIvfflatIndexStore(void* ivf_index_ptr, size_t *index_size, dsm_segment **return_index_seg)
{
    elog(DEBUG1, "enter FaissIvfflatIndexStore");

    faiss::IndexIVFFlat* index = static_cast<faiss::IndexIVFFlat*>(ivf_index_ptr);

    std::string index_data = serializeIndex(index);

    dsm_segment *index_seg = dsm_create(index_data.size(), 0);
    if (index_seg == NULL)
        elog(ERROR, "Failed to allocate dynamic shared memory segment");
    void *dsm_addr = dsm_segment_address(index_seg);
    memcpy(dsm_addr, index_data.data(), index_data.size());
    dsm_pin_mapping(index_seg);
    *index_size = index_data.size();
    if (return_index_seg)
    {
      *return_index_seg = index_seg;
    }
    dsm_handle index_hdl = dsm_segment_handle(index_seg);
    // cache the deserialize index
    cached_deserialized_indexes[cached_num] = dynamic_cast<faiss::IndexIVFFlat*>(deserializeIndex(index_data));

    cached_index_hdls[cached_num++] = index_hdl;
    return index_hdl;
}

extern "C" topKVector* 
FaissIvfflatIndexSearch(void *serialized_index, uint32_t index_hdl, size_t index_size, const float *query_vector, int top_k, int nprobe)
{
    // elog(DEBUG1, "enter FaissIvfflatIndexSearch, top_k = %d, nprobe = %d", top_k, nprobe);

    faiss::Index *index;

    if ((index = dynamic_cast<faiss::IndexIVFFlat*>(get_cached_deserialized_index(index_hdl))) == nullptr)
    {
        elog(DEBUG1, "[FaissIvfflatIndexSearch] the deserialized index is not cached");
        // Deserialize the index from the serialized data
        std::string index_data(reinterpret_cast<char*>(serialized_index), index_size);
        // cached the deserialized index
        index = dynamic_cast<faiss::IndexIVFFlat*>(deserializeIndex(index_data));
        cached_deserialized_indexes[cached_num] = index;
        cached_index_hdls[cached_num++] = index_hdl;
        elog(DEBUG1, "[FaissIvfflatIndexSearch] deserializes index");
    }

    if (index == nullptr) {
        elog(ERROR, "Failed to deserialize the Faiss index.");
        return nullptr;
    }

    // TODO: modify the nprobe
    ((faiss::IndexIVFFlat*) index)->nprobe = nprobe;
    // ((faiss::IndexIVFFlat*) index)->nprobe = (size_t) sqrt(((faiss::IndexIVFFlat*) index)->nlist);

    if (!index->is_trained)
        elog(ERROR, "FAISS index is not trained (required for IVF index)");
    
    if (index->ntotal == 0)
        elog(ERROR, "FAISS index contains no vectors.");

    // Prepare arrays to store the results
    std::vector<idx_t> indices(top_k);
    std::vector<float> distances(top_k);

    // Perform the search for the top_k nearest neighbors
    // TimestampTz start_time = GetCurrentTimestamp();
    index->search(1, query_vector, top_k, distances.data(), indices.data());
    // TimestampTz end_time = GetCurrentTimestamp();
    // long elapsed_ms = TimestampDifferenceMilliseconds(start_time, end_time);
    // elog(DEBUG1, "[FaissIvfflatIndexSearch] performed the search for the top_k nearest neighbors in %ld ms", elapsed_ms);

    // Allocate memory for the result structure
    topKVector *result = (topKVector *) palloc(sizeof(topKVector));
    result->num_results = top_k;

    result->vids = (int64_t *) palloc(top_k * sizeof(int64_t));
    result->distances = (float *) palloc(top_k * sizeof(float));

    for (int i = 0; i < top_k; i++) {
        result->vids[i] = static_cast<int64_t>(indices[i]);
        result->distances[i] = distances[i];
    }

    // Clean up (we cannot delete the serialized index since we cached it in the memory)
    // delete index;
    return result;
}

void
FaissComputeDistances(const void *vectors, uint32_t vector_num, uint32_t dim,
                               const float *query_vector, 
                               float *distances)
{
//   elog(DEBUG1, "enter FaissComputeDistances");
  Assert(vectors != NULL);
  Assert(query_vector != NULL);

  faiss::fvec_L2sqr_ny(distances, query_vector, (float *)vectors, dim, vector_num);
}

// ------------hnsw------------
// diskann index
std::unique_ptr<diskann::PQFlashIndex<float, uint32_t>> _pFlashIndex = nullptr;
std::shared_ptr<AlignedFileReader> reader = nullptr;
diskann::Metric metric = diskann::Metric::L2;
std::string fbin_path = "/ssd_root/liu4127/sift_1M_raw_vectors.fbin";
std::string index_path_prefix = "/ssd_root/liu4127/diskann/diskann_index_cohere_10M_b192_R68_L75_A1.2";
std::vector<uint32_t> nodes_to_cache;

extern "C" void
build_diskann_index()
{
    // step 0: build index
    elog(DEBUG1, "[build_diskann_index] step 0: build index");
    uint32_t R = 64;
    uint32_t L = 75;
    float B = 5;
    float M = 10;
    // set thread number
    // uint32_t num_threads = omp_get_num_procs();
    uint32_t num_threads = 1;
    uint32_t disk_PQ = 0;
    bool append_reorder_data = 0;
    uint32_t build_PQ = 0;
    uint32_t QD = 0;
    bool use_opq = 0;
    std::string codebook_prefix = "";
    bool use_filters = false;
    std::string label_file = "";
    std::string universal_label = "";
    uint32_t filter_threshold = 0;
    uint32_t Lf = 0;

    std::string params = std::string(std::to_string(R)) + " " + std::string(std::to_string(L)) + " " +
                         std::string(std::to_string(B)) + " " + std::string(std::to_string(M)) + " " +
                         std::string(std::to_string(num_threads)) + " " + std::string(std::to_string(disk_PQ)) + " " +
                         std::string(std::to_string(append_reorder_data)) + " " +
                         std::string(std::to_string(build_PQ)) + " " + std::string(std::to_string(QD));

    diskann::build_disk_index<float>(fbin_path.c_str(), index_path_prefix.c_str(), params.c_str(),
                                                        metric, use_opq, codebook_prefix, use_filters, label_file,
                                                        universal_label, filter_threshold, Lf);
    elog(DEBUG1, "[build_diskann_index] step 0 finished");
}

extern "C" void
load_diskann_index()
{
    // --------diskann--------
    // step 1: load index
    elog(DEBUG1, "[load_diskann_index] step 1: load index");
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    reader.reset(new LinuxAlignedFileReader());
#endif
    _pFlashIndex.reset(new diskann::PQFlashIndex<float, uint32_t>(reader, metric));
    int load_num_threads = 10;
    int res = _pFlashIndex->load(load_num_threads, index_path_prefix.c_str());
    if (res != 0)
    {
        elog(ERROR, "[load_diskann_index] An error occurs when loading the index");
    }
    elog(DEBUG1, "[load_diskann_index] step 1 finished");
    
    // step 2: cache medoid neighborhood
    elog(DEBUG1, "[load_diskann_index] step 2: cache medoid neighborhood");
    int num_nodes_to_cache = 322638;
    _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, nodes_to_cache);
    _pFlashIndex->load_cache_list(nodes_to_cache);
    nodes_to_cache.clear();
    nodes_to_cache.shrink_to_fit();
    elog(DEBUG1, "[load_diskann_index] step 2 finished");
}

extern "C" int
FaissHnswIndexInit(Relation index, int dimension, int M, int efConstruction, void** hnsw_index_ptr)
{
    elog(DEBUG1, "enter FaissHnswIndexInit, M = %d, efConstruction = %d", M, efConstruction);

    faiss::IndexHNSWFlat *faiss_hnsw_index = new faiss::IndexHNSWFlat(dimension, M);
    faiss_hnsw_index->hnsw.efConstruction = efConstruction;
    // Create IDMap wrapper on the heap
    faiss::IndexIDMap* map_index = new faiss::IndexIDMap(faiss_hnsw_index);

    // Return pointer
    *hnsw_index_ptr = static_cast<void*>(map_index);
    return 0;
}

static bool is_first_batch = true;
int32_t total_vectors_written = 0;

void write_vectors_to_fbin(const std::string &file_path,
                           const std::vector<float> &flat_data,
                           int64_t num_vectors,
                           int dim)
{
    std::ofstream out;

    if (is_first_batch) {
        // First time: open in write mode, write dimension and data
        out.open(file_path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!out) {
            elog(ERROR, "Failed to open %s for writing", file_path.c_str());
            return;
        }

        int32_t num_points = 0; // placeholder
        int32_t dim32 = static_cast<int32_t>(dim);
        // Write dimension and the total number once
        out.write(reinterpret_cast<const char *>(&num_points), sizeof(int32_t));
        out.write(reinterpret_cast<const char *>(&dim32), sizeof(int32_t));
        is_first_batch = false;
    } else {
        // Append mode for subsequent calls
        out.open(file_path, std::ios::binary | std::ios::out | std::ios::app);
        if (!out) {
            elog(ERROR, "Failed to open %s for appending", file_path.c_str());
            return;
        }
    }

    // Write vectors one by one (each is `dim` floats)
    for (int64_t i = 0; i < num_vectors; ++i) {
        out.write(reinterpret_cast<const char *>(&flat_data[i * dim]), sizeof(float) * dim);
    }
    total_vectors_written += num_vectors;

    out.close();
}

void finalize_fbin_header(const std::string &file_path)
{
    std::fstream out(file_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!out) {
        elog(ERROR, "Failed to reopen %s for header fix", file_path.c_str());
        return;
    }

    out.write(reinterpret_cast<const char *>(&total_vectors_written), sizeof(int32_t));
    out.close();
}


extern "C" int
FaissHnswIndexCreate(void* hnsw_index_ptr, int64_t lowest_vid, VectorArray vectors)
{
    elog(DEBUG1, "enter FaissHnswIndexCreate, lowest_vid = %d, vector_num = %d", lowest_vid, vectors->length);
    
    // omp_set_num_threads(1);

    faiss::IndexIDMap *map_index = static_cast<faiss::IndexIDMap *>(hnsw_index_ptr); 
    // faiss::IndexHNSWFlat *hnsw_index = dynamic_cast<faiss::IndexHNSWFlat*>(map_index->index);
    
    std::vector<float> flat_data(vectors->length * vectors->dim);
    for (int64_t i = 0; i < vectors->length; ++i)
    {
        // Get the vector pointer
        Vector * vec_ptr = (Vector *) VectorArrayGet(vectors, i);

        // Copy into flat_data at the correct offset
        memcpy(&flat_data[i * vectors->dim], vec_ptr->x, sizeof(float) * vectors->dim);
    }

    // ---------- diskann ------------
    write_vectors_to_fbin(fbin_path, flat_data, vectors->length, vectors->dim);
    
    idx_t* vids = new idx_t[vectors->length];
    for (size_t i = 0; i < vectors->length; i++)
    {
        vids[i] = (idx_t)lowest_vid + i;
    }

    elog(DEBUG1, "[FaissHnswIndexCreate] write vectors to fbin file, num = %d", vectors->length);
    // ---------- diskann ------------
    
    delete[] vids;
    return 0;
}

extern "C" dsm_handle 
FaissHnswIndexBuildAllocate(int dim, int M, int efConstruction, void *vectors, uint64_t lowest_vid, uint64_t highest_vid, 
                            int vector_num, size_t *index_size, dsm_segment **return_index_seg)
{
    elog(DEBUG1, "enter FaissHnswIndexBuildStore");

    faiss::IndexHNSWFlat *index = new faiss::IndexHNSWFlat(dim, M);
    index->hnsw.efConstruction = efConstruction;

    float * faiss_vectors = reinterpret_cast<float*>(vectors);
    idx_t* vids = new idx_t[vector_num];
    for (size_t i = 0; i < vector_num; i++)
    {
        vids[i] = lowest_vid + i;
    }

    index->add_with_ids(vector_num, faiss_vectors, vids);
    elog(DEBUG1, "[FaissHnswIndexBuildStore] finish building index: ntotal = %ld", index->ntotal);
    free(vids);

    std::string index_data = serializeIndex(dynamic_cast<faiss::Index *> (index));
    dsm_segment *index_seg = dsm_create(index_data.size(), 0);
    if (index_seg == NULL)
        elog(ERROR, "Failed to allocate dynamic shared memory segment");
    void *dsm_addr = dsm_segment_address(index_seg);
    memcpy(dsm_addr, index_data.data(), index_data.size());
    dsm_pin_mapping(index_seg);
    *index_size = index_data.size();
    elog(DEBUG1, "finish storing the vector index");

    delete index;
    *return_index_seg = index_seg;
    return dsm_segment_handle(index_seg);
}

extern "C" dsm_handle
FaissHnswIndexStore(void* hnsw_index_ptr, size_t *index_size, dsm_segment **return_index_seg)
{
    elog(DEBUG1, "enter FaissHnswIndexStore");

    // ---------- diskann ------------
    finalize_fbin_header(fbin_path);
    build_diskann_index();
    load_diskann_index();
    // ---------- diskann ------------
    
    faiss::Index* index = static_cast<faiss::Index*>(hnsw_index_ptr);

    std::string index_data = serializeIndex(index);

    dsm_segment *index_seg = dsm_create(index_data.size(), 0);
    if (index_seg == NULL)
        elog(ERROR, "Failed to allocate dynamic shared memory segment");
    void *dsm_addr = dsm_segment_address(index_seg);
    memcpy(dsm_addr, index_data.data(), index_data.size());
    dsm_pin_mapping(index_seg);
    *index_size = index_data.size();
    if (return_index_seg)
    {
      *return_index_seg = index_seg;
    }
    dsm_handle index_hdl = dsm_segment_handle(index_seg);
    // cache the deserialize index
    cached_deserialized_indexes[cached_num] = deserializeIndex(index_data);
    
    cached_index_hdls[cached_num++] = index_hdl;
    elog(DEBUG1, "completed FaissHnswIndexStore");
    return index_hdl;
}


extern "C" topKVector* 
FaissHnswIndexSearch(void *serialized_index, uint32_t index_hdl, size_t index_size, const float *query_vector, int top_k, int efSearch)
{
    // elog(DEBUG1, "enter FaissHnswIndexSearch");
    
    // faiss::IndexIDMap *map_index;

    // if ((map_index = dynamic_cast<faiss::IndexIDMap*>(get_cached_deserialized_index(index_hdl))) == nullptr)
    // {
    //     elog(DEBUG1, "[FaissHnswIndexSearch] the deserialized index is not cached");
    //     // Deserialize the index from the serialized data
    //     std::string index_data(reinterpret_cast<char*>(serialized_index), index_size);
    //     // cached the deserialized index
    //     map_index = dynamic_cast<faiss::IndexIDMap*>(deserializeIndex(index_data));
    //     cached_deserialized_indexes[cached_num] = map_index;
    //     cached_index_hdls[cached_num++] = index_hdl;
    //     elog(DEBUG1, "[FaissHnswIndexSearch] deserializes index");
    // }

    // if (map_index == nullptr) {
    //     elog(ERROR, "[FaissHnswIndexSearch] Failed to deserialize the Faiss index.");
    //     return nullptr;
    // }

    // faiss::IndexHNSWFlat* hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(map_index->index);

    // ((faiss::IndexHNSWFlat*) hnsw)->hnsw.efSearch = efSearch;
    // // elog(DEBUG1, "[FaissHnswIndexSearch] set the efsearch to %d", efSearch);

    // if (map_index->ntotal == 0)
    //     elog(ERROR, "[FaissHnswIndexSearch] FAISS index contains no vectors.");

    // // Prepare arrays to store the resultsq
    // std::vector<idx_t> indices(top_k);
    // std::vector<float> distances(top_k);

    // // Perform the search for the top_k nearest neighbors
    // // TimestampTz start_time = GetCurrentTimestamp();
    // map_index->search(1, query_vector, top_k, distances.data(), indices.data());
    // // TimestampTz end_time = GetCurrentTimestamp();
    // // long elapsed_ms = TimestampDifferenceMilliseconds(start_time, end_time);
    // // elog(DEBUG1, "[FaissHnswIndexSearch] performed the search for the top_k nearest neighbors in %ld ms", elapsed_ms);

    // // Allocate memory for the result structure
    // topKVector *result = (topKVector *) palloc(sizeof(topKVector));
    // result->num_results = top_k;

    // result->vids = (int64_t *) palloc(top_k * sizeof(int64_t));
    // result->distances = (float *) palloc(top_k * sizeof(float));

    // for (int i = 0; i < top_k; i++) {
    //     result->vids[i] = static_cast<int64_t>(indices[i]);
    //     result->distances[i] = distances[i];

    // }

    // // Clean up (we cannot delete the serialized index since we cached it in the memory)
    // // delete index;
    // return result;

    // --------diskann--------
    // elog(DEBUG1, "enter FaissHnswIndexSearch");

    if (_pFlashIndex == nullptr)
    {
        load_diskann_index();   
    }

    std::vector<uint64_t> query_result_ids_64(top_k, 0);
    std::vector<float> query_result_dists(top_k, 0);

    // _pFlashIndex->cached_beam_search(
    //     query_vector,        // query pointer
    //     top_k,                // top-k
    //     efSearch,               // L = search list size
    //     query_result_ids_64.data(),   // output: neighbor IDs
    //     query_result_dists.data(), // output: distances
    //     2                // beam width
    // );

    uint32_t optimized_beamwidth = 2;
    diskann::QueryStats stats;

    _pFlashIndex->cached_beam_search(query_vector, top_k, efSearch,
                                                 query_result_ids_64.data(),
                                                 query_result_dists.data(),
                                                 optimized_beamwidth, false, &stats);

    // Allocate result struct (PostgreSQL palloc)
    topKVector *result = (topKVector *) palloc(sizeof(topKVector));
    result->num_results = top_k;

    // Allocate memory for IDs and distances
    result->vids = (int64_t *) palloc(top_k * sizeof(int64_t));
    result->distances = (float *) palloc(top_k * sizeof(float));

    // Copy results
    for (int i = 0; i < top_k; i++) {
        result->vids[i] = static_cast<int64_t>(query_result_ids_64[i]);
        result->distances[i] = query_result_dists[i];
    }

    return result;
    // --------diskann--------
}
