#include "vectorindeximpl.hpp"
#include "knowhere/comp/index_param.h"
#include "lsmindex.h"
#include "utils/elog.h"
#include "vector.h"
// #include "lsm_segment.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <queue>
// TODO: for evaluation
#include <chrono>
#include <iostream>

// --- Fix gettext macro collisions when some headers pull in <libintl.h> ---
#ifdef gettext
#undef gettext
#endif
#ifdef dgettext
#undef dgettext
#endif
#ifdef ngettext
#undef ngettext
#endif
#ifdef dngettext
#undef dngettext
#endif

// Now **temporarily** remove PG's severities only around glog/Knowhere.
#if defined(INFO) || defined(WARNING) || defined(ERROR)
  #pragma push_macro("INFO")
  #pragma push_macro("WARNING")
  #pragma push_macro("ERROR")
  #undef INFO
  #undef WARNING
  #undef ERROR
#endif

// Keep glog's own severities usable without abbreviations conflict
#define GLOG_NO_ABBREVIATED_SEVERITIES 1

// knowhere
#include "knowhere/bitsetview.h"
#include "knowhere/index/index.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/binaryset.h"
#include "simd/hook.h"
#include "knowhere/comp/brute_force.h"


// Restore PG's macros so elog.h and friends keep working everywhere else.
#if defined(__clang__) || defined(__GNUC__)
  #pragma pop_macro("ERROR")
  #pragma pop_macro("WARNING")
  #pragma pop_macro("INFO")
#endif

// TODO: support different distance metrics

struct FileIOWriter {
    std::fstream fs;
    std::string name;

    explicit FileIOWriter(const std::string& fname) {
        name = fname;
        fs = std::fstream(name, std::ios::out | std::ios::binary);
    }

    ~FileIOWriter() {
        fs.close();
    }

    size_t
    operator()(void* ptr, size_t size) {
        fs.write(reinterpret_cast<char*>(ptr), size);
        return size;
    }
};

struct FileIOReader {
    std::fstream fs;
    std::string name;

    explicit FileIOReader(const std::string& fname) {
        name = fname;
        fs = std::fstream(name, std::ios::in | std::ios::binary);
    }

    ~FileIOReader() {
        fs.close();
    }

    size_t
    operator()(void* ptr, size_t size) {
        fs.read(reinterpret_cast<char*>(ptr), size);
        return size;
    }

    size_t
    size() {
        fs.seekg(0, fs.end);
        size_t len = fs.tellg();
        fs.seekg(0, fs.beg);
        return len;
    }
};

static void
write_binary_set(knowhere::BinarySet& binary_set, const std::string& filename) {
    FileIOWriter writer(filename);
    
    const auto& m = binary_set.binary_map_;
    for (auto it = m.begin(); it != m.end(); ++it) {
        const std::string& name = it->first;
        size_t name_size = name.length();
        const knowhere::BinaryPtr data = it->second;
        size_t data_size = data->size;

        writer(&name_size, sizeof(name_size));
        writer(&data_size, sizeof(data_size));
        writer((void*)name.c_str(), name_size);
        writer(data->data.get(), data_size);
    }
}

static void
read_binary_set(knowhere::BinarySet& binary_set, const std::string& filename) {
    FileIOReader reader(filename);
    int64_t file_size = reader.size();
    if (file_size < 0) {
        throw std::exception();
    }

    int64_t offset = 0;
    while (offset < file_size) {
        size_t name_size, data_size;
        reader(&name_size, sizeof(size_t));
        offset += sizeof(size_t);
        reader(&data_size, sizeof(size_t));
        offset += sizeof(size_t);

        std::string name;
        name.resize(name_size);
        reader((void*)name.data(), name_size);
        offset += name_size;
        auto data = new uint8_t[data_size];
        reader(data, data_size);
        offset += data_size;

        std::shared_ptr<uint8_t[]> data_ptr(data);
        binary_set.Append(name, data_ptr, data_size);
    }
}

static void
write_index(knowhere::Index<knowhere::IndexNode>& index, const std::string& filename) {
    knowhere::BinarySet binary_set;
    index.Serialize(binary_set);

    write_binary_set(binary_set, filename);
}

static void
read_index(knowhere::Index<knowhere::IndexNode>& index, const std::string& filename, const knowhere::Json& conf) {
    knowhere::BinarySet binary_set;
    read_binary_set(binary_set, filename);

    index.Deserialize(binary_set, conf);
}

extern "C" int
HnswIndexInit(int dimension, int M, int efConstruction, void** hnswIndexPtr)
{
    elog(DEBUG1, "enter HnswIndexInit, M = %d, efConstruction = %d", M, efConstruction);

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto kindex = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    knowhere::Index<knowhere::IndexNode> * kindex_ptr = new knowhere::Index<knowhere::IndexNode>(kindex);
    // Return pointer
    *hnswIndexPtr = static_cast<void*>(kindex_ptr);
    return 0;
}

extern "C" int
HnswIndexcreate(void* hnswIndexPtr, int M, int efConstruction, VectorArray vectors)
{
    elog(DEBUG1, "enter HnswIndexCreate, vector_num = %d", vectors->length);

    // knowhere
    const int64_t nb  = vectors->length;
    const int64_t dim = vectors->dim;

    // 1) Materialize a contiguous [nb x dim] float buffer
    std::unique_ptr<float[]> buf(new float[nb * dim]);

    for (int64_t i = 0; i < nb; ++i) {
        Vector* vec_ptr = (Vector*) VectorArrayGet(vectors, i); // vec_ptr->x points to float[dim]
        std::memcpy(buf.get() + i * dim, vec_ptr->x, sizeof(float) * dim);
    }
    auto dataset = knowhere::GenDataSet(nb, dim, buf.get());

    knowhere::Index<knowhere::IndexNode> *index = static_cast<knowhere::Index<knowhere::IndexNode>*>(hnswIndexPtr);

    // configuration
    knowhere::Json conf;
    conf[knowhere::meta::ROWS] = vectors->length;
    conf[knowhere::meta::DIM] = vectors->dim;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    conf[knowhere::indexparam::M] = M;
    conf[knowhere::indexparam::EFCONSTRUCTION] = efConstruction;

    if (index->Count() == -1)
    {
        auto build_res = index->Build(dataset, conf);
        Assert(build_res == knowhere::Status::success);
        Assert(index->Count() == vectors->length);
    }
    else
    {
        auto add_res = index->Add(dataset, conf);
        Assert(add_res == knowhere::Status::success);
    }
    return 0;
}

extern "C" int 
IvfflatTrain(VectorArray samples, int lists, void** ivfIndexPtr)
{
    elog(DEBUG1, "enter IvfflatTrain");
    
    const int64_t dim = samples->dim;
    const int64_t nlist = lists;
    const int64_t nsamples = samples->length;
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto kindex = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, version).value();
    knowhere::Index<knowhere::IndexNode> *kindex_ptr = new knowhere::Index<knowhere::IndexNode>(kindex);

    // configuration
    knowhere::Json conf;
    conf[knowhere::meta::ROWS] = nsamples;
    conf[knowhere::meta::DIM] = dim;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    conf[knowhere::indexparam::NLIST] = nlist;

    // generate train dataset
    std::unique_ptr<float[]> buf(new float[nsamples * dim]);
    for (int64_t i = 0; i < nsamples; ++i) {
        Vector* vec_ptr = (Vector *) VectorArrayGet(samples, i);
        std::memcpy(buf.get() + i * dim, vec_ptr->x, sizeof(float) * dim);
    }
    auto dataset = knowhere::GenDataSet(nsamples, dim, buf.get());

    auto train_res = kindex_ptr->Train(dataset, conf);
    Assert(train_res == knowhere::Status::success);

    *ivfIndexPtr = static_cast<void*>(kindex_ptr);
    return 0;
}

extern "C" int
IvfflatIndexCreate(void* ivfIndexPtr, VectorArray vectors)
{
    elog(DEBUG1, "enter IvfflatIndexCreate, vector_num = %d", vectors->length);

    const int64_t nb  = vectors->length;
    const int64_t dim = vectors->dim;

    std::unique_ptr<float[]> buf(new float[nb * dim]);

    for (int64_t i = 0; i < nb; ++i) {
        Vector* vec_ptr = (Vector*) VectorArrayGet(vectors, i); // vec_ptr->x points to float[dim]
        std::memcpy(buf.get() + i * dim, vec_ptr->x, sizeof(float) * dim);
    }
    auto dataset = knowhere::GenDataSet(nb, dim, buf.get());

    knowhere::Index<knowhere::IndexNode> *index = static_cast<knowhere::Index<knowhere::IndexNode>*>(ivfIndexPtr);

    knowhere::Json conf;
    conf[knowhere::meta::ROWS] = nb;
    conf[knowhere::meta::DIM] = dim;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;

    if (index->Count() == 0)
    {
        auto build_res = index->Build(dataset, conf);
        Assert(build_res == knowhere::Status::success);
    }
    else
    {
        auto add_res = index->Add(dataset, conf);
        Assert(add_res == knowhere::Status::success);
    }
    return 0;
}

extern "C" void
IndexBuild(IndexType type, ConcurrentMemTable mt, uint32_t valid_rows, void** indexPtr, int M, int efConstruction, int lists)
{
    elog(DEBUG1, "enter IndexBuild, type = %d, relation = %d, segment id = %d, valid_rows = %d, M = %d, efConstruction = %d, lists = %d",
         type, mt->rel, mt->memtable_id, valid_rows, M, efConstruction, lists);
    
    // initialize
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    knowhere::Index<knowhere::IndexNode> kindex;
    switch (type)
    {
        case FLAT:
            kindex = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version).value();
            break;
        case HNSW:
            kindex = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
            break;
        case IVFFLAT:
            kindex = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, version).value();
            break;
    }
    knowhere::Index<knowhere::IndexNode> * kindex_ptr = new knowhere::Index<knowhere::IndexNode>(kindex);

    // create
    auto dataset = knowhere::GenDataSet(valid_rows, mt->dim, VEC_BASE_FROM_MT(mt));
    knowhere::Index<knowhere::IndexNode> *index = static_cast<knowhere::Index<knowhere::IndexNode>*>(kindex_ptr);

    // configuration
    knowhere::Json conf;
    conf[knowhere::meta::ROWS] = valid_rows;
    conf[knowhere::meta::DIM] = mt->dim;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    switch (type)
    {
        case FLAT:
            break;
        case IVFFLAT:
            conf[knowhere::indexparam::NLIST] = lists;
            break;
        case HNSW:
            conf[knowhere::indexparam::M] = M;
            conf[knowhere::indexparam::EFCONSTRUCTION] = efConstruction;
            break;
    }

    auto build_res = index->Build(dataset, conf);
    Assert(build_res == knowhere::Status::success);
    Assert(index->Count() == valid_rows);

    *indexPtr = static_cast<void*>(kindex_ptr);
}

extern "C" int
IndexFree(void* indexPtr)
{
    // elog(DEBUG1, "enter IndexFree");
    if (indexPtr != nullptr)
    {
        auto index = static_cast<knowhere::Index<knowhere::IndexNode>*>(indexPtr);
        delete index;
    }
    return 0;
}

// FIXME: handle the situaltion when k > total vectors in the index
static topKVector*
VectorIndexSearchImpl(IndexType type, void* indexPtr, knowhere::BitsetView bitset_view, uint32_t count, const float* query_vector, int k, int efs_nprobe)
{
    // TODO: for evaluation
    // Timing instrumentation
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // elog(DEBUG1, "enter VectorIndexSearch, type = %d, efs_nprobe = %d, k = %d, count = %d", type, efs_nprobe, k, count);
    knowhere::Index<knowhere::IndexNode> *index = static_cast<knowhere::Index<knowhere::IndexNode>*>(indexPtr);
    if (index->Count() <= 0)
    {
        elog(ERROR, "IvfflatIndexSearch: the index is empty");
        return nullptr;
    }
    // elog(DEBUG1, "[VectorIndexSearchImpl] index->Count() = %d", index->Count());

    // configuration
    knowhere::Json conf;
    conf[knowhere::meta::TOPK] = k;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    switch (type)
    {
        case FLAT:
            break;
        case IVFFLAT:
            conf[knowhere::indexparam::NPROBE] = efs_nprobe;
            break;
        case HNSW:
            conf[knowhere::indexparam::EF] = efs_nprobe;
            break;
        default:
            elog(ERROR, "VectorIndexSearch: unsupported index type %d", type);
    }

    // generate query dataset
    int dim = index->Dim();
    auto dataset = knowhere::GenDataSet(1, dim, query_vector);

    // conduct search
    auto res = index->Search(dataset, conf, bitset_view);

    // convert knowhere::Dataset to topKVector
    topKVector* topk_result = (topKVector *) palloc(sizeof(topKVector));
    topk_result->num_results = k;
    topk_result->distances = (float *) palloc(sizeof(float) * k);
    topk_result->vids = (int64_t *) palloc(sizeof(int64_t) * k);

    const int64_t* ids = res.value()->GetIds();
    const float*   dis = res.value()->GetDistance();

    std::memcpy(topk_result->distances, dis, sizeof(float) * k);
    std::memcpy(topk_result->vids, ids, sizeof(int64_t) * k);

    // TODO: for evaluation
    // Timing instrumentation - calculate and log execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Static arrays to store last 1000 execution times
    int interval = 10000;
    static long execution_times[10000];
    static int call_count = 0;
    static int array_index = 0;
    
    // Store current execution time
    execution_times[array_index] = duration.count();
    array_index = (array_index + 1) % interval;
    call_count++;
    
    // Log statistics every 1000 calls
    if (call_count % interval == 0) {
        long total_time = 0;
        long min_time = LONG_MAX;
        long max_time = 0;
        
        // Calculate stats for the last 1000 calls
        for (int i = 0; i < interval; i++) {
            total_time += execution_times[i];
            if (execution_times[i] < min_time) min_time = execution_times[i];
            if (execution_times[i] > max_time) max_time = execution_times[i];
        }
        
        double avg_time = static_cast<double>(total_time) / (double)interval;
        elog(DEBUG1, "[VectorIndexSearchImpl] Stats for last %d calls - Avg: %.2fμs, Min: %ldμs, Max: %ldμs", 
             interval, avg_time, min_time, max_time);
    }
    // TODO: for evaluation (end here)

    return topk_result;
}

extern "C" topKVector*
VectorIndexSearch(IndexType type, void *index_ptr, uint8_t *bitmap_ptr, uint32_t count, const float* query_vector, int k, int efs_nprobe)
{
    // elog(DEBUG1, "enter VectorIndexSearch, type = %d, index_ptr = %p, bitmap_ptr = %p, count = %d, query_vector = %p, k = %d, efs_nprobe = %d", type, index_ptr, bitmap_ptr, count, query_vector, k, efs_nprobe);
    knowhere::BitsetView bitset_view(bitmap_ptr, count);
    return VectorIndexSearchImpl(type, index_ptr, bitset_view, count, query_vector, k, efs_nprobe);
}

static void
FreeBinarySet(void* bin)
{
    delete static_cast<knowhere::BinarySet*>(bin);
}

static inline Size
get_required_size(knowhere::BinarySet* bs) {
    if (!bs) return 0;
    Size total = 0;
    const auto& m = bs->binary_map_;
    for (const auto& kv : m) {
        const std::string& name = kv.first;
        const knowhere::BinaryPtr data = kv.second; // shared_ptr<uint8_t[]> + size
        total += sizeof(size_t)                 // name_size
               + sizeof(size_t)                 // data_size
               + name.size()                    // name bytes
               + data->size;                    // data bytes
    }
    // Sentinel {0,0} so a size-less reader can stop.
    total += sizeof(size_t) + sizeof(size_t);
    // elog(DEBUG1, "[get_required_size] total size = %ld", total);
    return total;
}

static void
convert_binary_set_to_string(knowhere::BinarySet* bs, void* buf) {
    // Caller must allocate buf with at least get_total_size(bs) bytes.
    uint8_t* p = reinterpret_cast<uint8_t*>(buf);
    const auto& m = bs->binary_map_;
    for (const auto& kv : m) {
        const std::string& name = kv.first;
        const knowhere::BinaryPtr data = kv.second;

        const size_t name_size = name.size();
        const size_t data_size = data->size;

        std::memcpy(p, &name_size, sizeof(size_t)); p += sizeof(size_t);
        std::memcpy(p, &data_size, sizeof(size_t)); p += sizeof(size_t);

        if (name_size) {
            std::memcpy(p, name.data(), name_size);
            p += name_size;
        }
        if (data_size) {
            std::memcpy(p, data->data.get(), data_size);
            p += data_size;
        }
    }

    // Write sentinel {0,0}
    const size_t zero = 0;
    std::memcpy(p, &zero, sizeof(size_t)); p += sizeof(size_t);
    std::memcpy(p, &zero, sizeof(size_t)); p += sizeof(size_t);
}

static void
convert_string_to_binary_set(void* buf, knowhere::BinarySet* bs) {
    // Expects a buffer produced by convert_binary_set_to_string (ends with sentinel {0,0}).
    const uint8_t* p = reinterpret_cast<const uint8_t*>(buf);

    while (true) {
        size_t name_size = 0, data_size = 0;

        std::memcpy(&name_size, p, sizeof(size_t)); p += sizeof(size_t);
        std::memcpy(&data_size, p, sizeof(size_t)); p += sizeof(size_t);

        // Sentinel: stop.
        if (name_size == 0 && data_size == 0) {
            elog(DEBUG1, "[convert_string_to_binary_set] sentinel reached at offset = %ld", (Size) (p - (const uint8_t*)buf));
            break;
        }

        std::string name;
        name.resize(name_size);
        if (name_size) {
            std::memcpy(&name[0], p, name_size);
            p += name_size;
        }

        // Allocate and copy payload.
        std::shared_ptr<uint8_t[]> data(new uint8_t[data_size],
                                        std::default_delete<uint8_t[]>());
        if (data_size) {
            std::memcpy(data.get(), p, data_size);
            p += data_size;
        }
        bs->Append(name, data, data_size);
    }
}

// extern "C" void
// IndexDeserialize(void* index_string, void *indexPtr, void *confPtr)
// {
//     elog(DEBUG1, "enter IndexDeserialize, index_string = %p, indexPtr = %p, confPtr = %p", index_string, indexPtr, confPtr);
//     knowhere::BinarySet binary_set;
//     convert_string_to_binary_set(index_string, &binary_set);

//     auto* index = static_cast<knowhere::Index<knowhere::IndexNode>*>(indexPtr);
//     auto* conf = static_cast<knowhere::Json*>(confPtr);

//     if (conf == nullptr || index == nullptr) {
//         elog(ERROR, "[IndexDeserialize] invalid input pointers");
//     }

//     index->Deserialize(binary_set, *conf);
// }

// extern "C" void
// IndexSerializeAndFlush(void *indexPtr, Size *seg_size, const char* filename, dsm_segment **ret_seg)
// {
//     auto* index = static_cast<knowhere::Index<knowhere::IndexNode>*>(indexPtr);
//     // create a binary set
//     knowhere::BinarySet binary_set;
//     index->Serialize(binary_set);
//     // write the index
//     write_binary_set(binary_set, filename);

//     *seg_size = get_required_size(&binary_set);
//     dsm_segment *index_seg = dsm_create(*seg_size, 0);
//     if (index_seg == NULL)
//         elog(ERROR, "[IndexSerializeAndFlush] Failed to allocate dynamic shared memory segment");
//     void *dsm_addr = dsm_segment_address(index_seg);

//     convert_binary_set_to_string(&binary_set, dsm_addr);
//     *ret_seg = index_seg;
// }

// extern "C" void
// IndexSerialize(void *indexPtr, Size *seg_size, dsm_segment **ret_seg, void **ret_bin_set)
// {
//     auto* index = static_cast<knowhere::Index<knowhere::IndexNode>*>(indexPtr);

//     // create a binary set
//     knowhere::BinarySet *bins = new knowhere::BinarySet();
//     index->Serialize(*bins);

//     *seg_size = get_required_size(bins);
//     dsm_segment *index_seg = dsm_create(*seg_size, 0);
//     if (index_seg == NULL)
//         elog(ERROR, "[IndexSerialize] Failed to allocate dynamic shared memory segment");
//     void *dsm_addr = dsm_segment_address(index_seg);

//     convert_binary_set_to_string(bins, dsm_addr);
//     *ret_seg = index_seg;
//     *ret_bin_set = static_cast<void*>(bins);
// }

extern "C" void
IndexSerialize(void *indexPtr, void **ret_bin_set)
{
    auto* index = static_cast<knowhere::Index<knowhere::IndexNode>*>(indexPtr);

    // create a binary set
    knowhere::BinarySet *bins = new knowhere::BinarySet();
    index->Serialize(*bins);
    *ret_bin_set = static_cast<void*>(bins);
}

// the index binary set will be free in this function
extern "C" void
IndexBinarySetFlush(const char* filename, void *ret_bin_set)
{
    knowhere::BinarySet *binary_set = static_cast<knowhere::BinarySet*>(ret_bin_set);
    write_binary_set(*binary_set, filename);
    FreeBinarySet(ret_bin_set);
}

// extern "C" void
// IndexDeserializeAndSave(void* bin, IndexType index_type, int idx0, int idx1)
// {
//     elog(DEBUG1, "enter IndexDeserializeAndSave, index_type = %d, idx0 = %d, idx1 = %d", index_type, idx0, idx1);

//     auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
//     switch (index_type)
//     {
//         case FLAT:
//         {
//             auto kindex = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version).value();
//             knowhere_index[idx0][idx1] = new knowhere::Index<knowhere::IndexNode>(kindex);
//             break;
//         }
//         case IVFFLAT:
//         {
//             auto kindex = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, version).value();
//             knowhere_index[idx0][idx1] = new knowhere::Index<knowhere::IndexNode>(kindex);
//             break;
//         }
//         case HNSW:
//         {
//             auto kindex = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
//             knowhere_index[idx0][idx1] = new knowhere::Index<knowhere::IndexNode>(kindex);
//             break;
//         }
//         default:
//         {
//             elog(DEBUG1, "[IndexDeserializeAndSave] Unsupported index type");
//             break;
//         }
//     }
//     knowhere_json[idx0][idx1] = new knowhere::Json();
//     IndexDeserialize(bin, knowhere_index[idx0][idx1], knowhere_json[idx0][idx1]);
// }

extern "C" void 
IndexLoadAndSave(const char* path, IndexType index_type, void** indexPtr)
{
    elog(DEBUG1, "enter IndexLoadAndSave, index_type = %d", index_type);
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    
    knowhere::Json conf;
    
    switch (index_type)
    {
        case FLAT:
        {
            auto kindex = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version).value();
            read_index(kindex, path, conf);            
            auto* index = new knowhere::Index<knowhere::IndexNode>(kindex);
            *indexPtr = static_cast<void*>(index);
            break;
        }
        case IVFFLAT:
        {
            auto kindex = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, version).value();
            read_index(kindex, path, conf);
            auto* index = new knowhere::Index<knowhere::IndexNode>(kindex);
            *indexPtr = static_cast<void*>(index);
            break;
        }
        case HNSW:
        {
            auto kindex = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
            read_index(kindex, path, conf);
            auto* index = new knowhere::Index<knowhere::IndexNode>(kindex);
            *indexPtr = static_cast<void*>(index);
            break;
        }
        default:
        {
            elog(DEBUG1, "[IndexDeserializeAndSave] Unsupported index type");
            break;
        }
    }
}

// ------------------ brute scan implementation ------------------

// currently a native implementation
extern "C" void
ComputeMultipleDistances(const void *vectors, uint32_t vector_num, uint32_t dim,
                               const float *query_vector, 
                               float *distances)
{
//   elog(DEBUG1, "enter FaissComputeDistances");
  Assert(vectors != NULL);
  Assert(query_vector != NULL);

  faiss::fvec_L2sqr_ny(distances, query_vector, (float *)vectors, dim, vector_num);
}

extern "C" float
ComputeDistance(const float *a, const float *b, uint32_t dim)
{
    Assert(a != NULL);
    Assert(b != NULL);

    return faiss::fvec_L2sqr(a, b, dim);
}

extern "C" topKVector*
BruteForceSearch(const float* vectors, const float* query_vector, const uint8_t *bitmap, int count, int k, int dim)
{
    // TODO: for evaluation
    // Timing instrumentation
    auto start_time = std::chrono::high_resolution_clock::now();

    auto query_dataset = knowhere::GenDataSet(1, dim, query_vector);
    auto base_dataset = knowhere::GenDataSet(count, dim, vectors);
    knowhere::Json conf;
    conf[knowhere::meta::TOPK] = k;
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    conf[knowhere::meta::DIM] = dim;
    conf[knowhere::meta::RADIUS] = 10.0;
    
    // generate bitset view
    knowhere::BitsetView bitset_view(bitmap, count);
    auto res = knowhere::BruteForce::Search<knowhere::fp32>(base_dataset, query_dataset, conf, bitset_view);
    
    // Convert knowhere::Dataset to topKVector
    topKVector* topk_result = (topKVector *) palloc(sizeof(topKVector));
    
    if (res.has_value()) {
        const int64_t* ids = res.value()->GetIds();
        const float* distances = res.value()->GetDistance();
        int actual_k = Min(k, count);
        
        topk_result->num_results = actual_k;
        topk_result->distances = (float *) palloc(sizeof(float) * actual_k);
        topk_result->vids = (int64_t *) palloc(sizeof(int64_t) * actual_k);
        
        // Copy results
        for (int i = 0; i < actual_k; i++) {
            topk_result->distances[i] = distances[i];
            topk_result->vids[i] = ids[i];
        }
    } else {
        // Handle error case - return empty result
        topk_result->num_results = 0;
        topk_result->distances = nullptr;
        topk_result->vids = nullptr;
    }

    // TODO: for evaluation
    // Timing instrumentation - calculate and log execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Static arrays to store last 1000 execution times
    int interval = 10000;
    static long execution_times[10000];
    static int call_count = 0;
    static int array_index = 0;
    
    // Store current execution time
    execution_times[array_index] = duration.count();
    array_index = (array_index + 1) % interval;
    call_count++;
    
    // Log statistics every 1000 calls
    if (call_count % interval == 0) {
        long total_time = 0;
        long min_time = LONG_MAX;
        long max_time = 0;
        
        // Calculate stats for the last 1000 calls
        for (int i = 0; i < interval; i++) {
            total_time += execution_times[i];
            if (execution_times[i] < min_time) min_time = execution_times[i];
            if (execution_times[i] > max_time) max_time = execution_times[i];
        }
        
        double avg_time = static_cast<double>(total_time) / (double)interval;
        elog(DEBUG1, "[BruteForceSearch] Stats for last %d calls - Avg: %.2fμs, Min: %ldμs, Max: %ldμs", 
             interval, avg_time, min_time, max_time);
    }
    // TODO: for evaluation (end here)
    
    return topk_result;
}

// merge index
extern "C" void
MergeIndex(void *index_ptr, uint8_t *bitmap_ptr, int count, IndexType old_index_type, IndexType new_index_type, void **new_index_ptr, int *new_index_count, int M, int efConstruction, int lists)
{
    elog(DEBUG1, "enter MergeIndex, index_ptr = %p, bitmap_ptr = %p, count = %d, old_index_type = %d, new_index_type = %d", 
        index_ptr, bitmap_ptr, count, old_index_type, new_index_type);
    
    auto *index = static_cast<knowhere::Index<knowhere::IndexNode>*>(index_ptr);

    // Check if the index has raw data
    if (!index->HasRawData(knowhere::metric::L2)) {
        elog(ERROR, "MergeIndex: the index does not have raw data");
        return;
    }

    // Get total number of vectors in the index
    int total_count = index->Count();
    if (total_count != count) {
        elog(ERROR, "MergeIndex: the total number of vectors in the index does not match the count");
        return;
    }
    
    // Generate a bitmap view from the bitmap pointer
    knowhere::BitsetView bitset_view(bitmap_ptr, count);
    int selected_count = count - bitset_view.count();
    if (selected_count <= 0) {
        // TODO: handle this case
        elog(ERROR, "MergeIndex: no vectors are selected");
        return;
    }
    elog(DEBUG1, "MergeIndex: selected_count = %d", selected_count);

    // Create an IDs dataset from the bitmap view
    std::vector<int64_t> selected_ids;
    selected_ids.reserve(selected_count);
    for (int i = 0; i < count; i++) {
        if (!bitset_view.test(i)) {
            selected_ids.push_back(i);
        }
    }
    int64_t actual_selected = (int64_t) selected_ids.size();
    auto ids_dataset = knowhere::GenIdsDataSet(actual_selected, selected_ids.data());

    // retrieve the selected vectors from the index
    auto vectors_result = index->GetVectorByIds(ids_dataset);
    if (!vectors_result.has_value()) {
        elog(ERROR, "MergeIndex: failed to retrieve the selected vectors from the index");
        return;
    }
    auto vectors = vectors_result.value();

    // Create a new index
    // initialize
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    knowhere::Index<knowhere::IndexNode> new_kindex;
    switch (new_index_type)
    {
        case FLAT:
            new_kindex = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version).value();
            break;
        case HNSW:
            new_kindex = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
            break;
        case IVFFLAT:
            new_kindex = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, version).value();
            break;
        default:
            elog(ERROR, "MergeIndex: unsupported new index type %d", new_index_type);
            return;
    }
    knowhere::Index<knowhere::IndexNode> * new_kindex_ptr = new knowhere::Index<knowhere::IndexNode>(new_kindex);

    // configuration
    knowhere::Json conf;
    conf[knowhere::meta::ROWS] = selected_count;
    conf[knowhere::meta::DIM] = index->Dim();
    conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    switch (new_index_type)
    {
        case FLAT:
            break;
        case IVFFLAT:
            conf[knowhere::indexparam::NLIST] = lists;
            break;
        case HNSW:
            conf[knowhere::indexparam::M] = M;
            conf[knowhere::indexparam::EFCONSTRUCTION] = efConstruction;
            break;
    }

    auto build_res = new_kindex_ptr->Build(vectors, conf);
    Assert(build_res == knowhere::Status::success);
    Assert(new_kindex_ptr->Count() == selected_count);

    *new_index_ptr = static_cast<void*>(new_kindex_ptr);
    *new_index_count = selected_count;
}

// Function to merge two built indices by inserting smaller index into larger one
extern "C" void *
MergeTwoIndices(void *lindex_ptr, int lcount, IndexType lindex_type,
                void *sindex_ptr, int scount, IndexType sindex_type,
                int *merged_count)
{
    elog(DEBUG1, "enter MergeTwoIndices, lindex_ptr = %p, lcount = %d, sindex_ptr = %p, scount = %d", 
         lindex_ptr, lcount, sindex_ptr, scount);
    
    auto *lindex = static_cast<knowhere::Index<knowhere::IndexNode>*>(lindex_ptr);
    auto *sindex = static_cast<knowhere::Index<knowhere::IndexNode>*>(sindex_ptr);

    // Verify that index1 is the larger index
    if (lcount < scount) {
        elog(ERROR, "MergeTwoIndices: lindex count (%d) must be >= sindex count (%d)", lcount, scount);
        return NULL;
    }

    // Check if both indices have raw data
    elog(DEBUG1, "MergeTwoIndices: lindex_type = %d, sindex_type = %d", lindex_type, sindex_type);
    if (!lindex->HasRawData(knowhere::metric::L2)) {
        elog(ERROR, "MergeTwoIndices: lindex does not have raw data");
        return NULL;
    }
    if (!sindex->HasRawData(knowhere::metric::L2)) {
        elog(ERROR, "MergeTwoIndices: sindex does not have raw data");
        return NULL;
    }

    // Validate counts match the actual index sizes
    int index1_actual_count = lindex->Count();
    int index2_actual_count = sindex->Count();
    if (index1_actual_count != lcount) {
        elog(ERROR, "MergeTwoIndices: lindex count (%d) does not match lcount (%d)", index1_actual_count, lcount);
        return NULL;
    }
    if (index2_actual_count != scount) {
        elog(ERROR, "MergeTwoIndices: sindex count (%d) does not match scount (%d)", index2_actual_count, scount);
        return NULL;
    }

    // Check dimension consistency
    if (lindex->Dim() != sindex->Dim()) {
        elog(ERROR, "MergeTwoIndices: dimension mismatch - lindex: %d, sindex: %d", lindex->Dim(), sindex->Dim());
        return NULL;
    }
    int dim = lindex->Dim();

    elog(DEBUG1, "MergeTwoIndices: larger_count = %d, smaller_count = %d", lcount, scount);

    // Get all vectors from the smaller index (index2)
    {
        std::vector<int64_t> all_ids(scount);
        for (int64_t i = 0; i < scount; ++i) all_ids[i] = i;
        auto id_ds = knowhere::GenIdsDataSet(scount, all_ids.data());
        auto smaller_vectors_result = sindex->GetVectorByIds(id_ds);
        if (!smaller_vectors_result.has_value()) {
            elog(ERROR, "MergeTwoIndices: failed to retrieve vectors from smaller index");
            return NULL;
        }
        auto smaller_dataset = smaller_vectors_result.value();

        // Insert vectors from smaller index into larger index
        knowhere::Json add_conf;
        add_conf[knowhere::meta::ROWS] = scount;
        add_conf[knowhere::meta::DIM] = dim;
        add_conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;

        auto add_res = lindex->Add(smaller_dataset, add_conf);

        if (add_res != knowhere::Status::success) {
            elog(ERROR, "MergeTwoIndices: failed to add vectors from smaller index to larger index");
            return NULL;
        }
    }

    // Verify the merged index count
    int merged_count_check = lindex->Count();
    int expected_count = lcount + scount;
    if (merged_count_check != expected_count) {
        elog(ERROR, "MergeTwoIndices: merged index count (%d) does not match expected (%d)", 
             merged_count_check, expected_count);
        return NULL;
    }

    elog(DEBUG1, "MergeTwoIndices: merged_count = %d", merged_count_check);

    // Return the merged count
    *merged_count = merged_count_check;
    return lindex;
}