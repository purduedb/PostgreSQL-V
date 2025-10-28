#ifndef UTILS_H
#define UTILS_H

#include "postgres.h"
#include "storage/itemptr.h"
#include "storage/dsm.h"

int64_t ItemPointerToInt64(const ItemPointer tid);
ItemPointerData Int64ToItemPointer(int64_t encoded);

// for searching
typedef struct DistancePair{
    float distance;
    int64_t id;
} DistancePair;

typedef struct TopKTuples {
    int num_results;
    DistancePair * pairs;
} TopKTuples;

// define a structure to store the result of searches
typedef struct topKVector {
    int num_results;
    float * distances;
    int64_t * vids;
} topKVector;

void free_topk_vector(topKVector *tkv);

// for index building
#define INDEX_BUILD_BATCH 1000000
#define DEFAULT_TIDS_SIZE 1000000

typedef struct VectorArrayData
{
	int			length;
	int			maxlen;
	int			dim;
	Size		itemsize;
	char	   *items;
}			VectorArrayData;

typedef VectorArrayData * VectorArray;

#define VECTOR_ARRAY_SIZE(_length, _size) (sizeof(VectorArrayData) + (_length) * MAXALIGN(_size))

/* Use functions instead of macros to avoid double evaluation */

static inline Pointer
VectorArrayGet(VectorArray arr, int offset)
{
	return ((char *) arr->items) + (offset * arr->itemsize);
}

static inline void
VectorArraySet(VectorArray arr, int offset, Pointer val)
{
	memcpy(VectorArrayGet(arr, offset), val, VARSIZE_ANY(val));
}

// Allocate a vector array
VectorArray VectorArrayInit(int maxlen, int dimensions, Size itemsize);
//Free a vector array
void VectorArrayFree(VectorArray arr);

// dms segment helper functions
void* get_pointer_from_cached_segment(dsm_handle handle);

#endif