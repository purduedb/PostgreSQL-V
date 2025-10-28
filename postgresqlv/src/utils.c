#include "postgres.h"
#include "utils.h"

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

void
free_topk_vector(topKVector *tkv)
{
	if (tkv != NULL)
	{
		if (tkv->distances != NULL)
			pfree(tkv->distances);
		if (tkv->vids != NULL)
			pfree(tkv->vids);
		pfree(tkv);
	}
}

/*
 * Allocate a vector array
 */
VectorArray
VectorArrayInit(int maxlen, int dimensions, Size itemsize)
{
	VectorArray res = (VectorArray) palloc(sizeof(VectorArrayData));

	/* Ensure items are aligned to prevent UB */
	itemsize = MAXALIGN(itemsize);

	res->length = 0;
	res->maxlen = maxlen;
	res->dim = dimensions;
	res->itemsize = itemsize;
	res->items = (char*) palloc_extended(maxlen * itemsize, MCXT_ALLOC_ZERO | MCXT_ALLOC_HUGE);
	return res;
}

/*
 * Free a vector array
 */
void
VectorArrayFree(VectorArray arr)
{
	pfree(arr->items);
	pfree(arr);
}

// dms segment helper functions
void*
get_pointer_from_cached_segment(dsm_handle handle)
{
    dsm_segment *seg = dsm_find_mapping(handle);
    if (seg == NULL)
    {
        seg = dsm_attach(handle);
        dsm_pin_mapping(seg);
    }
    return dsm_segment_address(seg); 
}