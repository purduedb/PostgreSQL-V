#include "postgres.h"

#include "access/relscan.h"
#include "hnsw.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/float.h"
#include "utils/memutils.h"

#include "lsmindex.h"

/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
		value = PointerGetDatum(NULL);
	else
	{
		value = scan->orderByData->sk_argument;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->support.normprocinfo != NULL)
			value = HnswNormValue(so->typeInfo, so->support.collation, value);
	}

	return value;
}

/*
 * Prepare for an index scan
 */
IndexScanDesc
hnswbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaque so;
	double		maxMemory;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	so = (HnswScanOpaque) palloc(sizeof(HnswScanOpaqueData));
	so->typeInfo = HnswGetTypeInfo(index);

	/* Set support functions */
	HnswInitSupport(&so->support, index);

	/*
	 * Use a lower max allocation size than default to allow scanning more
	 * tuples for iterative search before exceeding work_mem
	 */
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Hnsw scan temporary context",
									   0, 8 * 1024, 256 * 1024);

	/* Calculate max memory */
	/* Add 256 extra bytes to fill last block when close */
	maxMemory = (double) work_mem * hnsw_scan_mem_multiplier * 1024.0 + 256;
	so->maxMemory = Min(maxMemory, (double) SIZE_MAX);

	scan->opaque = so;

	return scan;
}

/*
 * Start or restart an index scan
 */
void
hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	
	// if (!so->first && so->topkTuples.pairs != NULL)
	// {
	// 	elog(DEBUG1, "[hnswrescan] checkpoint 01, so->first? %d, so->topkTuples.pairs != NULL? %d", so->first, so->topkTuples.pairs != NULL);
	// 	pfree(so->topkTuples.pairs);
	// 	elog(DEBUG1, "[hnswrescan] checkpoint 1");
	// }
	// elog(DEBUG1, "[hnswrescan] checkpoint 2");

	so->first = true;
	MemoryContextReset(so->tmpCtx);

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
 * Fetch the next tuple in the given scan
 */
bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan hnsw index without order");

		/* Requires MVCC-compliant snapshot as not able to maintain a pin */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with hnsw");

		/* Get scan value */
		value = GetScanValue(scan);

		// /*
		//  * Get a shared lock. This allows vacuum to ensure no in-flight scans
		//  * before marking tuples as deleted.
		//  */
		// LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		// so->w = GetScanItems(scan, value);

		// /* Release shared lock */
		// UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		// conduct hnsw search
		Vector *query_vector = (Vector *) PointerGetDatum(value);
		// FIXME: how are we going to set top_k?
		int top_k = 100;
		so->topkTuples = search_lsm_index(scan->indexRelation, query_vector->x, top_k, hnsw_ef_search);
		so->topkTuplesIdx = 0;	
		so->first = false;
	}

	// return the next tuple from the results of hnsw search
	if (so->topkTuplesIdx < so->topkTuples.num_results)
	{
		ItemPointerData heaptid_data = Int64ToItemPointer(so->topkTuples.pairs[so->topkTuplesIdx++].id);

		MemoryContextSwitchTo(oldCtx);
		scan->xs_heaptid = heaptid_data;
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
		return true;
	}

	// TODO: do iterative search if necessary and enable
	MemoryContextSwitchTo(oldCtx);
	return false;
}

/*
 * End a scan and release resources
 */
void
hnswendscan(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	MemoryContextDelete(so->tmpCtx);

	// pfree(so->topkTuples.pairs); // already freed by MemoryContextDelete

	// Free the scan opaque structure
	if (so != NULL)
	{
		pfree(so);
	}
	scan->opaque = NULL;
}
