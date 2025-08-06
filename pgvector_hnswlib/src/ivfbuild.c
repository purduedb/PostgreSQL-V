#include "postgres.h"

#include <float.h>

#include "access/table.h"
#include "access/tableam.h"
#include "access/parallel.h"
#include "access/xact.h"
#include "bitvec.h"
#include "catalog/index.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type_d.h"
#include "commands/progress.h"
#include "halfvec.h"
#include "ivfflat.h"
#include "miscadmin.h"
#include "optimizer/optimizer.h"
#include "storage/bufmgr.h"
#include "tcop/tcopprot.h"
#include "utils/memutils.h"
#include "vector.h"
#include "visibility.h"

#if PG_VERSION_NUM >= 140000
#include "utils/backend_progress.h"
#else
#include "pgstat.h"
#endif

#if PG_VERSION_NUM >= 130000
#define CALLBACK_ITEM_POINTER ItemPointer tid
#else
#define CALLBACK_ITEM_POINTER HeapTuple hup
#endif

#if PG_VERSION_NUM >= 140000
#include "utils/backend_status.h"
#include "utils/wait_event.h"
#endif

#define PARALLEL_KEY_IVFFLAT_SHARED		UINT64CONST(0xA000000000000001)
#define PARALLEL_KEY_TUPLESORT			UINT64CONST(0xA000000000000002)
#define PARALLEL_KEY_IVFFLAT_CENTERS	UINT64CONST(0xA000000000000003)
#define PARALLEL_KEY_QUERY_TEXT			UINT64CONST(0xA000000000000004)

#include "faiss_index.hpp"
#include "lsm_index_struct.h"

/*
 * Add sample
 */
static void
AddSample(Datum *values, IvfflatBuildState * buildstate)
{
	VectorArray samples = buildstate->samples;
	int			targsamples = samples->maxlen;

	/* Detoast once for all calls */
	Datum		value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));

	/*
	 * Normalize with KMEANS_NORM_PROC since spherical distance function
	 * expects unit vectors
	 */
	// if (buildstate->kmeansnormprocinfo != NULL)
	// {
	// 	if (!IvfflatCheckNorm(buildstate->kmeansnormprocinfo, buildstate->collation, value))
	// 		return;

	// 	value = IvfflatNormValue(buildstate->typeInfo, buildstate->collation, value);
	// }

	if (samples->length < targsamples)
	{
		VectorArraySet(samples, samples->length, DatumGetPointer(value));
		samples->length++;
	}
	else
	{
		if (buildstate->rowstoskip < 0)
			buildstate->rowstoskip = reservoir_get_next_S(&buildstate->rstate, samples->length, targsamples);

		if (buildstate->rowstoskip <= 0)
		{
#if PG_VERSION_NUM >= 150000
			int			k = (int) (targsamples * sampler_random_fract(&buildstate->rstate.randstate));
#else
			int			k = (int) (targsamples * sampler_random_fract(buildstate->rstate.randstate));
#endif

			Assert(k >= 0 && k < targsamples);
			VectorArraySet(samples, k, DatumGetPointer(value));
		}

		buildstate->rowstoskip -= 1;
	}
}

static inline void
AppendTid(IvfflatBuildState *buildstate, int64_t tid)
{
    if (buildstate->num_tids >= buildstate->cap_tids)
    {
        int newcap = buildstate->cap_tids * 2;
        buildstate->tids = (int64_t *) repalloc(buildstate->tids, sizeof(int64_t) * newcap);
        buildstate->cap_tids = newcap;
    }

    buildstate->tids[buildstate->num_tids++] = tid;
}


/*
 * Add vector
 */
static void
Our_AddVector(Datum *values, ItemPointer tid, IvfflatBuildState * buildstate)
{
	VectorArray vectors = buildstate->vectors;
	int			targvectors = vectors->maxlen;

	/* Detoast once for all calls */
	Datum		value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));
	
	/* Normalize if needed */
	if (buildstate->normprocinfo != NULL)
	{
		if (!IvfflatCheckNorm(buildstate->normprocinfo, buildstate->collation, value))
			return;

		value = IvfflatNormValue(buildstate->typeInfo, buildstate->collation, value);
	}

	if (vectors->length < targvectors)
	{
		VectorArraySet(vectors, vectors->length, DatumGetPointer(value));
		int64_t tid_int = ItemPointerToInt64(tid);
		AppendTid(buildstate, tid_int);
		vectors->length++;
	}
	else
	{
		elog(ERROR, "[Our_AddVector] An error occurs when building the IVFFlat index");
	}

	// batch add vectors to the IVFFlat index
	if (vectors->length == targvectors)
	{
		FaissIvfflatIndexCreate(buildstate->faissIndex, buildstate->lowest_vid, vectors);
		buildstate->lowest_vid += vectors->length;
		vectors->length = 0;
	}
}

/*
 * Callback for sampling
 */
static void
Our_BuildCallback(Relation index, CALLBACK_ITEM_POINTER, Datum *values,
			   bool *isnull, bool tupleIsAlive, void *state)
{
	IvfflatBuildState *buildstate = (IvfflatBuildState *) state;
	MemoryContext oldCtx;

	/* Skip nulls */
	if (isnull[0])
		return;

	/* Use memory context since detoast can allocate */
	oldCtx = MemoryContextSwitchTo(buildstate->tmpCtx);

	/* Add sample */
	Our_AddVector(values, tid, buildstate);

	/* Reset memory context */
	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(buildstate->tmpCtx);
}

/* MY FUNCTION
 * Scan all rows
 */
static void
ScanAllRows(IvfflatBuildState *buildstate)
{
	elog(DEBUG1, "enter ScanAllRows");

    BlockNumber totalblocks = RelationGetNumberOfBlocks(buildstate->heap);

    buildstate->rowstoskip = -1;

    buildstate->reltuples = table_index_build_scan(buildstate->heap, buildstate->index, buildstate->indexInfo,
		true, true, Our_BuildCallback, (void *) buildstate, NULL);
	
	// FIXME: code refactoring
	// add the remaining vectors
	elog(DEBUG1, "check whether there are remaining vectors");
	if (buildstate->vectors->length > 0)
	{
		FaissIvfflatIndexCreate(buildstate->faissIndex, buildstate->lowest_vid, buildstate->vectors);
		buildstate->lowest_vid += buildstate->vectors->length;
		elog(DEBUG1, "[ScanAllRows] return from FaissIvfflatIndexCreate, updated lowest_vid = %d", buildstate->lowest_vid);
		buildstate->vectors->length = 0;
	}
}


/*
 * Initialize the build state
 */
static void
OurInitBuildState(IvfflatBuildState * buildstate, Relation heap, Relation index, IndexInfo *indexInfo)
{
	buildstate->heap = heap;
	buildstate->index = index;
	buildstate->indexInfo = indexInfo;
	buildstate->typeInfo = IvfflatGetTypeInfo(index);

	buildstate->lists = IvfflatGetLists(index);
	buildstate->dimensions = TupleDescAttr(index->rd_att, 0)->atttypmod;

	/* Disallow varbit since require fixed dimensions */
	if (TupleDescAttr(index->rd_att, 0)->atttypid == VARBITOID)
		elog(ERROR, "type not supported for ivfflat index");

	/* Require column to have dimensions to be indexed */
	if (buildstate->dimensions < 0)
		elog(ERROR, "column does not have dimensions");

	if (buildstate->dimensions > buildstate->typeInfo->maxDimensions)
		elog(ERROR, "column cannot have more than %d dimensions for ivfflat index", buildstate->typeInfo->maxDimensions);

	buildstate->reltuples = 0;
	buildstate->indtuples = 0;

	/* Get support functions */
	buildstate->procinfo = index_getprocinfo(index, 1, IVFFLAT_DISTANCE_PROC);
	buildstate->normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);
	buildstate->kmeansnormprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_KMEANS_NORM_PROC);
	buildstate->collation = index->rd_indcollation[0];

	/* Require more than one dimension for spherical k-means */
	if (buildstate->kmeansnormprocinfo != NULL && buildstate->dimensions == 1)
		elog(ERROR, "dimensions must be greater than one for this opclass");

	/* Create tuple description for sorting */
	buildstate->tupdesc = CreateTemplateTupleDesc(3);
	TupleDescInitEntry(buildstate->tupdesc, (AttrNumber) 1, "list", INT4OID, -1, 0);
	TupleDescInitEntry(buildstate->tupdesc, (AttrNumber) 2, "tid", TIDOID, -1, 0);
	TupleDescInitEntry(buildstate->tupdesc, (AttrNumber) 3, "vector", TupleDescAttr(buildstate->tupdesc, 0)->atttypid, -1, 0);

	buildstate->slot = MakeSingleTupleTableSlot(buildstate->tupdesc, &TTSOpsVirtual);

	buildstate->centers = VectorArrayInit(buildstate->lists, buildstate->dimensions, buildstate->typeInfo->itemSize(buildstate->dimensions));
	buildstate->listInfo = palloc(sizeof(ListInfo) * buildstate->lists);

	buildstate->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											   "Ivfflat build temporary context",
											   ALLOCSET_DEFAULT_SIZES);
						
	// Add for faiss's ivfflat index build phase
	buildstate->vectors = VectorArrayInit(INDEX_BUILD_BATCH, buildstate->dimensions, buildstate->typeInfo->itemSize(buildstate->dimensions));

	buildstate->lowest_vid = 0;
	buildstate->tids = (int64_t *) palloc(sizeof(int64_t) * DEFAULT_TIDS_SIZE);
	buildstate->num_tids = 0;
	buildstate->cap_tids = DEFAULT_TIDS_SIZE;
#ifdef IVFFLAT_KMEANS_DEBUG
	buildstate->inertia = 0;
	buildstate->listSums = palloc0(sizeof(double) * buildstate->lists);
	buildstate->listCounts = palloc0(sizeof(int) * buildstate->lists);
#endif

	buildstate->ivfleader = NULL;

	/* Create visibility tuple description */
	buildstate->vitupdesc = CreateTemplateTupleDesc(1);
	TupleDescInitEntry(buildstate->vitupdesc, (AttrNumber) 1, "vid", INT8OID, -1, 0);
}

/*
 * Free resources
 */
static void
Our_FreeBuildState(IvfflatBuildState * buildstate)
{
	VectorArrayFree(buildstate->centers);
	pfree(buildstate->listInfo);
	VectorArrayFree(buildstate->vectors);
	pfree(buildstate->tids);
	FaissIndexFree(buildstate->faissIndex);

#ifdef IVFFLAT_KMEANS_DEBUG
	pfree(buildstate->listSums);
	pfree(buildstate->listCounts);
#endif

	MemoryContextDelete(buildstate->tmpCtx);
}

/*
 * Callback for sampling
 */
static void
SampleCallback(Relation index, ItemPointer tid, Datum *values,
			   bool *isnull, bool tupleIsAlive, void *state)
{
	IvfflatBuildState *buildstate = (IvfflatBuildState *) state;
	MemoryContext oldCtx;

	/* Skip nulls */
	if (isnull[0])
		return;

	/* Use memory context since detoast can allocate */
	oldCtx = MemoryContextSwitchTo(buildstate->tmpCtx);

	/* Add sample */
	AddSample(values, buildstate);

	/* Reset memory context */
	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(buildstate->tmpCtx);
}

/*
 * Sample rows with same logic as ANALYZE
 */
static void
SampleRows(IvfflatBuildState * buildstate)
{
	elog(DEBUG1, "enter SampleRows");

	int			targsamples = buildstate->samples->maxlen;
	BlockNumber totalblocks = RelationGetNumberOfBlocks(buildstate->heap);

	buildstate->rowstoskip = -1;

	BlockSampler_Init(&buildstate->bs, totalblocks, targsamples, RandomInt());

	reservoir_init_selection_state(&buildstate->rstate, targsamples);
	while (BlockSampler_HasMore(&buildstate->bs))
	{
		BlockNumber targblock = BlockSampler_Next(&buildstate->bs);

		table_index_build_range_scan(buildstate->heap, buildstate->index, buildstate->indexInfo,
									 false, true, false, targblock, 1, SampleCallback, (void *) buildstate, NULL);
	}
}

/*
 * Compute centers
 */
static void
OurComputeCenters(IvfflatBuildState * buildstate)
{
	elog(DEBUG1, "enter OurComputeCenters");

	int			numSamples;

	pgstat_progress_update_param(PROGRESS_CREATEIDX_SUBPHASE, PROGRESS_IVFFLAT_PHASE_KMEANS);

	/* Target 50 samples per list, with at least 10000 samples */
	/* The number of samples has a large effect on index build time */
	numSamples = buildstate->lists * 50;
	if (numSamples < 10000)
		numSamples = 10000;

	/* Skip samples for unlogged table */
	if (buildstate->heap == NULL)
		numSamples = 1;

	elog(DEBUG1, "[OurComputeCenters] numSamples = %d", numSamples);

	/* Sample rows */
	buildstate->samples = VectorArrayInit(numSamples, buildstate->dimensions, buildstate->centers->itemsize);
	
	if (buildstate->heap != NULL)
	{
		SampleRows(buildstate);
		elog(DEBUG1, "return from SampleRows");

		if (buildstate->samples->length < buildstate->lists)
		{
			ereport(NOTICE,
					(errmsg("ivfflat index created with little data"),
					 errdetail("This will cause low recall."),
					 errhint("Drop the index until the table has more data.")));
		}
	}

	Assert(buildstate->lists == buildstate->centers->maxlen);
	IvfflatBench("k-means", FaissIvfflatTrain(buildstate->index, buildstate->samples, buildstate->centers, &buildstate->faissIndex));
	elog(DEBUG1, "return from FaissIvfflatTrain");
	VectorArrayFree(buildstate->samples);
}

/*
 * Create the metapage
 */
static void
CreateMetaPage(Relation index, int dimensions, int lists, ForkNumber forkNum)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	IvfflatMetaPage metap;

	buf = IvfflatNewBuffer(index, forkNum);
	IvfflatInitRegisterPage(index, &buf, &page, &state);

	/* Set metapage data */
	metap = IvfflatPageGetMeta(page);
	metap->magicNumber = IVFFLAT_MAGIC_NUMBER;
	metap->version = IVFFLAT_VERSION;
	metap->dimensions = dimensions;
	metap->lists = lists;
	((PageHeader) page)->pd_lower =
		((char *) metap + sizeof(IvfflatMetaPageData)) - (char *) page;

	IvfflatCommitBuffer(buf, state);
}

/*
 * Build the index
 */
static void
OurBuildIndex(Relation heap, Relation index, IndexInfo *indexInfo,
		   IvfflatBuildState * buildstate, ForkNumber forkNum)
{
	elog(DEBUG1, "enter OurBuildIndex");

	OurInitBuildState(buildstate, heap, index, indexInfo);

	MemoryContext ivfflatBuildCtx = AllocSetContextCreate(CurrentMemoryContext,
        "Ivfflat build temporary context",
        ALLOCSET_DEFAULT_SIZES);
    MemoryContext oldCtx = MemoryContextSwitchTo(ivfflatBuildCtx);

	OurComputeCenters(buildstate);

	// for the estimation function
	CreateMetaPage(index, buildstate->dimensions, buildstate->lists, forkNum);

	ScanAllRows(buildstate);

	Oid relId = RelationGetRelid(buildstate->index);
	void *tids = buildstate->tids;
	uint32_t dim = buildstate->dimensions;
	uint32_t elem_size = buildstate->vectors->itemsize / buildstate->dimensions;
	VectorId lowest_vid = 0;
	VectorId highest_vid = buildstate->num_tids - 1;

	int status = build_lsm_index(IVFFLAT, relId, buildstate->faissIndex, tids, dim,
		elem_size, lowest_vid, highest_vid);

	// write visibility tuples
	CreateVisibilityMetaPage(index, forkNum);
	IvfflatBench("write visibility files", InsertVisibilityTuples(forkNum, buildstate->index, buildstate->num_tids, buildstate->vitupdesc, buildstate->tids));

	MemoryContextSwitchTo(oldCtx);
    MemoryContextDelete(ivfflatBuildCtx);

	Our_FreeBuildState(buildstate);
}

/*
 * Build the index for a logged table
 */
IndexBuildResult *
our_ivfflatbuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	elog(DEBUG1, "call our_ivfflatbuild");

	IndexBuildResult *result;
	IvfflatBuildState buildstate;

	OurBuildIndex(heap, index, indexInfo, &buildstate, MAIN_FORKNUM);

	elog(DEBUG1, "return from OurBuildIndex and begin writing result");

	result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
	// TODO: set `buildstate.reltuples` and `buildstate.indtuples`
	result->heap_tuples = buildstate.reltuples;
	result->index_tuples = buildstate.indtuples;

	return result;
}

/*
 * Build the index for an unlogged table
 */
void
our_ivfflatbuildempty(Relation index)
{
	IndexInfo  *indexInfo = BuildIndexInfo(index);
	IvfflatBuildState buildstate;

	OurBuildIndex(NULL, index, indexInfo, &buildstate, INIT_FORKNUM);
}
