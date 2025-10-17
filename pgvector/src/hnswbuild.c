/*
 * The HNSW build happens in two phases:
 *
 * 1. In-memory phase
 *
 * In this first phase, the graph is held completely in memory. When the graph
 * is fully built, or we run out of memory reserved for the build (determined
 * by maintenance_work_mem), we materialize the graph to disk (see
 * FlushPages()), and switch to the on-disk phase.
 *
 * In a parallel build, a large contiguous chunk of shared memory is allocated
 * to hold the graph. Each worker process has its own HnswBuildState struct in
 * private memory, which contains information that doesn't change throughout
 * the build, and pointers to the shared structs in shared memory. The shared
 * memory area is mapped to a different address in each worker process, and
 * 'HnswBuildState.hnswarea' points to the beginning of the shared area in the
 * worker process's address space. All pointers used in the graph are
 * "relative pointers", stored as an offset from 'hnswarea'.
 *
 * Each element is protected by an LWLock. It must be held when reading or
 * modifying the element's neighbors or 'heaptids'.
 *
 * In a non-parallel build, the graph is held in backend-private memory. All
 * the elements are allocated in a dedicated memory context, 'graphCtx', and
 * the pointers used in the graph are regular pointers.
 *
 * 2. On-disk phase
 *
 * In the on-disk phase, the index is built by inserting each vector to the
 * index one by one, just like on INSERT. The only difference is that we don't
 * WAL-log the individual inserts. If the graph fit completely in memory and
 * was fully built in the in-memory phase, the on-disk phase is skipped.
 *
 * After we have finished building the graph, we perform one more scan through
 * the index and write all the pages to the WAL.
 */
#include "postgres.h"

#include <math.h>

#include "access/parallel.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/xact.h"
#include "access/xloginsert.h"
#include "catalog/index.h"
#include "catalog/pg_type_d.h"
#include "commands/progress.h"
#include "hnsw.h"
#include "miscadmin.h"
#include "optimizer/optimizer.h"
#include "storage/bufmgr.h"
#include "tcop/tcopprot.h"
#include "utils/datum.h"
#include "utils/memutils.h"
#include "vectorindeximpl.hpp"
#include "lsmindex.h"

#if PG_VERSION_NUM >= 140000
#include "utils/backend_progress.h"
#else
#include "pgstat.h"
#endif

#if PG_VERSION_NUM >= 140000
#include "utils/backend_status.h"
#include "utils/wait_event.h"
#endif

#define PARALLEL_KEY_HNSW_SHARED		UINT64CONST(0xA000000000000001)
#define PARALLEL_KEY_HNSW_AREA			UINT64CONST(0xA000000000000002)
#define PARALLEL_KEY_QUERY_TEXT			UINT64CONST(0xA000000000000003)

/*
 * Create the metapage
 */
static void
CreateMetaPage(HnswBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;

	buf = HnswNewBuffer(index, forkNum);
	page = BufferGetPage(buf);
	HnswInitPage(buf, page);

	/* Set metapage data */
	metap = HnswPageGetMeta(page);
	metap->magicNumber = HNSW_MAGIC_NUMBER;
	metap->version = HNSW_VERSION;
	metap->dimensions = buildstate->dimensions;
	metap->m = buildstate->m;
	metap->efConstruction = buildstate->efConstruction;
	metap->entryBlkno = InvalidBlockNumber;
	metap->entryOffno = InvalidOffsetNumber;
	metap->entryLevel = -1;
	metap->insertPage = InvalidBlockNumber;
	((PageHeader) page)->pd_lower =
		((char *) metap + sizeof(HnswMetaPageData)) - (char *) page;

	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
}

/*
 * Initialize the build state
 */
static void
InitBuildState(HnswBuildState * buildstate, Relation heap, Relation index, IndexInfo *indexInfo, ForkNumber forkNum)
{
	buildstate->heap = heap;
	buildstate->index = index;
	buildstate->indexInfo = indexInfo;
	buildstate->forkNum = forkNum;
	buildstate->typeInfo = HnswGetTypeInfo(index);

	buildstate->m = HnswGetM(index);
	buildstate->efConstruction = HnswGetEfConstruction(index);
	buildstate->dimensions = TupleDescAttr(index->rd_att, 0)->atttypmod;

	/* Disallow varbit since require fixed dimensions */
	if (TupleDescAttr(index->rd_att, 0)->atttypid == VARBITOID)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("type not supported for hnsw index")));

	/* Require column to have dimensions to be indexed */
	if (buildstate->dimensions < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("column does not have dimensions")));

	if (buildstate->dimensions > buildstate->typeInfo->maxDimensions)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("column cannot have more than %d dimensions for hnsw index", buildstate->typeInfo->maxDimensions)));

	if (buildstate->efConstruction < 2 * buildstate->m)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ef_construction must be greater than or equal to 2 * m")));

	buildstate->reltuples = 0;
	buildstate->indtuples = 0;

	/* Get support functions */
	HnswInitSupport(&buildstate->support, index);


	buildstate->graphCtx = GenerationContextCreate(CurrentMemoryContext,
												   "Hnsw build graph context",
#if PG_VERSION_NUM >= 150000
												   1024 * 1024, 1024 * 1024,
#endif
												   1024 * 1024);
	buildstate->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											   "Hnsw build temporary context",
											   ALLOCSET_DEFAULT_SIZES);

	// TODO: avoid hardcoding vector size
	buildstate->vectors = VectorArrayInit(INDEX_BUILD_BATCH, buildstate->dimensions, VECTOR_SIZE(buildstate->dimensions));

	buildstate->tids = (int64_t *) palloc(sizeof(int64_t) * DEFAULT_TIDS_SIZE);
	buildstate->num_tids = 0;
	buildstate->cap_tids = DEFAULT_TIDS_SIZE;

	/* Create visibility tuple description */
	buildstate->vitupdesc = CreateTemplateTupleDesc(1);
	TupleDescInitEntry(buildstate->vitupdesc, (AttrNumber) 1, "vid", INT8OID, -1, 0);
}

/*
 * Free resources
 */
static void
FreeBuildState(HnswBuildState * buildstate)
{
	VectorArrayFree(buildstate->vectors);
	pfree(buildstate->tids);
	IndexFree(buildstate->hnswIndex);
	MemoryContextDelete(buildstate->graphCtx);
	MemoryContextDelete(buildstate->tmpCtx);
}

static inline void
AppendTid(HnswBuildState *buildstate, int64_t tid)
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
AddVector(Datum *values, bool *isnull, ItemPointer tid, HnswBuildState * buildstate)
{
	const		HnswTypeInfo *typeInfo = buildstate->typeInfo;
	HnswSupport *support = &buildstate->support;
	VectorArray vectors = buildstate->vectors;
	int			targvectors = vectors->maxlen;
	Datum		value;

	if (!HnswFormIndexValue(&value, values, isnull, buildstate->typeInfo, support))
		return;

	if (vectors->length < targvectors)
	{
		VectorArraySet(vectors, vectors->length, DatumGetPointer(value));
		int64_t tid_int = ItemPointerToInt64(tid);
		AppendTid(buildstate, tid_int);
		vectors->length++;
	}
	else
	{
		elog(ERROR, "[AddVector] An error occurs when building the HNSW index");
	}

	// batch add vectors to the IVFFlat index
	if (vectors->length == targvectors)
	{
		HnswIndexcreate(buildstate->hnswIndex, buildstate->m, buildstate->efConstruction, buildstate->vectors);
		vectors->length = 0;
	}
}

/*
 * Callback for sampling
 */
static void
BuildCallback(Relation index, ItemPointer tid, Datum *values,
			   bool *isnull, bool tupleIsAlive, void *state)
{
	HnswBuildState *buildstate = (HnswBuildState *) state;
	MemoryContext oldCtx;

	/* Skip nulls */
	if (isnull[0])
		return;

	/* Use memory context since detoast can allocate */
	oldCtx = MemoryContextSwitchTo(buildstate->tmpCtx);

	/* Add sample */
	AddVector(values, isnull, tid, buildstate);

	/* Reset memory context */
	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(buildstate->tmpCtx);
}

/* MY FUNCTION
 * Scan all rows
 */
static void
ScanAllRows(HnswBuildState *buildstate)
{
	elog(DEBUG1, "enter ScanAllRows");

    BlockNumber totalblocks = RelationGetNumberOfBlocks(buildstate->heap);
	elog(DEBUG1, "[ScanAllRows] totalblocks = %d", totalblocks);

    buildstate->reltuples = table_index_build_scan(buildstate->heap, buildstate->index, buildstate->indexInfo,
		true, true, BuildCallback, (void *) buildstate, NULL);
	
	// add the remaining vectors
	if (buildstate->vectors->length > 0)
	{
		HnswIndexcreate(buildstate->hnswIndex, buildstate->m, buildstate->efConstruction, buildstate->vectors);
		buildstate->vectors->length = 0;
	}
}

/*
 * Build the index
 */
static void
BuildIndex(Relation heap, Relation index, IndexInfo *indexInfo,
		   HnswBuildState * buildstate, ForkNumber forkNum)
{
#ifdef HNSW_MEMORY
	SeedRandom(42);
#endif
	elog(DEBUG1, "enter BuildIndex");
	InitBuildState(buildstate, heap, index, indexInfo, forkNum);

	MemoryContext hnswBuildCtx = AllocSetContextCreate(CurrentMemoryContext,
        "hnsw build temporary context",
        ALLOCSET_DEFAULT_SIZES);
    MemoryContext oldCtx = MemoryContextSwitchTo(hnswBuildCtx);

	uint32_t dim = buildstate->dimensions;

	HnswIndexInit(dim, buildstate->m, buildstate->efConstruction, &(buildstate->hnswIndex));
	CreateMetaPage(buildstate);
	ScanAllRows(buildstate);

	Oid relId = RelationGetRelid(buildstate->index);
	void *tids = buildstate->tids;
	uint32_t elem_size = buildstate->vectors->itemsize / buildstate->dimensions;
	
	build_lsm_index(HNSW, relId, buildstate->hnswIndex, (int64_t)tids, dim, elem_size, buildstate->num_tids);
	
	// TODO: write the status pages

	MemoryContextSwitchTo(oldCtx);
    MemoryContextDelete(hnswBuildCtx);

	// FIXME: ???
	// if (RelationNeedsWAL(index) || forkNum == INIT_FORKNUM)
	// 	log_newpage_range(index, forkNum, 0, RelationGetNumberOfBlocksInFork(index, forkNum), true);

	FreeBuildState(buildstate);
}

/*
 * Build the index for a logged table
 */
IndexBuildResult *
hnswbuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result;
	HnswBuildState buildstate;

	BuildIndex(heap, index, indexInfo, &buildstate, MAIN_FORKNUM);

	result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
	result->heap_tuples = buildstate.reltuples;
	result->index_tuples = buildstate.indtuples;

	return result;
}

/*
 * Build the index for an unlogged table
 */
void
hnswbuildempty(Relation index)
{
	IndexInfo  *indexInfo = BuildIndexInfo(index);
	HnswBuildState buildstate;

	BuildIndex(NULL, index, indexInfo, &buildstate, INIT_FORKNUM);
}
