#ifndef VISIBILITY_H
#define VISIBILITY_H

#include "postgres.h"
#include "access/genam.h"
#include "storage/bufpage.h"
#include "storage/smgr.h"
#include "storage/buf_internals.h"

typedef struct VisibilityMetaPageData
{
    BlockNumber startPage;
    BlockNumber insertPage;
} VisibilityMetaPageData;

#define VISIBILITY_HEAD_BLKNO 1

typedef VisibilityMetaPageData *VisibilityMetaPage;

#define VisibilityPageGetMeta(page)	((VisibilityMetaPage) PageGetContents(page))

void VisibilityUpdateMetadata(Relation index, BlockNumber insertPage, BlockNumber originalInsertPage, BlockNumber startPage, ForkNumber forkNum);

void GetVisibilityInsertPage(Relation index, BlockNumber *insertPage);

void GetVisibilityStartPage(Relation index, BlockNumber *startPage);

void CreateVisibilityMetaPage(Relation index, ForkNumber forkNum);

void InsertVisibilityTuples(ForkNumber forkNum, Relation index, int num_tids, TupleDesc tupdesc, int64_t *tids);

void InsertVisibilityTuple(Relation index, uint64_t vid, ItemPointer heap_tid, Relation heapRel);
#endif