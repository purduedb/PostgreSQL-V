#ifndef STATUSPAGE_H
#define STATUSPAGE_H

#include "postgres.h"
#include "access/genam.h"
#include "storage/bufpage.h"
#include "storage/smgr.h"
#include "storage/buf_internals.h"
#include "lsmindex.h"

#define STATUS_PAGE_ID	0xFF91
// IVFFLAT_METAPAGE_BLKNO & HNSW_METAPAGE_BLKNO = 0
#define STATUS_METAPAGE_BLKNO 1
#define STATUS_MEMTABLE_ARRAY_BLKNO 2

typedef struct StatusPageMetaData
{
    BlockNumber freePageHead;
    BlockNumber freePageTail;
    // BlockNumber MemtableArray;
} StatusPageMetaData;
typedef StatusPageMetaData* StatusPageMeta;

typedef struct StatusMemtableData
{
    BlockNumber memtablePageHead;
    // BlockNumber MemtablePageTail;
    BlockNumber memtableInsertPage;
    SegmentId sid;
} StatusMemtableData;
typedef StatusMemtableData* StatusMemtable;

typedef struct StatusPageOpaqueData
{
    BlockNumber nextblkno;
    // TODO: not used yet
    uint16_t unused;
    uint16_t page_id;
} StatusPageOpaqueData;

typedef struct MemtableInfo
{
    BlockNumber blkno;
    OffsetNumber offno;
} MemtableInfo;

typedef struct StatusTupleData
{
    ItemPointerData t_tid;
} StatusTupleData;
typedef StatusTupleData* StatusTuple;

typedef StatusPageOpaqueData* StatusPageOpaque;

#define StatusPageGetOpaque(page) ((StatusPageOpaque) PageGetSpecialPointer(page))
#define StatusPageGetMeta(page) ((StatusPageMeta) PageGetContents(page))

void CreateStatusMetaPage(Relation index, ForkNumber forkNum);
void InitializeStatusMemtableArray(Relation index, ForkNumber forkNum);
void RegisterStatusMemtable(Relation index, SegmentId sid);
void ReleaseStatusMemtable(Relation index, SegmentId sid);
void AddToStatusMemtable(Relation index, ForkNumber forkNum, SegmentId sid, ItemPointerData tid);

#endif