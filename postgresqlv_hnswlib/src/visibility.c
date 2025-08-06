#include "postgres.h"

#include "access/genam.h"
#include "storage/bufpage.h"
#include "storage/smgr.h"
#include "storage/buf_internals.h"
#include "access/generic_xlog.h"
#include "visibility.h"
#include "ivfflat.h"
#include "lsm_index_struct.h"

void
VisibilityUpdateMetadata(Relation index, BlockNumber insertPage, BlockNumber originalInsertPage,
    BlockNumber startPage, ForkNumber forkNum)
{
    // elog(DEBUG1, "enter VisibilityUpdateMetadata: insertPage = %u, startPage = %u", insertPage, startPage);

    Buffer		buf;
	Page		page;
	GenericXLogState *state;
    bool		changed = false;

    buf = ReadBufferExtended(index, forkNum, VISIBILITY_HEAD_BLKNO, RBM_NORMAL, NULL);
    LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
    
    state = GenericXLogStart(index);
    page = GenericXLogRegisterBuffer(state, buf, 0);

    VisibilityMetaPage visip = VisibilityPageGetMeta(page);


    if (BlockNumberIsValid(insertPage) && insertPage != visip->insertPage)
    {
        if (!BlockNumberIsValid(originalInsertPage) || insertPage >= originalInsertPage)
        {
            visip->insertPage = insertPage;
            changed = true;
        }
    }

    if (BlockNumberIsValid(startPage) && startPage != visip->startPage)
    {
        visip->startPage = startPage;
        changed = true;
    }

    // elog(DEBUG1, "[VisibilityUpdateMetadata] after update: visip->insertPage = %u, visip->startPage = %u", visip->insertPage, visip->startPage);
    if (changed)
    {
        ((PageHeader) page)->pd_lower = ((char *) visip + sizeof(VisibilityMetaPageData)) - (char *) page;
        IvfflatCommitBuffer(buf, state);
    }
    else
    {
        GenericXLogAbort(state);
        UnlockReleaseBuffer(buf);
    }
}

void
GetVisibilityInsertPage(Relation index, BlockNumber *insertPage)
{
    Buffer buf;
    Page page;

    buf = ReadBuffer(index, VISIBILITY_HEAD_BLKNO);
    LockBuffer(buf, BUFFER_LOCK_SHARE);
    page = BufferGetPage(buf);

    VisibilityMetaPage visip = VisibilityPageGetMeta(page);
    *insertPage = visip->insertPage;
    UnlockReleaseBuffer(buf);
}

void
GetVisibilityStartPage(Relation index, BlockNumber *startPage)
{
    Buffer buf;
    Page page;

    buf = ReadBuffer(index, VISIBILITY_HEAD_BLKNO);
    LockBuffer(buf, BUFFER_LOCK_SHARE);
    page = BufferGetPage(buf);

    VisibilityMetaPage visip = VisibilityPageGetMeta(page);
    *startPage = visip->startPage;
    UnlockReleaseBuffer(buf);
}


void
CreateVisibilityMetaPage(Relation index, ForkNumber forkNum)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	VisibilityMetaPage visip;
	
	buf = IvfflatNewBuffer(index, forkNum);
	IvfflatInitRegisterPage(index, &buf, &page, &state);

	/* set visibility metadata*/
	visip = VisibilityPageGetMeta(page);
	visip->insertPage = InvalidBlockNumber;
	visip->startPage = InvalidBlockNumber;
	
	IvfflatCommitBuffer(buf, state);
}

void
InsertVisibilityTuples(ForkNumber forkNum, Relation index, int num_tids, TupleDesc tupdesc, int64_t *tids)
{
	IndexTuple	itup = NULL;
	int64_t vid = 0;
	Datum vid_datum;
	bool isnull = false;

	Buffer buf;
	Page page;
	GenericXLogState *state;
	BlockNumber startPage;
	BlockNumber insertPage;

	buf = IvfflatNewBuffer(index, forkNum);
	IvfflatInitRegisterPage(index, &buf, &page, &state);
	startPage = BufferGetBlockNumber(buf);

	for (size_t i = 0; i < num_tids; i++)
	{
		// generate an `IndexTuple`
		vid_datum = Int64GetDatum(vid);
		isnull = false;

		itup = index_form_tuple(tupdesc, &vid_datum, &isnull);
		itup->t_tid = Int64ToItemPointer(tids[i]);
		++vid;

		/* Check for free space */
		Size		itemsz = MAXALIGN(IndexTupleSize(itup));

		if (PageGetFreeSpace(page) < itemsz)
			IvfflatAppendPage(index, &buf, &page, &state, forkNum);
		
		/* Add the item */
		if (PageAddItem(page, (Item) itup, itemsz, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

		pfree(itup);
	}

	insertPage = BufferGetBlockNumber(buf);

	IvfflatCommitBuffer(buf, state);

	// set the start and insert pages
	VisibilityUpdateMetadata(index, insertPage, InvalidBlockNumber, startPage, forkNum);
}


void
InsertVisibilityTuple(Relation index, uint64_t vid, ItemPointer heap_tid, Relation heapRel)
{
	BlockNumber insertPage = InvalidBlockNumber;
	BlockNumber originalInsertPage;

	// get the insert page
	GetVisibilityInsertPage(index, &insertPage);
	Assert(BlockNumberIsValid(insertPage));
	originalInsertPage = insertPage;

	Datum vid_datum = Int64GetDatum(vid);
	bool isnull = false;
	IndexTuple itup = NULL;

	// form the index tuple
	// TODO: we need to store the `TupleDesc` somewhere
	TupleDesc vitupdesc = CreateTemplateTupleDesc(1);
	TupleDescInitEntry(vitupdesc, (AttrNumber) 1, "vid", INT8OID, -1, 0);
	itup = index_form_tuple(vitupdesc, &vid_datum, &isnull);
	ItemPointerCopy(heap_tid, &(itup->t_tid));

	// get the tuple size
	Size itemsz;
	itemsz = MAXALIGN(IndexTupleSize(itup));
	Assert(itemsz <= BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(IvfflatPageOpaqueData)) - sizeof(ItemIdData));

	// find a page to insert the item
	Buffer buf;
	Page page;
	GenericXLogState *state;
	for (;;)
	{
		buf = ReadBuffer(index, insertPage);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);

		if (PageGetFreeSpace(page) >= itemsz)
			break;
		
		// check next page
		insertPage = IvfflatPageGetOpaque(page)->nextblkno;

		if (BlockNumberIsValid(insertPage))
		{
			/* Move to next page */
			GenericXLogAbort(state);
			UnlockReleaseBuffer(buf);
		}
		else
		{
			Buffer		newbuf;
			Page		newpage;

			/* Add a new page */
			LockRelationForExtension(index, ExclusiveLock);
			newbuf = IvfflatNewBuffer(index, MAIN_FORKNUM);
			UnlockRelationForExtension(index, ExclusiveLock);

			/* Init new page */
			newpage = GenericXLogRegisterBuffer(state, newbuf, GENERIC_XLOG_FULL_IMAGE);
			IvfflatInitPage(newbuf, newpage);

			/* Update insert page */
			insertPage = BufferGetBlockNumber(newbuf);

			/* Update previous buffer */
			IvfflatPageGetOpaque(page)->nextblkno = insertPage;

			/* Commit */
			GenericXLogFinish(state);

			/* Unlock previous buffer */
			UnlockReleaseBuffer(buf);

			/* Prepare new buffer */
			state = GenericXLogStart(index);
			buf = newbuf;
			page = GenericXLogRegisterBuffer(state, buf, 0);
			break;
		}
	}

	/* Add to next offset */
	if (PageAddItem(page, (Item) itup, itemsz, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
		elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

	IvfflatCommitBuffer(buf, state);
	/* Update the insert page */
	if (insertPage != originalInsertPage)
		VisibilityUpdateMetadata(index, insertPage, originalInsertPage, InvalidBlockNumber, MAIN_FORKNUM);
}