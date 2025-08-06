#include "postgres.h"

#include "access/generic_xlog.h"
#include "commands/vacuum.h"
#include "ivfflat.h"
#include "storage/bufmgr.h"
#include "visibility.h"
#include "lsm_index_struct.h"

/*
 * Bulk delete tuples from the index
 */
IndexBulkDeleteResult *
ivfflatbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
				  IndexBulkDeleteCallback callback, void *callback_state)
{
	Relation	index = info->index;
	BlockNumber blkno = IVFFLAT_HEAD_BLKNO;
	BufferAccessStrategy bas = GetAccessStrategy(BAS_BULKREAD);

	if (stats == NULL)
		stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));

	/* Iterate over list pages */
	while (BlockNumberIsValid(blkno))
	{
		Buffer		cbuf;
		Page		cpage;
		OffsetNumber coffno;
		OffsetNumber cmaxoffno;
		BlockNumber startPages[MaxOffsetNumber];
		ListInfo	listInfo;

		cbuf = ReadBuffer(index, blkno);
		LockBuffer(cbuf, BUFFER_LOCK_SHARE);
		cpage = BufferGetPage(cbuf);

		cmaxoffno = PageGetMaxOffsetNumber(cpage);

		/* Iterate over lists */
		for (coffno = FirstOffsetNumber; coffno <= cmaxoffno; coffno = OffsetNumberNext(coffno))
		{
			IvfflatList list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, coffno));

			startPages[coffno - FirstOffsetNumber] = list->startPage;
		}

		listInfo.blkno = blkno;
		blkno = IvfflatPageGetOpaque(cpage)->nextblkno;

		UnlockReleaseBuffer(cbuf);

		for (coffno = FirstOffsetNumber; coffno <= cmaxoffno; coffno = OffsetNumberNext(coffno))
		{
			BlockNumber searchPage = startPages[coffno - FirstOffsetNumber];
			BlockNumber insertPage = InvalidBlockNumber;

			/* Iterate over entry pages */
			while (BlockNumberIsValid(searchPage))
			{
				Buffer		buf;
				Page		page;
				GenericXLogState *state;
				OffsetNumber offno;
				OffsetNumber maxoffno;
				OffsetNumber deletable[MaxOffsetNumber];
				int			ndeletable;

				vacuum_delay_point();

				buf = ReadBufferExtended(index, MAIN_FORKNUM, searchPage, RBM_NORMAL, bas);

				/*
				 * ambulkdelete cannot delete entries from pages that are
				 * pinned by other backends
				 *
				 * https://www.postgresql.org/docs/current/index-locking.html
				 */
				LockBufferForCleanup(buf);

				state = GenericXLogStart(index);
				page = GenericXLogRegisterBuffer(state, buf, 0);

				maxoffno = PageGetMaxOffsetNumber(page);
				ndeletable = 0;

				/* Find deleted tuples */
				for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
				{
					IndexTuple	itup = (IndexTuple) PageGetItem(page, PageGetItemId(page, offno));
					ItemPointer htup = &(itup->t_tid);

					if (callback(htup, callback_state))
					{
						deletable[ndeletable++] = offno;
						stats->tuples_removed++;
					}
					else
						stats->num_index_tuples++;
				}

				/* Set to first free page */
				/* Must be set before searchPage is updated */
				if (!BlockNumberIsValid(insertPage) && ndeletable > 0)
					insertPage = searchPage;

				searchPage = IvfflatPageGetOpaque(page)->nextblkno;

				if (ndeletable > 0)
				{
					/* Delete tuples */
					PageIndexMultiDelete(page, deletable, ndeletable);
					GenericXLogFinish(state);
				}
				else
					GenericXLogAbort(state);

				UnlockReleaseBuffer(buf);
			}

			/*
			 * Update after all tuples deleted.
			 *
			 * We don't add or delete items from lists pages, so offset won't
			 * change.
			 */
			if (BlockNumberIsValid(insertPage))
			{
				listInfo.offno = coffno;
				IvfflatUpdateList(index, listInfo, insertPage, InvalidBlockNumber, InvalidBlockNumber, MAIN_FORKNUM);
			}
		}
	}

	FreeAccessStrategy(bas);

	return stats;
}

IndexBulkDeleteResult *
our_ivfflatbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
					  IndexBulkDeleteCallback callback, void *callback_state)
{
	Relation index = info->index;
	BufferAccessStrategy bas = GetAccessStrategy(BAS_BULKREAD);

	if (stats == NULL)
	{
		stats = (IndexBulkDeleteResult *)palloc0(sizeof(IndexBulkDeleteResult));
	}

	BlockNumber searchPage = InvalidBlockNumber;
	GetVisibilityStartPage(index, &searchPage);
	BlockNumber insertPage = InvalidBlockNumber;

	// create the index tuple descriptor
	TupleDesc vitupdesc = CreateTemplateTupleDesc(1);
	TupleDescInitEntry(vitupdesc, (AttrNumber) 1, "vid", INT8OID, -1, 0);
	Datum vid_datum;
	bool isnull;

	while (BlockNumberIsValid(searchPage))
	{
		Buffer buf;
		Page page;
		GenericXLogState *state;
		OffsetNumber offno;
		OffsetNumber maxoffno;
		OffsetNumber deletable[MaxOffsetNumber];
		VectorId vids[MaxOffsetNumber];
		int ndeletable;

		vacuum_delay_point();

		buf = ReadBufferExtended(index, MAIN_FORKNUM, searchPage, RBM_NORMAL, bas);

		LockBufferForCleanup(buf);

		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);

		maxoffno = PageGetMaxOffsetNumber(page);
		ndeletable = 0;

		for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			IndexTuple itup = (IndexTuple) PageGetItem(page, PageGetItemId(page, offno));
			ItemPointer htup = &(itup->t_tid);
			vid_datum = index_getattr(itup, 1, vitupdesc, &isnull);

			if (callback(htup, callback_state))
			{
				deletable[ndeletable++] = offno;
				stats->tuples_removed++;
				// elog(DEBUG1, "[our_ivfflatbulkdelete] deletable[%d] = (OffsetNumber) %u", ndeletable-1, offno);
				if (!isnull)
				{
					VectorId vid = DatumGetInt64(vid_datum);
					vids[ndeletable - 1] = vid;
					// elog(DEBUG1, "[our_ivfflatbulkdelete] vids[%d] = %ld", ndeletable - 1, vid);
				}
				else
				{
					elog(ERROR, "[our_ivfflatbulkdelete] failed to retrieve vid from the index tuple");
				}
			}
			else
				stats->num_index_tuples++;
		}

		if (!BlockNumberIsValid(insertPage) && ndeletable > 0)
			insertPage = searchPage;
		
		searchPage = IvfflatPageGetOpaque(page)->nextblkno;
		
		if (ndeletable > 0)
		{
			/* Delete tuples */
			bulk_delete_lsm_index(index, vids, ndeletable);
			PageIndexMultiDelete(page, deletable, ndeletable);
			GenericXLogFinish(state);
		}
		else
			GenericXLogAbort(state);

		UnlockReleaseBuffer(buf);		
	}
	if (BlockNumberIsValid(insertPage))
	{
		VisibilityUpdateMetadata(index, insertPage, InvalidBlockNumber, InvalidBlockNumber, MAIN_FORKNUM);
	}

	FreeAccessStrategy(bas);

	return stats;
}

/*
 * Clean up after a VACUUM operation
 */
IndexBulkDeleteResult *
ivfflatvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	Relation	rel = info->index;

	if (info->analyze_only)
		return stats;

	/* stats is NULL if ambulkdelete not called */
	/* OK to return NULL if index not changed */
	if (stats == NULL)
		return NULL;

	stats->num_pages = RelationGetNumberOfBlocks(rel);

	return stats;
}
