#include "c.h"
#include "postgres.h"
#include "statuspage.h"
#include "access/generic_xlog.h"
#include "storage/bufmgr.h"
#include "storage/itemptr.h"
#include "storage/bufpage.h"
#include "storage/lmgr.h"
#include "access/itup.h"
#include <stdbool.h>


static Buffer
StatusNewBuffer(Relation index, ForkNumber forkNum)
{
    Buffer buf = ReadBufferExtended(index, forkNum, P_NEW, RBM_NORMAL, NULL);
    
    LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
    return buf;
}

static void
StatusInitPage(Buffer buf, Page page)
{
    PageInit(page, BufferGetPageSize(buf), sizeof(StatusPageOpaqueData));
    StatusPageGetOpaque(page)->nextblkno = InvalidBlockNumber;
    StatusPageGetOpaque(page)->page_id = STATUS_PAGE_ID;
}

static void
StatusInitRegisterPage(Relation index, Buffer *buf, Page *page, GenericXLogState **state)
{
    *state = GenericXLogStart(index);
    *page = GenericXLogRegisterBuffer(*state, *buf, GENERIC_XLOG_FULL_IMAGE);
    StatusInitPage(*buf, *page);
}

static void
StatusCommitBuffer(Buffer buf, GenericXLogState *state)
{
    GenericXLogFinish(state);
    UnlockReleaseBuffer(buf);
}

// This function is called when the index is built
void 
CreateStatusMetaPage(Relation index, ForkNumber forkNum)
{
    Buffer buf;
    Page page;
    GenericXLogState *state;
    StatusPageMeta metap;

    buf = StatusNewBuffer(index, forkNum);
    StatusInitRegisterPage(index, &buf, &page, &state);

    /* Set metapage data */
    metap = StatusPageGetMeta(page);
    metap->freePageHead = InvalidBlockNumber;
    metap->freePageTail = InvalidBlockNumber;

    StatusCommitBuffer(buf, state);
}

// This function is called when the index is built
void
InitializeStatusMemtableArray(Relation index, ForkNumber forkNum)
{
    Buffer buf;
    Page page;
    GenericXLogState *state;

    // the head of the memtable array pages
    buf = StatusNewBuffer(index, forkNum);
    StatusInitRegisterPage(index, &buf, &page, &state);

    StatusCommitBuffer(buf, state);
}

void
ReleaseStatusMemtable(Relation index, SegmentId sid)
{
    // TODO: ensure the index is valid

    Buffer buf;
    Page page;
    GenericXLogState *state;
    OffsetNumber maxoffno;
    bool found = false;

    Buffer freePageTailBuf = InvalidBuffer;
    Page freePageTail;
    StatusPageOpaque freePageTailOpaque;
    Buffer metapagebuf = InvalidBuffer;
    Page metapage;
    StatusPageMeta statuspagemeta;

    BlockNumber nextblkno = STATUS_MEMTABLE_ARRAY_BLKNO;
    while (BlockNumberIsValid(nextblkno))
    {
        buf = ReadBuffer(index, nextblkno);
        LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

        state = GenericXLogStart(index);
        page = GenericXLogRegisterBuffer(state, buf, 0);
        maxoffno = PageGetMaxOffsetNumber(page);

        for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
        {
            StatusMemtable mt = (StatusMemtable) PageGetItem(page, PageGetItemId(page, offno));
            if (mt->sid == sid)
            {
                // move the pages to the free page list
                metapagebuf = ReadBuffer(index, STATUS_METAPAGE_BLKNO);
                LockBuffer(metapagebuf, BUFFER_LOCK_EXCLUSIVE);
                metapage = GenericXLogRegisterBuffer(state, metapagebuf, 0);
                statuspagemeta = StatusPageGetMeta(metapage);
                if (statuspagemeta->freePageHead == InvalidBlockNumber)
                {
                    statuspagemeta->freePageHead = mt->memtablePageHead;
                    // At this point, the memtable's page tail is the same as its insert page
                    statuspagemeta->freePageTail = mt->memtableInsertPage;
                }
                else
                {
                    freePageTailBuf = ReadBuffer(index, statuspagemeta->freePageTail);
                    LockBuffer(freePageTailBuf, BUFFER_LOCK_EXCLUSIVE);
                    freePageTail = GenericXLogRegisterBuffer(state, freePageTailBuf, 0);
                    freePageTailOpaque = StatusPageGetOpaque(freePageTail);
                    freePageTailOpaque->nextblkno = mt->memtablePageHead;
                    // At this point, the memtable's page tail is the same as its insert page
                    statuspagemeta->freePageTail = mt->memtableInsertPage;
                }

                // delete the memtable item
                PageIndexTupleDelete(page, offno);

                StatusCommitBuffer(buf, state);
                if (BufferIsValid(freePageTailBuf))
                {
                    UnlockReleaseBuffer(freePageTailBuf);
                }
                if (BufferIsValid(metapagebuf))
                {
                    UnlockReleaseBuffer(metapagebuf);
                }
                found = true;
                break;
            }
        }
        if (found)
        {
            break;
        }
        nextblkno = StatusPageGetOpaque(page)->nextblkno;
        GenericXLogAbort(state);
        UnlockReleaseBuffer(buf);
    }
}

// Need to start GenericXLogState before calling this function
// finish GenericXLog and release the buffers after using the returned buffers
static bool
GetFreePage(Relation index, ForkNumber forkNum, GenericXLogState *state, Buffer *ret_metabuf, Buffer *ret_buf)
{
    Buffer metabuf;
    Page metapage;
    StatusPageMeta statuspagemeta;
    Buffer buf;
    Page page;
    *ret_metabuf = InvalidBuffer;
    *ret_buf = InvalidBuffer;

    metabuf = ReadBuffer(index, STATUS_METAPAGE_BLKNO);
    LockBuffer(metabuf, BUFFER_LOCK_EXCLUSIVE);

    metapage = GenericXLogRegisterBuffer(state, metabuf, 0);
    statuspagemeta = StatusPageGetMeta(metapage);

    // no free page available
    if (statuspagemeta->freePageHead == InvalidBlockNumber)
    {
        UnlockReleaseBuffer(metabuf);
        return false;
    }

    // read and register the free page
    buf = ReadBuffer(index, statuspagemeta->freePageHead);
    LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
    page = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);

    // pop the free page from the free page list
    statuspagemeta->freePageHead = StatusPageGetOpaque(page)->nextblkno;
    if (statuspagemeta->freePageHead == InvalidBlockNumber)
    {
        statuspagemeta->freePageTail = InvalidBlockNumber;
    }

    // initialize the page
    StatusInitPage(buf, page);

    *ret_metabuf = metabuf;
    *ret_buf = buf;
    return true;
}

// FIXME: potential race condition when registering the new memtable?
void 
RegisterStatusMemtable(Relation index, SegmentId sid)
{
    Buffer buf;
    Page page;
    GenericXLogState *state;
    BlockNumber insertPage = STATUS_MEMTABLE_ARRAY_BLKNO;
    StatusMemtable mt;
    Size mtSize;

    /* Form tuple */
    mtSize = MAXALIGN(sizeof(StatusMemtableData));
    mt = (StatusMemtable) palloc0(mtSize);
    MemSet(mt, 0, mtSize);
    mt->sid = sid;
    mt->memtablePageHead = InvalidBlockNumber;
    mt->memtableInsertPage = InvalidBlockNumber;

    for (;;)
    {
        buf = ReadBuffer(index, insertPage);
        LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

        state = GenericXLogStart(index);
        page = GenericXLogRegisterBuffer(state, buf, 0);
        
        if (PageGetFreeSpace(page) > MAXALIGN(sizeof(StatusMemtableData)))
            break;

        insertPage = StatusPageGetOpaque(page)->nextblkno;

        if (BlockNumberIsValid(insertPage))
        {
            GenericXLogAbort(state);
            UnlockReleaseBuffer(buf);
        }
        else
        {
            Buffer newbuf;
            Page newpage;

            /* Add a new page */
            LockRelationForExtension(index, ExclusiveLock);
            newbuf = StatusNewBuffer(index, MAIN_FORKNUM);
            UnlockRelationForExtension(index, ExclusiveLock);

            /* Initialize the new page */
            newpage = GenericXLogRegisterBuffer(state, newbuf, GENERIC_XLOG_FULL_IMAGE);
            StatusInitPage(newbuf, newpage);

            /* Update insert page */
            insertPage = BufferGetBlockNumber(newbuf);

            /* Update previous page */
            StatusPageGetOpaque(page)->nextblkno = insertPage;

            /* Commit the changes */
            GenericXLogFinish(state);
            UnlockReleaseBuffer(buf);

            /* Prepare new buffer */
            state = GenericXLogStart(index);
            buf = newbuf;
            page = GenericXLogRegisterBuffer(state, buf, 0);
            break;
        }
    }

    /* Add to next offset */
    if(PageAddItem(page, (Item) mt, mtSize, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
        elog(ERROR, "[RegisterStatusMemtable] Failed to add memtable to page");

    /* Get a free page */
    Buffer ret_metabuf;
    Buffer ret_buf;
    bool free_page_result = GetFreePage(index, state, &ret_metabuf, &ret_buf);
    if (free_page_result)
    {
        // update the memtable page head
        mt->memtablePageHead = BufferGetBlockNumber(ret_buf);
        // update the memtable insert page
        mt->memtableInsertPage = BufferGetBlockNumber(ret_buf);
    
        StatusCommitBuffer(buf, state);
        UnlockReleaseBuffer(ret_buf);
        UnlockReleaseBuffer(ret_metabuf);
    }
    /* If no free page is available, add a new page */
    else
    {
        // add a new page
        Buffer newbuf;
        Page newpage;
        LockRelationForExtension(index, ExclusiveLock);
        newbuf = StatusNewBuffer(index, MAIN_FORKNUM);
        UnlockRelationForExtension(index, ExclusiveLock);
        
        newpage = GenericXLogRegisterBuffer(state, newbuf, GENERIC_XLOG_FULL_IMAGE);
        StatusInitPage(newbuf, newpage);

        mt->memtablePageHead = BufferGetBlockNumber(newbuf);
        mt->memtableInsertPage = BufferGetBlockNumber(newbuf);

        StatusCommitBuffer(buf, state);
        UnlockReleaseBuffer(newbuf);
    }
}

static bool
FindMemtableInsertPage(Relation index, SegmentId sid, BlockNumber *insertPage, MemtableInfo *info)
{
    BlockNumber nextblkno = STATUS_MEMTABLE_ARRAY_BLKNO;

    // Search memtable array 
    while (BlockNumberIsValid(nextblkno))
    {
        Buffer buf;
        Page page;
        OffsetNumber maxoffno;

        buf = ReadBuffer(index, nextblkno);
        LockBuffer(buf, BUFFER_LOCK_SHARE);
        page = BufferGetPage(buf);
        maxoffno = PageGetMaxOffsetNumber(page);

        for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
        {
            StatusMemtable mt;
            
            mt = (StatusMemtable) PageGetItem(page, PageGetItemId(page, offno));
            if (mt->sid == sid)
            {
                info->blkno = nextblkno;
                info->offno = offno;
                *insertPage = mt->memtableInsertPage;
                UnlockReleaseBuffer(buf);
                return true;
            }
        }
        nextblkno = StatusPageGetOpaque(page)->nextblkno;
        UnlockReleaseBuffer(buf);
    }
    return false;
}

static void
StatusUpdateInsertPage(Relation index, MemtableInfo info, BlockNumber insertPage, BlockNumber originalInsertPage, ForkNumber forkNum)
{
    Buffer buf;
    Page page;
    GenericXLogState *state;
    StatusMemtable mt;
    bool changed = false;

    buf = ReadBufferExtended(index, forkNum, info.blkno, RBM_NORMAL, NULL);
    LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
    state = GenericXLogStart(index);
    page = GenericXLogRegisterBuffer(state, buf, 0);
    mt = (StatusMemtable) PageGetItem(page, PageGetItemId(page, info.offno));
    
    if (BlockNumberIsValid(insertPage) && insertPage != mt->memtableInsertPage)
    {
        if (!BlockNumberIsValid(originalInsertPage) || insertPage >= originalInsertPage)
        {
            mt->memtableInsertPage = insertPage;
            changed = true;
        }
    }
    if (changed)
    {
        StatusCommitBuffer(buf, state);
    }
    else
    {
        GenericXLogAbort(state);
        UnlockReleaseBuffer(buf);
    }
}

void 
AddToStatusMemtable(Relation index, ForkNumber forkNum, SegmentId sid, ItemPointerData tid)
{
    Buffer buf;
    Page page;
    GenericXLogState *state;
    BlockNumber insertPage = InvalidBlockNumber;
    MemtableInfo info;
    StatusTuple itup;
    Size itupSize;
    BlockNumber originalInsertPage;

    // TODO: ensure the index is valid

    /* Form tuple */
    itupSize = MAXALIGN(sizeof(StatusTupleData));
    itup = (StatusTuple) palloc0(itupSize);
    MemSet(itup, 0, itupSize);
    itup->t_tid = tid;

    /* Find the insert page */
    FindMemtableInsertPage(index, sid, &insertPage, &info);
    Assert(BlockNumberIsValid(insertPage));
    originalInsertPage = insertPage;

    for (;;)
    {
        buf = ReadBuffer(index, insertPage);
        LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

        state = GenericXLogStart(index);
        page = GenericXLogRegisterBuffer(state, buf, 0);

        if (PageGetFreeSpace(page) > itupSize)
            break;

        insertPage = StatusPageGetOpaque(page)->nextblkno;

        if (BlockNumberIsValid(insertPage))
        {
            GenericXLogAbort(state);
            UnlockReleaseBuffer(buf);
        }
        else
        {
            /* Get a free page */
            Buffer metabuf;
            Buffer new_buf;
            bool free_page_result = GetFreePage(index, forkNum, state, &metabuf, &new_buf);
            if (free_page_result)
            {
                // update nextblkno
                StatusPageGetOpaque(page)->nextblkno = BufferGetBlockNumber(new_buf);
                // update the insert page
                insertPage = BufferGetBlockNumber(new_buf);
                StatusCommitBuffer(buf, state);
                UnlockReleaseBuffer(new_buf);
                UnlockReleaseBuffer(metabuf);
            }
            /* If no free page is available, add a new page */
            else
            {
                Page newpage;

                // add a new page
                LockRelationForExtension(index, ExclusiveLock);
                new_buf = StatusNewBuffer(index, forkNum);
                UnlockRelationForExtension(index, ExclusiveLock);
                
                newpage = GenericXLogRegisterBuffer(state, new_buf, GENERIC_XLOG_FULL_IMAGE);
                StatusInitPage(new_buf, newpage);

                // update nextblkno
                StatusPageGetOpaque(page)->nextblkno = BufferGetBlockNumber(new_buf);
                // update the insert page
                insertPage = BufferGetBlockNumber(new_buf);

                StatusCommitBuffer(buf, state);
                UnlockReleaseBuffer(new_buf);
            }

            /* Prepare new buffer */
            state = GenericXLogStart(index);
            buf = new_buf;
            page = GenericXLogRegisterBuffer(state, buf, 0);
            break;
        }
    }

    /* Add to next offset */
    if(PageAddItem(page, (Item) itup, itupSize, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
        elog(ERROR, "[AddToStatusMemtable] Failed to add tuple to page");
    
    StatusCommitBuffer(buf, state);
    
    /* Update the insert page */    
    if (insertPage != originalInsertPage) {
        StatusUpdateInsertPage(index, info, insertPage, originalInsertPage, forkNum);
    }
}