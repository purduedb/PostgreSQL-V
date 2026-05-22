#ifndef DPV_PENDING_FETCH_QUEUE_H
#define DPV_PENDING_FETCH_QUEUE_H

#include "postgres.h"
#include "access/xlog.h"
#include "lsm_segment.h"  /* SegmentId */

typedef enum {
    DPV_FETCH_PENDING  = 0,
    DPV_FETCH_FETCHING = 1,
    DPV_FETCH_DONE     = 2,
    DPV_FETCH_FAILED   = 3,
} DpvFetchStatus;

typedef enum {
    DPV_FETCH_KIND_CREATED  = 1,
    DPV_FETCH_KIND_REPLACED = 2,
} DpvFetchKind;

#define DPV_FETCH_QUEUE_DIR "_pending_fetches"

typedef struct {
    Oid             indexRelId;
    SegmentId       start_sid;
    SegmentId       end_sid;
    uint32          version;
    DpvFetchKind    kind;
    XLogRecPtr      source_lsn;       /* diagnostic only */
    DpvFetchStatus  status;
    /* offsets[] is appended after this fixed header in the on-disk format
     * (for kind=REPLACED). Length is recovered from the file size. */
} DpvFetchEntryHeader;

/* Lifecycle. */
extern void dpv_queue_init(void);              /* called on bgworker startup */

/* Producer (redo callback). Returns true if newly enqueued, false if duplicate. */
extern bool dpv_queue_enqueue(const DpvFetchEntryHeader *hdr,
                              const void *trailer, Size trailer_size);

/* Consumer (fetcher worker). Returns NULL if no pending entry available; the
 * returned pointer is palloc'd in the caller's memory context. */
typedef struct {
    DpvFetchEntryHeader hdr;
    char                filename[64];  /* identifies the on-disk entry file */
    Size                trailer_size;
    char               *trailer;       /* palloc'd; NULL if trailer_size==0 */
} DpvFetchEntry;

extern DpvFetchEntry *dpv_queue_pop_pending(void);
extern void dpv_queue_mark(const char *filename, DpvFetchStatus new_status);

/* Re-scan disk on standby startup and put PENDING/FETCHING back into the
 * in-memory ready set (FETCHING means a worker crashed mid-fetch). */
extern void dpv_queue_recover_on_startup(void);

#endif
