# Physical Replication — Plan 2: Side-Channel + Segment Adoption

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Plan 1 is merged. Custom rmgr exists; memtable lifecycle records replicate.

**Goal:** Build the side channel (file server on primary, fetcher worker on standby), the persistent fetch queue, and the WAL records that trigger pulls (`SegmentCreated`, `SegmentReplaced`). Implement the coverage-based adoption rule from spec §8 so that pulled segments atomically replace the memtables/segments they cover on the standby's `FlushedSegmentPool` and `SharedMemtableBuffer`. (Bitmap translation across versions is **out of scope** for this plan — that arrives in Plan 3 alongside `SegmentVacuumTombstones`.)

**Architecture:** The primary registers a new bgworker `replication_server` that listens on TCP and streams segment-content files from `VECTOR_STORAGE_BASE_DIR` on request. The standby registers one or more `segment_fetcher` bgworkers and a small persistent queue under `VECTOR_STORAGE_BASE_DIR/_pending_fetches/`. Redo for `SegmentCreated` and `SegmentReplaced` appends to the queue; the fetcher pops, pulls the five files (`index`, `mapping`, `offset`, `bitmap`, `metadata` — metadata last), and then attempts adoption via the coverage rule. Adoption holds `pool->seg_lock` + `lsm->mt_lock` and either (a) replaces a covered group with the new segment, (b) discards the file as stale, or (c) leaves the file in place for `load_lsm_index_internal` to pick up later (index not loaded).

**Tech Stack:** PostgreSQL 17 (PGXS extension), C (file IO, bgworker, custom WAL records), Perl integration tests. No external libraries — the protocol is a length-prefixed binary stream over TCP using PG's `pq_*` helpers (also usable outside the FE/BE wire protocol).

**Scope of this plan (deferred to Plan 3):**
- No `SegmentVacuumTombstones` record / vacuum WAL → file protocol.
- No bitmap translation (lazy sorted-permutation aux structure).
- No standby attach-time barrier ("not queryable" wait until fetcher catches up).
- No write-worker disable on standby (`lsm_index_bgworker`, merge pool) — for now they're harmless because they require WAL-emitting paths that don't run on the standby anyway; Plan 3 makes the gate explicit.

---

## File structure

| File | Role | Action |
| --- | --- | --- |
| [pgvector/src/replication_rmgr.h](../../../pgvector/src/replication_rmgr.h) | Add `SegmentCreated` and `SegmentReplaced` record types + payload structs + emit prototypes | Modify |
| [pgvector/src/replication_rmgr.c](../../../pgvector/src/replication_rmgr.c) | Add emit helpers and redo callbacks (redo enqueues into the fetch queue) | Modify |
| [pgvector/src/replication_gucs.h](../../../pgvector/src/replication_gucs.h) | GUC name macros and accessor extern decls | Create |
| [pgvector/src/replication_gucs.c](../../../pgvector/src/replication_gucs.c) | `DefineCustomStringVariable` / `DefineCustomIntVariable` calls; accessors | Create |
| [pgvector/src/pending_fetch_queue.h](../../../pgvector/src/pending_fetch_queue.h) | Queue API: enqueue, dequeue-pending, mark-done, scan-pending-on-startup | Create |
| [pgvector/src/pending_fetch_queue.c](../../../pgvector/src/pending_fetch_queue.c) | One small file per entry under `VECTOR_STORAGE_BASE_DIR/_pending_fetches/<uuid>.entry`; atomic temp-then-rename writes | Create |
| [pgvector/src/replication_server.h](../../../pgvector/src/replication_server.h) | Header for the file server bgworker | Create |
| [pgvector/src/replication_server.c](../../../pgvector/src/replication_server.c) | bgworker; TCP listen; per-connection request/response loop; streams files using `mmap` or chunked `read`/`pq_putbytes` | Create |
| [pgvector/src/segment_fetcher.h](../../../pgvector/src/segment_fetcher.h) | Header for the fetcher bgworker | Create |
| [pgvector/src/segment_fetcher.c](../../../pgvector/src/segment_fetcher.c) | bgworker; pops queue, connects to primary, downloads the 5 files (metadata last via atomic rename), then calls `dpv_attempt_adoption` | Create |
| [pgvector/src/segment_adoption.h](../../../pgvector/src/segment_adoption.h) | Coverage-rule API: `dpv_attempt_adoption(indexRelId, start_sid, end_sid, version)` | Create |
| [pgvector/src/segment_adoption.c](../../../pgvector/src/segment_adoption.c) | Coverage rule (§8 table); bitmap UNION for v1 (no translation — translate identity is correct only when versions match exactly; when they don't, fall through to "wait for Plan 3") | Create |
| [pgvector/src/lsmbackground.c](../../../pgvector/src/lsmbackground.c) | After `flush_segment_to_disk` returns (call site), emit `SegmentCreated` | Modify |
| [pgvector/src/vector_index_worker.c](../../../pgvector/src/vector_index_worker.c) | After merge writes new file via atomic rename, emit `SegmentReplaced` | Modify |
| [pgvector/src/vector.c](../../../pgvector/src/vector.c) | Register the two new bgworkers conditional on role GUC; define GUCs | Modify |
| [pgvector/Makefile](../../../pgvector/Makefile) | Add new `.o` files | Modify |
| [pgvector/test/t/110_replication_segment_flush.pl](../../../pgvector/test/t/110_replication_segment_flush.pl) | Flush on primary → standby pool gets the segment | Create |
| [pgvector/test/t/111_replication_segment_merge.pl](../../../pgvector/test/t/111_replication_segment_merge.pl) | Merge on primary → standby pool reflects merge | Create |
| [pgvector/test/t/112_replication_queue_restart.pl](../../../pgvector/test/t/112_replication_queue_restart.pl) | Standby crash mid-fetch → queue re-enqueues, fetch completes after restart | Create |
| [pgvector/test/t/113_replication_out_of_order_pulls.pl](../../../pgvector/test/t/113_replication_out_of_order_pulls.pl) | Force out-of-order pull arrival; coverage rule keeps state consistent | Create |

---

## Phase 1 — GUCs

### Task 1.1: Define replication GUCs

**Files:**
- Create: `pgvector/src/replication_gucs.h`
- Create: `pgvector/src/replication_gucs.c`

- [ ] **Step 1: Header**

```c
#ifndef DPV_REPLICATION_GUCS_H
#define DPV_REPLICATION_GUCS_H

typedef enum {
    DPV_ROLE_DISABLED = 0,
    DPV_ROLE_PRIMARY  = 1,
    DPV_ROLE_STANDBY  = 2,
} DpvReplicationRole;

extern int  dpv_replication_role;            /* DpvReplicationRole */
extern char *dpv_replication_primary_host;
extern int   dpv_replication_primary_port;
extern char *dpv_replication_shared_secret;
extern int   dpv_replication_fetch_parallelism;

extern void dpv_replication_gucs_register(void);  /* call from _PG_init */

#endif
```

- [ ] **Step 2: Body**

```c
#include "postgres.h"
#include "utils/guc.h"
#include "replication_gucs.h"

int   dpv_replication_role = DPV_ROLE_DISABLED;
char *dpv_replication_primary_host = NULL;
int   dpv_replication_primary_port = 0;
char *dpv_replication_shared_secret = NULL;
int   dpv_replication_fetch_parallelism = 2;

static const struct config_enum_entry role_options[] = {
    { "disabled", DPV_ROLE_DISABLED, false },
    { "primary",  DPV_ROLE_PRIMARY,  false },
    { "standby",  DPV_ROLE_STANDBY,  false },
    { NULL, 0, false }
};

void
dpv_replication_gucs_register(void)
{
    DefineCustomEnumVariable("pgvector.replication_role",
        "Role of this node in pgvector replication.",
        NULL, &dpv_replication_role, DPV_ROLE_DISABLED,
        role_options, PGC_POSTMASTER, 0, NULL, NULL, NULL);

    DefineCustomStringVariable("pgvector.replication_primary_host",
        "Host of the primary's pgvector file server (standby only).",
        NULL, &dpv_replication_primary_host, "",
        PGC_POSTMASTER, 0, NULL, NULL, NULL);

    DefineCustomIntVariable("pgvector.replication_primary_port",
        "Port of the primary's pgvector file server (both roles).",
        NULL, &dpv_replication_primary_port, 0,
        0, 65535, PGC_POSTMASTER, 0, NULL, NULL, NULL);

    DefineCustomStringVariable("pgvector.replication_shared_secret",
        "Shared secret for the pgvector replication side channel.",
        NULL, &dpv_replication_shared_secret, "",
        PGC_POSTMASTER, GUC_SUPERUSER_ONLY, NULL, NULL, NULL);

    DefineCustomIntVariable("pgvector.replication_fetch_parallelism",
        "Number of segment-fetcher background workers (standby only).",
        NULL, &dpv_replication_fetch_parallelism, 2,
        1, 8, PGC_POSTMASTER, 0, NULL, NULL, NULL);
}
```

- [ ] **Step 3: Wire into `_PG_init`** in [vector.c](../../../pgvector/src/vector.c)

Near the existing extension registration calls (e.g., after `vector_replication_rmgr_register()` from Plan 1), add:

```c
#include "replication_gucs.h"
...
    dpv_replication_gucs_register();
```

- [ ] **Step 4: Add to OBJS and build**

```makefile
       src/replication_rmgr.o src/standby_memtable.o src/replication_gucs.o
```

Build:

```bash
cd pgvector && make 2>&1 | tail -20
```

- [ ] **Step 5: Commit**

```bash
git add pgvector/src/replication_gucs.h pgvector/src/replication_gucs.c \
        pgvector/src/vector.c pgvector/Makefile
git commit -m "feat: GUCs for pgvector physical replication"
```

---

## Phase 2 — Persistent fetch queue

### Task 2.1: Define the queue API

**Files:**
- Create: `pgvector/src/pending_fetch_queue.h`

- [ ] **Step 1: Header**

```c
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
```

### Task 2.2: Implement the queue

**Files:**
- Create: `pgvector/src/pending_fetch_queue.c`

Use a `dshash` table for in-memory dedup keyed by `(indexRelId, start_sid, end_sid, version, kind)`. The on-disk format is the source of truth; the in-memory table is a cache.

- [ ] **Step 1: Sketch the implementation**

```c
#include "postgres.h"
#include "miscadmin.h"
#include "storage/fd.h"
#include "storage/lwlock.h"
#include "utils/builtins.h"
#include "utils/dsa.h"
#include "utils/dynahash.h"
#include "common/file_perm.h"

#include "lsm_segment.h"
#include "lsmindex.h"          /* VECTOR_STORAGE_BASE_DIR */
#include "pending_fetch_queue.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dirent.h>
#include <unistd.h>

typedef struct {
    Oid          indexRelId;
    SegmentId    start_sid;
    SegmentId    end_sid;
    uint32       version;
    DpvFetchKind kind;
} DpvFetchKey;

typedef struct {
    DpvFetchKey key;
    char        filename[64];
} DpvFetchIndexEntry;

static HTAB *queue_index = NULL;
static LWLock *queue_lock = NULL;  /* registered tranche on init */

/* Internal: build the queue dir path. */
static void
queue_dir_path(char *buf, size_t buflen)
{
    snprintf(buf, buflen, "%s/%s", VECTOR_STORAGE_BASE_DIR, DPV_FETCH_QUEUE_DIR);
}

static void
ensure_queue_dir(void)
{
    char dir[MAXPGPATH];
    queue_dir_path(dir, sizeof(dir));
    if (mkdir(dir, pg_dir_create_mode) != 0 && errno != EEXIST)
        elog(ERROR, "[dpv_queue] mkdir %s: %m", dir);
}

void
dpv_queue_init(void)
{
    HASHCTL ctl;
    ensure_queue_dir();

    /* In-memory dedup index. Backend-private hash is fine: each fetcher worker
     * holds its own; deduplication across workers relies on the on-disk entry
     * filename being a content hash (see dpv_queue_enqueue). */
    MemSet(&ctl, 0, sizeof(ctl));
    ctl.keysize   = sizeof(DpvFetchKey);
    ctl.entrysize = sizeof(DpvFetchIndexEntry);
    ctl.hcxt      = CurrentMemoryContext;
    queue_index = hash_create("dpv_queue_index", 256, &ctl,
                               HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
}

/* Filename is deterministic: hex of (indexRelId, start_sid, end_sid, version, kind).
 * Same key → same filename → atomic rename gives natural deduplication. */
static void
key_to_filename(const DpvFetchKey *k, char *out, size_t out_size)
{
    snprintf(out, out_size, "%08x_%010u_%010u_%010u_%d.entry",
             k->indexRelId, k->start_sid, k->end_sid, k->version, (int) k->kind);
}

bool
dpv_queue_enqueue(const DpvFetchEntryHeader *hdr, const void *trailer, Size trailer_size)
{
    char entry_path[MAXPGPATH];
    char tmp_path[MAXPGPATH];
    char dir[MAXPGPATH];
    char fname[64];
    DpvFetchKey key = {
        .indexRelId = hdr->indexRelId, .start_sid = hdr->start_sid,
        .end_sid    = hdr->end_sid,    .version   = hdr->version,
        .kind       = hdr->kind,
    };
    int fd;

    queue_dir_path(dir, sizeof(dir));
    key_to_filename(&key, fname, sizeof(fname));
    snprintf(entry_path, sizeof(entry_path), "%s/%s", dir, fname);
    snprintf(tmp_path,   sizeof(tmp_path),   "%s/.%s.tmp", dir, fname);

    /* Atomic existence check. */
    if (access(entry_path, F_OK) == 0)
        return false;  /* already enqueued */

    fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, pg_file_create_mode);
    if (fd < 0)
        elog(ERROR, "[dpv_queue] open %s: %m", tmp_path);

    if (write(fd, hdr, sizeof(*hdr)) != (ssize_t) sizeof(*hdr) ||
        (trailer_size && write(fd, trailer, trailer_size) != (ssize_t) trailer_size))
        elog(ERROR, "[dpv_queue] write %s: %m", tmp_path);

    pg_fsync(fd);
    close(fd);

    if (rename(tmp_path, entry_path) != 0)
        elog(ERROR, "[dpv_queue] rename %s -> %s: %m", tmp_path, entry_path);

    /* fsync the directory so the rename is durable. */
    fsync_fname(dir, true);

    return true;
}

DpvFetchEntry *
dpv_queue_pop_pending(void)
{
    char dir[MAXPGPATH];
    DIR *d;
    struct dirent *de;
    DpvFetchEntry *out = NULL;

    queue_dir_path(dir, sizeof(dir));
    d = AllocateDir(dir);
    if (!d)
        return NULL;

    while ((de = ReadDir(d, dir)) != NULL)
    {
        char path[MAXPGPATH];
        int  fd;
        struct stat st;
        DpvFetchEntryHeader hdr;
        size_t trailer_size;

        if (strncmp(de->d_name, ".", 1) == 0) continue;
        if (!pg_str_endswith(de->d_name, ".entry")) continue;

        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        fd = open(path, O_RDWR);
        if (fd < 0) continue;
        if (fstat(fd, &st) != 0) { close(fd); continue; }
        if (read(fd, &hdr, sizeof(hdr)) != sizeof(hdr)) { close(fd); continue; }
        if (hdr.status != DPV_FETCH_PENDING) { close(fd); continue; }

        /* Atomically claim by rewriting the header with status = FETCHING. */
        hdr.status = DPV_FETCH_FETCHING;
        if (pwrite(fd, &hdr, sizeof(hdr), 0) != sizeof(hdr))
        {
            close(fd);
            continue;
        }
        pg_fsync(fd);

        trailer_size = st.st_size - sizeof(hdr);
        out = palloc(sizeof(DpvFetchEntry));
        out->hdr = hdr;
        strlcpy(out->filename, de->d_name, sizeof(out->filename));
        out->trailer_size = trailer_size;
        out->trailer = trailer_size ? palloc(trailer_size) : NULL;
        if (trailer_size && read(fd, out->trailer, trailer_size) != (ssize_t) trailer_size)
        {
            close(fd);
            pfree(out->trailer);
            pfree(out);
            out = NULL;
            continue;
        }
        close(fd);
        break;
    }
    FreeDir(d);
    return out;
}

void
dpv_queue_mark(const char *filename, DpvFetchStatus new_status)
{
    char path[MAXPGPATH];
    char dir[MAXPGPATH];
    int  fd;
    DpvFetchEntryHeader hdr;

    queue_dir_path(dir, sizeof(dir));
    snprintf(path, sizeof(path), "%s/%s", dir, filename);

    if (new_status == DPV_FETCH_DONE)
    {
        if (unlink(path) != 0 && errno != ENOENT)
            elog(WARNING, "[dpv_queue] unlink %s: %m", path);
        fsync_fname(dir, true);
        return;
    }

    fd = open(path, O_RDWR);
    if (fd < 0)
    {
        elog(WARNING, "[dpv_queue] open %s for mark: %m", path);
        return;
    }
    if (read(fd, &hdr, sizeof(hdr)) == (ssize_t) sizeof(hdr))
    {
        hdr.status = new_status;
        pwrite(fd, &hdr, sizeof(hdr), 0);
        pg_fsync(fd);
    }
    close(fd);
}

void
dpv_queue_recover_on_startup(void)
{
    /* On startup, FETCHING entries are orphans from a crashed worker. Demote
     * them back to PENDING so they get picked up again. */
    char dir[MAXPGPATH];
    DIR *d;
    struct dirent *de;

    queue_dir_path(dir, sizeof(dir));
    d = AllocateDir(dir);
    if (!d)
        return;

    while ((de = ReadDir(d, dir)) != NULL)
    {
        char path[MAXPGPATH];
        int  fd;
        DpvFetchEntryHeader hdr;

        if (!pg_str_endswith(de->d_name, ".entry")) continue;
        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);

        fd = open(path, O_RDWR);
        if (fd < 0) continue;
        if (read(fd, &hdr, sizeof(hdr)) == (ssize_t) sizeof(hdr) &&
            hdr.status == DPV_FETCH_FETCHING)
        {
            hdr.status = DPV_FETCH_PENDING;
            pwrite(fd, &hdr, sizeof(hdr), 0);
            pg_fsync(fd);
        }
        close(fd);
    }
    FreeDir(d);
    fsync_fname(dir, true);

    /* Also: clean orphan .tmp files. */
    d = AllocateDir(dir);
    while ((de = ReadDir(d, dir)) != NULL)
    {
        char path[MAXPGPATH];
        if (de->d_name[0] != '.' || !pg_str_endswith(de->d_name, ".tmp")) continue;
        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        unlink(path);
    }
    FreeDir(d);
}
```

- [ ] **Step 2: Add to OBJS and build**

Append `src/pending_fetch_queue.o` to `OBJS` in [pgvector/Makefile](../../../pgvector/Makefile). Build:

```bash
cd pgvector && make 2>&1 | tail -20
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add pgvector/src/pending_fetch_queue.h pgvector/src/pending_fetch_queue.c pgvector/Makefile
git commit -m "feat: persistent fetch queue for replication"
```

---

## Phase 3 — Segment lifecycle WAL records

### Task 3.1: Extend `replication_rmgr.h` with the two new record types

**Files:**
- Modify: `pgvector/src/replication_rmgr.h`

- [ ] **Step 1: Add the info bits**

In [replication_rmgr.h](../../../pgvector/src/replication_rmgr.h), under the existing `XLOG_DPV_*` macros, add:

```c
#define XLOG_DPV_SEGMENT_CREATED   0x50
#define XLOG_DPV_SEGMENT_REPLACED  0x60
```

- [ ] **Step 2: Add the record payload structs**

```c
typedef struct {
    Oid       indexRelId;
    SegmentId start_sid;
    SegmentId end_sid;
    uint32    version;
} xl_dpv_segment_created;

/*
 * For Replaced: a header followed by old_count xl_dpv_seg_range records, then
 * optionally an offsets[] trailer. Encoded as variable-length WAL data.
 */
typedef struct {
    SegmentId start_sid;
    SegmentId end_sid;
    uint32    version;
} xl_dpv_seg_range;

typedef struct {
    SegmentId source_sid;
    uint32    start_offset;
} xl_dpv_seg_offset;

typedef struct {
    Oid       indexRelId;
    SegmentId new_start_sid;
    SegmentId new_end_sid;
    uint32    new_version;
    uint16    old_count;       /* number of xl_dpv_seg_range entries to follow */
    uint16    offset_count;    /* number of xl_dpv_seg_offset entries after old[] */
    /* followed by: xl_dpv_seg_range old[old_count]
     *              xl_dpv_seg_offset offsets[offset_count] */
} xl_dpv_segment_replaced;
```

- [ ] **Step 3: Add emit prototypes**

```c
extern XLogRecPtr dpv_emit_segment_created(Oid indexRelId,
                                            SegmentId start_sid, SegmentId end_sid,
                                            uint32 version);

extern XLogRecPtr dpv_emit_segment_replaced(Oid indexRelId,
                                             const xl_dpv_seg_range *old_ranges, int old_count,
                                             SegmentId new_start_sid, SegmentId new_end_sid,
                                             uint32 new_version,
                                             const xl_dpv_seg_offset *offsets, int offset_count);
```

### Task 3.2: Implement emit + redo in `replication_rmgr.c`

**Files:**
- Modify: `pgvector/src/replication_rmgr.c`

- [ ] **Step 1: Add the includes**

```c
#include "pending_fetch_queue.h"
```

- [ ] **Step 2: Add to the redo dispatcher**

In `dpv_replication_redo`, add:

```c
case XLOG_DPV_SEGMENT_CREATED:  redo_segment_created(record);  break;
case XLOG_DPV_SEGMENT_REPLACED: redo_segment_replaced(record); break;
```

Add corresponding entries to `dpv_replication_identify`:

```c
case XLOG_DPV_SEGMENT_CREATED:  return "SEGMENT_CREATED";
case XLOG_DPV_SEGMENT_REPLACED: return "SEGMENT_REPLACED";
```

- [ ] **Step 3: Emit helpers**

```c
XLogRecPtr
dpv_emit_segment_created(Oid indexRelId, SegmentId start_sid, SegmentId end_sid,
                         uint32 version)
{
    xl_dpv_segment_created xlrec = {
        .indexRelId = indexRelId, .start_sid = start_sid,
        .end_sid    = end_sid,    .version   = version,
    };
    XLogRecPtr lsn;

    XLogBeginInsert();
    XLogRegisterData((char *) &xlrec, sizeof(xlrec));
    lsn = XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_SEGMENT_CREATED);
    return lsn;
}

XLogRecPtr
dpv_emit_segment_replaced(Oid indexRelId,
                          const xl_dpv_seg_range *old_ranges, int old_count,
                          SegmentId new_start_sid, SegmentId new_end_sid,
                          uint32 new_version,
                          const xl_dpv_seg_offset *offsets, int offset_count)
{
    xl_dpv_segment_replaced hdr = {
        .indexRelId    = indexRelId,
        .new_start_sid = new_start_sid,
        .new_end_sid   = new_end_sid,
        .new_version   = new_version,
        .old_count     = (uint16) old_count,
        .offset_count  = (uint16) offset_count,
    };
    XLogRecPtr lsn;

    XLogBeginInsert();
    XLogRegisterData((char *) &hdr, sizeof(hdr));
    if (old_count > 0)
        XLogRegisterData((char *) old_ranges, old_count * sizeof(xl_dpv_seg_range));
    if (offset_count > 0)
        XLogRegisterData((char *) offsets, offset_count * sizeof(xl_dpv_seg_offset));
    lsn = XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_SEGMENT_REPLACED);
    return lsn;
}
```

These records register no buffer reference — they don't modify status pages. Their purpose is to trigger a fetch on the standby.

- [ ] **Step 4: Redo callbacks (standby: enqueue; primary recovery: skip)**

```c
static void
redo_segment_created(XLogReaderState *r)
{
    xl_dpv_segment_created *rec = (xl_dpv_segment_created *) XLogRecGetData(r);
    DpvFetchEntryHeader hdr;

    if (!InHotStandby)
        return;  /* primary crash recovery: load_lsm_index_internal handles it */

    hdr = (DpvFetchEntryHeader) {
        .indexRelId = rec->indexRelId, .start_sid = rec->start_sid,
        .end_sid    = rec->end_sid,    .version   = rec->version,
        .kind       = DPV_FETCH_KIND_CREATED,
        .source_lsn = r->ReadRecPtr,
        .status     = DPV_FETCH_PENDING,
    };
    (void) dpv_queue_enqueue(&hdr, NULL, 0);
}

static void
redo_segment_replaced(XLogReaderState *r)
{
    char *data = XLogRecGetData(r);
    xl_dpv_segment_replaced *hdr = (xl_dpv_segment_replaced *) data;
    DpvFetchEntryHeader qhdr;
    Size trailer_size;

    if (!InHotStandby)
        return;

    /* Trailer = old_ranges[] + offsets[]. Pack as-is on disk. */
    trailer_size = hdr->old_count * sizeof(xl_dpv_seg_range)
                 + hdr->offset_count * sizeof(xl_dpv_seg_offset);

    qhdr = (DpvFetchEntryHeader) {
        .indexRelId = hdr->indexRelId,
        .start_sid  = hdr->new_start_sid,
        .end_sid    = hdr->new_end_sid,
        .version    = hdr->new_version,
        .kind       = DPV_FETCH_KIND_REPLACED,
        .source_lsn = r->ReadRecPtr,
        .status     = DPV_FETCH_PENDING,
    };
    (void) dpv_queue_enqueue(&qhdr, data + sizeof(*hdr), trailer_size);
}
```

- [ ] **Step 5: Build**

```bash
cd pgvector && make 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add pgvector/src/replication_rmgr.h pgvector/src/replication_rmgr.c
git commit -m "feat: SegmentCreated/SegmentReplaced WAL records"
```

---

## Phase 4 — Emit segment lifecycle WAL on the primary

### Task 4.1: Emit `SegmentCreated` after flush

**Files:**
- Modify: `pgvector/src/lsmbackground.c`

- [ ] **Step 1: Find the caller of `flush_segment_to_disk`**

In [lsmbackground.c](../../../pgvector/src/lsmbackground.c), find where `flush_segment_to_disk(Oid, PrepareFlushMeta)` is called. It's in `lsm_index_bgworker_main` or a helper. The `prep` struct carries `start_sid`, `end_sid`, and the flush worker can read back the chosen version from `find_latest_segment_version` (or have it returned via the prep). The spec calls for emitting *after* the rename in `flush_segment_to_disk`. Two options:

(a) Make `flush_segment_to_disk` return the new version, and emit in the caller.
(b) Emit inside `flush_segment_to_disk` itself, right after `write_lsm_segment_metadata` (the last step, line [lsmindex_io.c:757](../../../pgvector/src/lsmindex_io.c#L757)).

Option (b) keeps the WAL emit co-located with the file write — chosen.

- [ ] **Step 2: Modify `flush_segment_to_disk`**

In [lsmindex_io.c:716-760](../../../pgvector/src/lsmindex_io.c#L716-L760), at the end (after `elog(DEBUG1, "[flush_segment_to_disk] Successfully wrote segment …`), add:

```c
#include "replication_gucs.h"
#include "replication_rmgr.h"
...
    /* Replication: announce the new segment to standbys. Spec §11 inverse
     * direction — file rename is finalized; now WAL says it exists. */
    if (dpv_replication_role == DPV_ROLE_PRIMARY)
    {
        dpv_emit_segment_created(indexRelId, prep->start_sid, prep->end_sid,
                                  new_version);
    }
```

- [ ] **Step 3: Commit**

```bash
git add pgvector/src/lsmindex_io.c
git commit -m "feat: emit SegmentCreated WAL after flush"
```

### Task 4.2: Emit `SegmentReplaced` after merge

**Files:**
- Modify: `pgvector/src/vector_index_worker.c`

- [ ] **Step 1: Find the merge file-write site**

In [vector_index_worker.c:1085](../../../pgvector/src/vector_index_worker.c#L1085), `merge_adjacent_segments_pool` writes a merged segment via `flush_segment_to_disk` (or analogous). Step through to the line where the metadata rename completes for the new merged segment, then emit `SegmentReplaced` after that.

- [ ] **Step 2: Build the `xl_dpv_seg_range[]` and `xl_dpv_seg_offset[]` arrays**

The merge knows: (a) the input segments' sid ranges and versions, (b) the new segment's range and version, (c) the offsets of each source in the merged layout (this is what the merge code already computes — see [vector_index_worker.c:1252-1256](../../../pgvector/src/vector_index_worker.c#L1252-L1256)).

```c
xl_dpv_seg_range  old_ranges[MERGE_MAX_INPUTS];
xl_dpv_seg_offset offsets   [MERGE_MAX_INPUTS];
int n_inputs = /* known to merge_adjacent_segments_pool */;
int n_offsets = n_inputs;

for (int i = 0; i < n_inputs; i++) {
    old_ranges[i] = (xl_dpv_seg_range) {
        .start_sid = inputs[i].start_sid,
        .end_sid   = inputs[i].end_sid,
        .version   = inputs[i].version,
    };
    offsets[i] = (xl_dpv_seg_offset) {
        .source_sid   = inputs[i].start_sid,
        .start_offset = inputs[i].merge_start_offset,
    };
}

if (dpv_replication_role == DPV_ROLE_PRIMARY) {
    dpv_emit_segment_replaced(indexRelId,
                              old_ranges, n_inputs,
                              new_start_sid, new_end_sid, new_version,
                              offsets, n_offsets);
}
```

`MERGE_MAX_INPUTS` and `merge_start_offset` names are placeholders — use whatever the existing merge code names them. Read `merge_adjacent_segments_pool` to identify the exact variables.

- [ ] **Step 3: Commit**

```bash
git add pgvector/src/vector_index_worker.c
git commit -m "feat: emit SegmentReplaced WAL after merge"
```

### Task 4.3: Rebuild also emits `SegmentReplaced` (same range, new version)

**Files:**
- Modify: `pgvector/src/vector_index_worker.c` (rebuild code path; search for `REBUILD` in the merge thread)

- [ ] **Step 1: At the rebuild output site, emit `SegmentReplaced` with old_count=1**

```c
xl_dpv_seg_range old = { .start_sid = sid, .end_sid = sid, .version = old_version };
if (dpv_replication_role == DPV_ROLE_PRIMARY) {
    dpv_emit_segment_replaced(indexRelId,
                              &old, 1,
                              sid, sid, new_version,
                              NULL, 0);
}
```

- [ ] **Step 2: Commit**

```bash
git add pgvector/src/vector_index_worker.c
git commit -m "feat: emit SegmentReplaced WAL after rebuild"
```

---

## Phase 5 — File server bgworker (primary)

### Task 5.1: Define the wire protocol

**Files:**
- Create: `pgvector/src/replication_server.h`

- [ ] **Step 1: Header**

```c
#ifndef DPV_REPLICATION_SERVER_H
#define DPV_REPLICATION_SERVER_H

#include "postgres.h"

/* Wire protocol (binary, network byte order on the wire; helpers convert):
 *
 *   client → server:
 *     [4]  protocol version (currently 1)
 *     [N]  shared secret bytes (length-prefixed string)
 *     [4]  request kind (1 = GET_FILE)
 *     [4]  index_rel_id
 *     [4]  start_sid
 *     [4]  end_sid
 *     [4]  version
 *     [1]  file_kind enum (1=index 2=mapping 3=offset 4=bitmap 5=metadata)
 *
 *   server → client:
 *     [1]  result code (0=ok, 1=auth_fail, 2=not_found, 3=io_error)
 *     [8]  file size in bytes (if ok)
 *     [...]  raw file bytes
 *
 * One request per TCP connection (no pipelining in v1).
 */

typedef enum {
    DPV_REQ_GET_FILE = 1
} DpvReqKind;

typedef enum {
    DPV_FILE_INDEX    = 1,
    DPV_FILE_MAPPING  = 2,
    DPV_FILE_OFFSET   = 3,
    DPV_FILE_BITMAP   = 4,
    DPV_FILE_METADATA = 5,
} DpvFileKind;

typedef enum {
    DPV_RESULT_OK         = 0,
    DPV_RESULT_AUTH_FAIL  = 1,
    DPV_RESULT_NOT_FOUND  = 2,
    DPV_RESULT_IO_ERROR   = 3,
} DpvResultCode;

/* bgworker entry point. */
extern void replication_server_main(Datum main_arg) pg_attribute_noreturn();

#endif
```

### Task 5.2: Implement the server bgworker

**Files:**
- Create: `pgvector/src/replication_server.c`

Use the same socket idioms PG itself uses for streaming replication: `socket(2)`, `bind(2)`, `listen(2)`, `accept(2)` directly. Latch-aware via `WaitLatchOrSocket`.

- [ ] **Step 1: Skeleton**

```c
#include "postgres.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "postmaster/bgworker.h"
#include "postmaster/interrupt.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "tcop/tcopprot.h"
#include "utils/elog.h"

#include "lsmindex.h"               /* VECTOR_STORAGE_BASE_DIR */
#include "lsmindex_io.h"            /* GetLSM*FilePathWithVersion helpers */
#include "replication_gucs.h"
#include "replication_server.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

static volatile sig_atomic_t got_sighup = false;
static volatile sig_atomic_t got_sigterm = false;

static void sigterm_handler(SIGNAL_ARGS) { got_sigterm = true; SetLatch(MyLatch); }
static void sighup_handler(SIGNAL_ARGS)  { got_sighup  = true; SetLatch(MyLatch); }

static int
open_listen_socket(int port)
{
    int s = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    int opt = 1;
    struct sockaddr_in addr = { 0 };

    if (s < 0)
        ereport(FATAL, errmsg("dpv: socket(): %m"));
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (bind(s, (struct sockaddr *) &addr, sizeof(addr)) < 0)
        ereport(FATAL, errmsg("dpv: bind(:%d): %m", port));
    if (listen(s, 16) < 0)
        ereport(FATAL, errmsg("dpv: listen(): %m"));
    return s;
}

static bool
read_n(int fd, void *buf, size_t n)
{
    size_t got = 0;
    while (got < n)
    {
        ssize_t k = read(fd, (char *) buf + got, n - got);
        if (k == 0) return false;
        if (k < 0)
        {
            if (errno == EINTR) continue;
            return false;
        }
        got += k;
    }
    return true;
}

static bool
write_n(int fd, const void *buf, size_t n)
{
    size_t sent = 0;
    while (sent < n)
    {
        ssize_t k = write(fd, (const char *) buf + sent, n - sent);
        if (k < 0)
        {
            if (errno == EINTR) continue;
            return false;
        }
        sent += k;
    }
    return true;
}

static void
build_file_path(char *out, size_t out_size,
                Oid indexRelId, uint32 start, uint32 end, uint32 version,
                DpvFileKind kind)
{
    switch (kind) {
        case DPV_FILE_INDEX:
            GetLSMIndexFilePathWithVersion(out, out_size, indexRelId, start, end, version);
            break;
        case DPV_FILE_MAPPING:
            GetLSMMappingFilePathWithVersion(out, out_size, indexRelId, start, end, version);
            break;
        case DPV_FILE_OFFSET:
            GetLSMOffsetFilePathWithVersion(out, out_size, indexRelId, start, end, version);
            break;
        case DPV_FILE_BITMAP:
            GetLSMBitmapFilePathWithVersion(out, out_size, indexRelId, start, end, version);
            break;
        case DPV_FILE_METADATA:
            GetLSMMetadataFilePathWithVersion(out, out_size, indexRelId, start, end, version);
            break;
        default:
            out[0] = '\0';
    }
}

static void
handle_one_connection(int conn)
{
    uint32 proto_ver, secret_len;
    char   secret[256];
    uint32 req_kind, idx, start, end, version;
    uint8  fkind;
    char   path[MAXPGPATH];
    int    fd;
    uint64 size;
    uint8  result;

    if (!read_n(conn, &proto_ver, 4)) goto out;
    proto_ver = pg_ntoh32(proto_ver);
    if (proto_ver != 1) goto out;

    if (!read_n(conn, &secret_len, 4)) goto out;
    secret_len = pg_ntoh32(secret_len);
    if (secret_len >= sizeof(secret)) goto out;
    if (!read_n(conn, secret, secret_len)) goto out;
    secret[secret_len] = '\0';

    if (strcmp(secret, dpv_replication_shared_secret) != 0)
    {
        result = DPV_RESULT_AUTH_FAIL;
        write_n(conn, &result, 1);
        goto out;
    }

    if (!read_n(conn, &req_kind, 4)) goto out;
    if (!read_n(conn, &idx,      4)) goto out;
    if (!read_n(conn, &start,    4)) goto out;
    if (!read_n(conn, &end,      4)) goto out;
    if (!read_n(conn, &version,  4)) goto out;
    if (!read_n(conn, &fkind,    1)) goto out;

    req_kind = pg_ntoh32(req_kind);
    idx      = pg_ntoh32(idx);
    start    = pg_ntoh32(start);
    end      = pg_ntoh32(end);
    version  = pg_ntoh32(version);
    if (req_kind != DPV_REQ_GET_FILE) goto out;

    build_file_path(path, sizeof(path), idx, start, end, version, fkind);
    fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        result = DPV_RESULT_NOT_FOUND;
        write_n(conn, &result, 1);
        goto out;
    }

    /* stat for size */
    {
        struct stat st;
        if (fstat(fd, &st) != 0)
        {
            close(fd);
            result = DPV_RESULT_IO_ERROR;
            write_n(conn, &result, 1);
            goto out;
        }
        size = (uint64) st.st_size;
    }

    result = DPV_RESULT_OK;
    write_n(conn, &result, 1);
    {
        uint64 sz_be = pg_hton64(size);
        write_n(conn, &sz_be, 8);
    }

    /* Stream the file in 1 MiB chunks. */
    {
        char buf[1 << 20];
        ssize_t k;
        while ((k = read(fd, buf, sizeof(buf))) > 0)
        {
            if (!write_n(conn, buf, (size_t) k)) break;
        }
    }
    close(fd);

out:
    close(conn);
}

PG_FUNCTION_INFO_V1(replication_server_main);
void
replication_server_main(Datum main_arg)
{
    int listen_fd;
    pqsignal(SIGTERM, sigterm_handler);
    pqsignal(SIGHUP,  sighup_handler);
    BackgroundWorkerUnblockSignals();

    if (dpv_replication_role != DPV_ROLE_PRIMARY)
        proc_exit(0);
    if (dpv_replication_primary_port <= 0)
    {
        elog(LOG, "dpv replication_server: no port configured, exiting");
        proc_exit(0);
    }

    listen_fd = open_listen_socket(dpv_replication_primary_port);

    while (!got_sigterm)
    {
        int rc;
        rc = WaitLatchOrSocket(MyLatch,
                               WL_LATCH_SET | WL_SOCKET_READABLE | WL_EXIT_ON_PM_DEATH,
                               listen_fd, -1, WAIT_EVENT_EXTENSION);
        ResetLatch(MyLatch);

        if (rc & WL_SOCKET_READABLE)
        {
            int conn = accept(listen_fd, NULL, NULL);
            if (conn >= 0)
                handle_one_connection(conn);   /* synchronous; v1 single-thread */
        }
    }
    close(listen_fd);
    proc_exit(0);
}
```

- [ ] **Step 2: Register the bgworker from `_PG_init`** in [vector.c](../../../pgvector/src/vector.c)

```c
#include "replication_server.h"
...
    BackgroundWorker server_worker;
    memset(&server_worker, 0, sizeof(server_worker));
    server_worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
    server_worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
    server_worker.bgw_restart_time = 5;
    snprintf(server_worker.bgw_name, BGW_MAXLEN, "DpvReplicationServer");
    snprintf(server_worker.bgw_library_name, BGW_MAXLEN, "vector.so");
    snprintf(server_worker.bgw_function_name, BGW_MAXLEN, "replication_server_main");
    RegisterBackgroundWorker(&server_worker);
```

The worker self-terminates immediately on standbys (where `dpv_replication_role != DPV_ROLE_PRIMARY`), so it's always registered but only active on the primary.

- [ ] **Step 3: Add to OBJS, build, commit**

Append `src/replication_server.o` to OBJS. Build.

```bash
cd pgvector && make 2>&1 | tail -10
git add pgvector/src/replication_server.h pgvector/src/replication_server.c \
        pgvector/src/vector.c pgvector/Makefile
git commit -m "feat: replication file-server bgworker (primary)"
```

---

## Phase 6 — Segment fetcher bgworker (standby)

### Task 6.1: Adoption API skeleton

**Files:**
- Create: `pgvector/src/segment_adoption.h`
- Create: `pgvector/src/segment_adoption.c`

- [ ] **Step 1: Header**

```c
#ifndef DPV_SEGMENT_ADOPTION_H
#define DPV_SEGMENT_ADOPTION_H

#include "postgres.h"
#include "lsm_segment.h"

typedef enum {
    DPV_ADOPT_ADOPTED         = 0,
    DPV_ADOPT_STALE_DISCARD   = 1,
    DPV_ADOPT_INDEX_UNLOADED  = 2,
} DpvAdoptionOutcome;

/*
 * Coverage rule (spec §8). Caller has already placed the 5 files on disk under
 * VECTOR_STORAGE_BASE_DIR. This function inspects the pool + memtable buffer
 * and either:
 *  - replaces the covered group with the pulled segment (ADOPTED), or
 *  - leaves state alone and asks caller to unlink the pulled files (STALE),
 *  - skips entirely if the index is not loaded (file persists on disk).
 *
 * For v1 (this plan): bitmap merge is a straight UNION of the pulled bitmap
 * and the local representations' bitmaps. Version-aware translation arrives
 * in Plan 3.
 */
extern DpvAdoptionOutcome dpv_attempt_adoption(Oid indexRelId,
                                                SegmentId start_sid,
                                                SegmentId end_sid,
                                                uint32 version);

#endif
```

- [ ] **Step 2: Body (skeleton — flesh out under Task 6.2 once we know which existing helpers we can call)**

```c
#include "postgres.h"
#include "segment_adoption.h"
#include "lsmindex.h"
#include "lsm_segment.h"

DpvAdoptionOutcome
dpv_attempt_adoption(Oid indexRelId, SegmentId start_sid, SegmentId end_sid,
                     uint32 version)
{
    /* See Task 6.2 */
    return DPV_ADOPT_INDEX_UNLOADED;
}
```

### Task 6.2: Implement the coverage rule

**Files:**
- Modify: `pgvector/src/segment_adoption.c`
- Read: `pgvector/src/lsm_segment.h`, `pgvector/src/lsm_segment.c` (for `FlushedSegmentPool`, `seg_lock`, helpers `find_segment_by_sids`, `register_flushed_segment`, `replace_flushed_segment`)
- Read: `pgvector/src/lsmindex.h`, `pgvector/src/lsmindex.c` (`mt_lock`, memtable buffer)

- [ ] **Step 1: Read the existing helpers**

Locate and read:
- `find_segment_by_sids(Oid indexRelId, SegmentId start_sid, SegmentId end_sid)` in [lsm_segment.c](../../../pgvector/src/lsm_segment.c)
- `register_flushed_segment` (used by primary's flush path; the standby will need an analogous helper that takes a pre-loaded pool entry from disk)
- `replace_flushed_segment(old_seg, new_seg, ...)`
- Iteration over pool entries while holding `seg_lock` read

If `register_flushed_segment` / `replace_flushed_segment` are tightly coupled to the primary's flush/merge code paths and can't be called from the fetcher worker, factor out the lower-level "swap N entries in the pool atomically" core and call that from both places.

- [ ] **Step 2: Implement the coverage rule**

```c
DpvAdoptionOutcome
dpv_attempt_adoption(Oid indexRelId, SegmentId start_sid, SegmentId end_sid,
                     uint32 version)
{
    LSMIndexBufferSlot *slot = lookup_lsm_index_buffer_slot(indexRelId);
    FlushedSegmentPool *pool;
    DpvAdoptionOutcome outcome;

    if (slot == NULL || pg_atomic_read_u32(&slot->valid) != 1)
        return DPV_ADOPT_INDEX_UNLOADED;

    pool = slot->pool;
    LWLockAcquire(&pool->seg_lock, LW_EXCLUSIVE);
    LWLockAcquire(&slot->mt_lock,  LW_EXCLUSIVE);

    outcome = evaluate_and_adopt_locked(slot, pool, indexRelId,
                                        start_sid, end_sid, version);

    LWLockRelease(&slot->mt_lock);
    LWLockRelease(&pool->seg_lock);
    return outcome;
}

/* Implementation of the coverage rule with both locks held. */
static DpvAdoptionOutcome
evaluate_and_adopt_locked(LSMIndexBufferSlot *slot, FlushedSegmentPool *pool,
                          Oid indexRelId,
                          SegmentId start_sid, SegmentId end_sid, uint32 version)
{
    /* Step 1: collect existing pool entries whose ranges overlap [start, end]. */
    FlushedSegment overlap[POOL_MAX_ENTRIES];
    int overlap_n = collect_overlapping_segments(pool, start_sid, end_sid, overlap);

    /* Step 2: collect memtables whose sids overlap [start, end]. */
    ConcurrentMemTable mts[MEMTABLE_BUF_SIZE];
    int mt_n = collect_overlapping_memtables(slot, start_sid, end_sid, mts);

    /* Case A: a single existing segment has the EXACT range. */
    if (overlap_n == 1 && mt_n == 0 &&
        overlap[0]->start_sid == start_sid && overlap[0]->end_sid == end_sid)
    {
        if (version > overlap[0]->version)
        {
            /* Rebuild case: same range, newer version → adopt. */
            load_and_swap_in_segment(pool, overlap[0], start_sid, end_sid, version);
            return DPV_ADOPT_ADOPTED;
        }
        else
        {
            return DPV_ADOPT_STALE_DISCARD;
        }
    }

    /* Case B: one existing segment strictly contains [start, end] —
     * pull is stale (an earlier merge took us past this range). */
    if (overlap_n == 1 && mt_n == 0 &&
        overlap[0]->start_sid <= start_sid && overlap[0]->end_sid >= end_sid &&
        !(overlap[0]->start_sid == start_sid && overlap[0]->end_sid == end_sid))
    {
        return DPV_ADOPT_STALE_DISCARD;
    }

    /* Case C: union of overlap[] ∪ mts[] covers EXACTLY [start, end].
     * Adopt — replace the group with the new segment. */
    if (group_covers_exactly(overlap, overlap_n, mts, mt_n, start_sid, end_sid))
    {
        load_and_swap_in_segment_grouped(pool, slot, overlap, overlap_n,
                                          mts, mt_n,
                                          start_sid, end_sid, version);
        return DPV_ADOPT_ADOPTED;
    }

    /* No coverage: per spec §8 invariant, this is unreachable for a loaded
     * index. Log loudly and discard for v1; future versions may want a
     * structured assertion. */
    elog(WARNING, "[dpv adoption] no coverage for indexRelId=%u range=[%u,%u] v=%u — discarding",
         indexRelId, start_sid, end_sid, version);
    return DPV_ADOPT_STALE_DISCARD;
}
```

The helpers `collect_overlapping_segments`, `collect_overlapping_memtables`, `group_covers_exactly`, `load_and_swap_in_segment`, and `load_and_swap_in_segment_grouped` are static inside `segment_adoption.c`. They wrap existing pool/memtable manipulation. Concretely:

- `load_and_swap_in_segment` calls `load_and_set_segment` ([lsm_segment.c:430](../../../pgvector/src/lsm_segment.c#L430)) to attach the new segment file to a `FlushedSegment` struct, then calls the existing pool-mutation helper to replace `overlap[0]` with the new entry. **Bitmap union**: after the swap, OR the old segment's bitmap into the new segment's bitmap (`for i in 0..N: new->bitmap[i] |= old->bitmap[i]`) — this carries forward any local bitmap mutations not present in the pulled file.
- `load_and_swap_in_segment_grouped` is the multi-input version: union all `overlap[].bitmap` and all `mts[].bitmap` into `new->bitmap` (after translation, which for v1 — same version, same layout — is identity, i.e., the bit at position `i` in the predecessor sits at the same `i` in the new file). Then remove all the inputs from the pool / memtable buffer and insert the new segment.

**Caveat for this plan:** identity translation is *not* universally correct. It works for:
- Plain flush (memtable → segment, same `i`).
- Plain merge (concatenation, requires per-input offset — `offsets[]` from `xl_dpv_segment_replaced`).
- Rebuild **without** compaction.

It is *not* correct when `REBUILD_DELETION` compacts out deleted entries (the new segment's `local_idx` doesn't match the old one's). The §13.3 lazy sorted-permutation aux structure is needed for that case. Plan 3 implements it. For Plan 2, document the limitation:

```c
/*
 * NOTE: for v1 (plan 2), bitmap merge is *identity translation* — it assumes
 * predecessor->new uses the same local_idx layout. This is correct for:
 *   - flush (memtable → segment, same layout)
 *   - merge (predecessors concatenated with known offsets)
 *   - rebuild WITHOUT REBUILD_DELETION (same layout)
 * It is INCORRECT when REBUILD_DELETION compacts the index. Plan 3 adds the
 * lazy sorted-permutation aux structure to handle this case.
 *
 * As a guard for v1: detect "rebuild that may have compacted" by checking
 * whether old->bitmap has any deleted bits set; if so, log loudly and skip
 * the bitmap merge (accept that one bit-flip window may exist; vacuum will
 * re-issue it from WAL once Plan 3 lands).
 */
```

The guard above is a v1 sanity floor; in practice, for Plan 2's tests we avoid hitting that case by not running vacuum-during-rebuild in the test workloads.

### Task 6.3: Fetcher bgworker

**Files:**
- Create: `pgvector/src/segment_fetcher.h`
- Create: `pgvector/src/segment_fetcher.c`

- [ ] **Step 1: Header**

```c
#ifndef DPV_SEGMENT_FETCHER_H
#define DPV_SEGMENT_FETCHER_H

#include "postgres.h"

extern void segment_fetcher_main(Datum main_arg) pg_attribute_noreturn();

#endif
```

- [ ] **Step 2: Body**

```c
#include "postgres.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "postmaster/bgworker.h"
#include "postmaster/interrupt.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "utils/elog.h"

#include "lsmindex.h"
#include "lsmindex_io.h"
#include "pending_fetch_queue.h"
#include "replication_gucs.h"
#include "replication_server.h"
#include "segment_adoption.h"
#include "segment_fetcher.h"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

static volatile sig_atomic_t got_sigterm = false;
static void sigterm_handler(SIGNAL_ARGS) { got_sigterm = true; SetLatch(MyLatch); }

static int
connect_primary(void)
{
    int s;
    struct sockaddr_in addr = { 0 };

    s = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0) return -1;
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(dpv_replication_primary_port);
    if (inet_pton(AF_INET, dpv_replication_primary_host, &addr.sin_addr) != 1)
    {
        close(s);
        return -1;
    }
    if (connect(s, (struct sockaddr *) &addr, sizeof(addr)) < 0)
    {
        close(s);
        return -1;
    }
    return s;
}

static bool
write_n(int fd, const void *buf, size_t n)  /* same as in server */;
static bool
read_n(int fd, void *buf, size_t n)         /* same as in server */;

/* Returns true on success. Writes via temp-then-rename. */
static bool
pull_one_file(int conn, Oid idx, uint32 start, uint32 end, uint32 version,
              DpvFileKind kind, const char *out_path)
{
    uint32 v;
    uint32 secret_len;
    char   tmp_path[MAXPGPATH];
    int    out_fd;
    uint8  result;
    uint64 size;

    v = pg_hton32(1);
    if (!write_n(conn, &v, 4)) return false;

    secret_len = pg_hton32((uint32) strlen(dpv_replication_shared_secret));
    if (!write_n(conn, &secret_len, 4)) return false;
    if (!write_n(conn, dpv_replication_shared_secret,
                 strlen(dpv_replication_shared_secret))) return false;

    v = pg_hton32(DPV_REQ_GET_FILE);  if (!write_n(conn, &v, 4)) return false;
    v = pg_hton32(idx);                if (!write_n(conn, &v, 4)) return false;
    v = pg_hton32(start);              if (!write_n(conn, &v, 4)) return false;
    v = pg_hton32(end);                if (!write_n(conn, &v, 4)) return false;
    v = pg_hton32(version);            if (!write_n(conn, &v, 4)) return false;
    {
        uint8 fk = (uint8) kind;
        if (!write_n(conn, &fk, 1)) return false;
    }

    if (!read_n(conn, &result, 1)) return false;
    if (result != DPV_RESULT_OK)
    {
        elog(WARNING, "[dpv fetcher] server returned %d for %s", result, out_path);
        return false;
    }
    if (!read_n(conn, &size, 8)) return false;
    size = pg_ntoh64(size);

    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", out_path);
    out_fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, pg_file_create_mode);
    if (out_fd < 0) return false;

    {
        char buf[1 << 20];
        uint64 remaining = size;
        while (remaining > 0)
        {
            size_t chunk = remaining < sizeof(buf) ? (size_t) remaining : sizeof(buf);
            if (!read_n(conn, buf, chunk)) { close(out_fd); unlink(tmp_path); return false; }
            if (write(out_fd, buf, chunk) != (ssize_t) chunk) { close(out_fd); unlink(tmp_path); return false; }
            remaining -= chunk;
        }
    }
    pg_fsync(out_fd);
    close(out_fd);

    if (rename(tmp_path, out_path) != 0)
    {
        unlink(tmp_path);
        return false;
    }
    return true;
}

/* Pulls index/mapping/offset/bitmap/metadata (in that order — metadata LAST). */
static bool
pull_all_files(Oid idx, uint32 start, uint32 end, uint32 version)
{
    int conn;
    char path[MAXPGPATH];
    DpvFileKind kinds[] = {
        DPV_FILE_INDEX, DPV_FILE_MAPPING, DPV_FILE_OFFSET,
        DPV_FILE_BITMAP, DPV_FILE_METADATA,
    };

    /* Idempotence: if metadata already exists (from rsync or a prior run),
     * assume the segment is complete and skip the pull. */
    GetLSMMetadataFilePathWithVersion(path, sizeof(path), idx, start, end, version);
    if (access(path, F_OK) == 0)
        return true;

    for (size_t i = 0; i < lengthof(kinds); i++)
    {
        bool ok;
        switch (kinds[i]) {
            case DPV_FILE_INDEX:    GetLSMIndexFilePathWithVersion(path, sizeof(path), idx, start, end, version); break;
            case DPV_FILE_MAPPING:  GetLSMMappingFilePathWithVersion(path, sizeof(path), idx, start, end, version); break;
            case DPV_FILE_OFFSET:   GetLSMOffsetFilePathWithVersion(path, sizeof(path), idx, start, end, version); break;
            case DPV_FILE_BITMAP:   GetLSMBitmapFilePathWithVersion(path, sizeof(path), idx, start, end, version); break;
            case DPV_FILE_METADATA: GetLSMMetadataFilePathWithVersion(path, sizeof(path), idx, start, end, version); break;
        }
        conn = connect_primary();
        if (conn < 0) return false;
        ok = pull_one_file(conn, idx, start, end, version, kinds[i], path);
        close(conn);
        if (!ok) return false;
    }
    return true;
}

PG_FUNCTION_INFO_V1(segment_fetcher_main);
void
segment_fetcher_main(Datum main_arg)
{
    pqsignal(SIGTERM, sigterm_handler);
    BackgroundWorkerUnblockSignals();

    if (dpv_replication_role != DPV_ROLE_STANDBY)
        proc_exit(0);

    dpv_queue_init();
    dpv_queue_recover_on_startup();

    while (!got_sigterm)
    {
        DpvFetchEntry *e = dpv_queue_pop_pending();
        if (e == NULL)
        {
            (void) WaitLatch(MyLatch, WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                             1000, WAIT_EVENT_EXTENSION);
            ResetLatch(MyLatch);
            continue;
        }

        bool fetched = pull_all_files(e->hdr.indexRelId, e->hdr.start_sid,
                                       e->hdr.end_sid, e->hdr.version);
        if (!fetched)
        {
            dpv_queue_mark(e->filename, DPV_FETCH_FAILED);
        }
        else
        {
            (void) dpv_attempt_adoption(e->hdr.indexRelId, e->hdr.start_sid,
                                         e->hdr.end_sid, e->hdr.version);
            dpv_queue_mark(e->filename, DPV_FETCH_DONE);
        }

        if (e->trailer) pfree(e->trailer);
        pfree(e);
    }
    proc_exit(0);
}
```

- [ ] **Step 3: Register N fetcher workers**

In [vector.c](../../../pgvector/src/vector.c)'s `_PG_init`, after the server-worker registration:

```c
#include "segment_fetcher.h"
...
    for (int i = 0; i < dpv_replication_fetch_parallelism; i++)
    {
        BackgroundWorker fw;
        memset(&fw, 0, sizeof(fw));
        fw.bgw_flags = BGWORKER_SHMEM_ACCESS;
        fw.bgw_start_time = BgWorkerStart_RecoveryFinished;
        fw.bgw_restart_time = 5;
        snprintf(fw.bgw_name, BGW_MAXLEN, "DpvSegmentFetcher%d", i);
        snprintf(fw.bgw_library_name, BGW_MAXLEN, "vector.so");
        snprintf(fw.bgw_function_name, BGW_MAXLEN, "segment_fetcher_main");
        fw.bgw_main_arg = Int32GetDatum(i);
        RegisterBackgroundWorker(&fw);
    }
```

Note: `dpv_replication_fetch_parallelism` is a GUC read at `_PG_init`. Because the GUC defaults to 2 and is `PGC_POSTMASTER`, the value will be authoritative by the time `_PG_init` runs (GUC processing happens before `shared_preload_libraries` `_PG_init`s in PG).

- [ ] **Step 4: Add to OBJS, build, commit**

```bash
cd pgvector && make 2>&1 | tail -10
git add pgvector/src/segment_adoption.h pgvector/src/segment_adoption.c \
        pgvector/src/segment_fetcher.h pgvector/src/segment_fetcher.c \
        pgvector/src/vector.c pgvector/Makefile
git commit -m "feat: segment fetcher bgworker + adoption (identity translation)"
```

---

## Phase 7 — Integration tests

The test harness uses a few extra setup steps compared to Plan 1:

1. Both primary and standby need `pgvector.replication_role` set.
2. The primary needs `pgvector.replication_primary_port` set to a free port.
3. The standby needs the same port and `pgvector.replication_primary_host = '127.0.0.1'`.
4. Both need `pgvector.replication_shared_secret` set to the same value.
5. **Storage dir is per-cluster** (Plan 1 converted `VECTOR_STORAGE_BASE_DIR` to the GUC `pgvector.storage_base_dir`, which defaults to `<DataDir>/pgvector_storage/`). Each cluster gets its own storage root automatically — no rsync required for the test harness. (Spec §2 non-goal #3 still applies to production deployments across hosts, where an initial rsync is the operator's responsibility.)

### Task 7.1: Common test helper

**Files:**
- Create: `pgvector/test/perl/DpvReplication.pm`

- [ ] **Step 1: Helper module**

```perl
package DpvReplication;
use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use File::Path qw(rmtree make_path);
use Exporter 'import';
our @EXPORT_OK = qw(setup_primary setup_standby wait_catchup);

my $port_counter = 18000;

sub _next_replication_port { return $port_counter++; }

sub setup_primary {
    my %args = @_;
    my $node = PostgreSQL::Test::Cluster->new('primary');
    $node->init(allows_streaming => 1);
    my $port = _next_replication_port();
    my $secret = "dpv_test_secret";
    $node->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
wal_level = replica
max_wal_senders = 4
wal_keep_size = 256MB
pgvector.replication_role = 'primary'
pgvector.replication_primary_port = $port
pgvector.replication_shared_secret = '$secret'
pgvector.replication_fetch_parallelism = 2
));
    $node->start;
    $node->{_dpv_port}   = $port;
    $node->{_dpv_secret} = $secret;
    return $node;
}

sub setup_standby {
    my ($primary) = @_;
    $primary->backup('basebackup');
    my $node = PostgreSQL::Test::Cluster->new('standby');
    $node->init_from_backup($primary, 'basebackup', has_streaming => 1);
    my $port   = $primary->{_dpv_port};
    my $secret = $primary->{_dpv_secret};
    $node->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
hot_standby = on
pgvector.replication_role = 'standby'
pgvector.replication_primary_host = '127.0.0.1'
pgvector.replication_primary_port = $port
pgvector.replication_shared_secret = '$secret'
pgvector.replication_fetch_parallelism = 2
));

    # Plan 1 converted VECTOR_STORAGE_BASE_DIR to the GUC pgvector.storage_base_dir,
    # which defaults to <DataDir>/pgvector_storage/. Each cluster therefore has its
    # own storage root; no explicit override is needed in the harness. _PG_init
    # creates the directory at startup.

    $node->start;
    return $node;
}

sub wait_catchup {
    my ($primary, $standby) = @_;
    my $appname = $standby->name;
    my $q = "SELECT pg_current_wal_lsn() <= replay_lsn "
          . "FROM pg_stat_replication WHERE application_name = '$appname';";
    $primary->poll_query_until('postgres', $q) or die "standby never caught up";
}

1;
```

Storage isolation between primary and standby is already correct: Plan 1 converted `VECTOR_STORAGE_BASE_DIR` to the GUC `pgvector.storage_base_dir` (default `<DataDir>/pgvector_storage/`). Each cluster has its own physical storage root, so the side channel is genuinely exercised — the standby's queue receives entries via WAL redo, the fetcher pulls files from the primary's server, and adoption runs against the standby's own pool. The "metadata already on disk" idempotence fast-path only fires on standby restart / restarted fetcher (when a prior pull completed before the crash).

### Task 7.2: Convert `VECTOR_STORAGE_BASE_DIR` to a GUC — **DONE in Plan 1**

> **Status: completed during Plan 1 implementation (2026-05-15).**
>
> This task was pulled forward into Plan 1 because the previous hardcoded absolute path masked Plan 1's segment-shipment gap in tests (shared filesystem on a single machine → standby silently saw the primary's segments). See Plan 1's Implementation Notes, section "Storage layout: `VECTOR_STORAGE_BASE_DIR` → `pgvector.storage_base_dir` GUC (resolved under `<DataDir>` by default)" for the full record.
>
> Summary of what landed in Plan 1:
>
> - `#define VECTOR_STORAGE_BASE_DIR ...` removed from `pgvector/src/lsmindex.h`.
> - GUC `pgvector.storage_base_dir` (string, `PGC_POSTMASTER`) registered in `_PG_init`. Default empty → resolves to `<DataDir>/pgvector_storage/`. Non-empty user values are used with auto-appended trailing slash.
> - Accessor `get_vector_storage_dir(void)` defined in `pgvector/src/lsmindex.c`. All 9 call sites in `pgvector/src/lsmindex_io.c` and `pgvector/src/hnswbuild.c` were converted from `VECTOR_STORAGE_BASE_DIR "%u/…"` string concatenation to `"%s%u/…"` format + `get_vector_storage_dir()` argument.
> - `_PG_init` creates the base directory via `MakePGDirectory`.
>
> Test harness no longer needs the per-cluster mkdir step shown in the original draft of Task 7.2 — `_PG_init` creates the directory automatically using each cluster's own `DataDir`. The `DpvReplication.pm` helper can omit the explicit GUC override entirely; the default behaviour is correct.
>
> Plan 2 work can proceed assuming `get_vector_storage_dir()` is available and each cluster has its own storage root.

### Task 7.3: Test — flush on primary, segment arrives on standby

**Files:**
- Create: `pgvector/test/t/110_replication_segment_flush.pl`

- [ ] **Step 1: Write the test**

```perl
use strict;
use warnings FATAL => 'all';
use lib 'test/perl';
use DpvReplication qw(setup_primary setup_standby wait_catchup);
use Test::More;

my $primary = setup_primary();
my $standby = setup_standby($primary);

my $dim = 32;
my $array_sql = join(",", ('random()') x $dim);

$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');
$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    CREATE INDEX t_v_idx ON t USING <YOUR_LSM_AM_NAME> (v vector_l2_ops);
));

# Insert enough rows to force at least one memtable flush.
# MEMTABLE_BUF_SIZE * MEMTABLE_CAPACITY rows (substitute the actual constants
# from lsmindex.h; the test must exceed one full memtable's capacity).
$primary->safe_psql('postgres',
    "INSERT INTO t SELECT i, ARRAY[$array_sql]::vector FROM generate_series(1, 50000) i;");
$primary->safe_psql('postgres', "SELECT pg_sleep(2);");  # allow flush bgworker to run
wait_catchup($primary, $standby);

# Wait for the standby's fetcher to consume the queue.
for (my $i = 0; $i < 30; $i++) {
    last if $standby->safe_psql('postgres',
        "SELECT count(*) FROM pg_ls_dir('pgvector_storage/_pending_fetches')") eq '0';
    sleep 1;
}

# The standby's adoption replaces memtables with the new segment.
# Query against the index — must still return all rows.
my $count = $standby->safe_psql('postgres', q(
    SET enable_seqscan = off;
    SELECT count(*) FROM (
        SELECT * FROM t
        ORDER BY v <-> '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector
        LIMIT 50000
    ) sub;
));
is($count, '50000', "standby returns all rows after primary flush+standby adopt");

done_testing();
```

If pg_ls_dir on the queue directory fails (path permissions), substitute a Perl `opendir/closedir` check.

- [ ] **Step 2: Run, commit**

```bash
prove -I test/perl test/t/110_replication_segment_flush.pl
git add pgvector/test/t/110_replication_segment_flush.pl
git commit -m "test: segment flush replicates and adopts on standby"
```

### Task 7.4: Test — merge replicates

**Files:**
- Create: `pgvector/test/t/111_replication_segment_merge.pl`

- [ ] **Step 1: Test design**

Drive enough inserts that two adjacent flushed segments exist on the primary, then trigger merge (the merge worker should run automatically; if not, document how to force it via a SQL function in this fork). After catchup, verify the standby's pool has a single merged segment of the union range.

Probe approach: expose a debugging SQL function `pgvector_pool_dump(indexRelId)` returning `(start_sid, end_sid, version)` rows from the standby's pool — if it doesn't exist, add it as a helper in `vector.c` for tests.

```perl
use strict;
use warnings FATAL => 'all';
use lib 'test/perl';
use DpvReplication qw(setup_primary setup_standby wait_catchup);
use Test::More;

my $primary = setup_primary();
my $standby = setup_standby($primary);

my $dim = 32;
my $array_sql = join(",", ('random()') x $dim);

$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');
$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    CREATE INDEX t_v_idx ON t USING <YOUR_LSM_AM_NAME> (v vector_l2_ops);
));

# Force at least two flushes.
$primary->safe_psql('postgres',
    "INSERT INTO t SELECT i, ARRAY[$array_sql]::vector FROM generate_series(1, 100000) i;");
$primary->safe_psql('postgres', "SELECT pg_sleep(3);");

# Force a merge (replace with the real trigger function or SQL helper):
$primary->safe_psql('postgres', "SELECT pgvector_force_merge('t_v_idx'::regclass);");
$primary->safe_psql('postgres', "SELECT pg_sleep(3);");
wait_catchup($primary, $standby);

# Wait for the fetcher to drain.
for (my $i = 0; $i < 30; $i++) {
    last if $standby->safe_psql('postgres',
        "SELECT count(*) FROM pg_ls_dir('pgvector_storage/_pending_fetches')") eq '0';
    sleep 1;
}

# Query — must still return all rows.
my $count = $standby->safe_psql('postgres', q(
    SET enable_seqscan = off;
    SELECT count(*) FROM (
        SELECT * FROM t
        ORDER BY v <-> '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector
        LIMIT 100000
    ) sub;
));
is($count, '100000', "standby returns all rows after primary merges segments");

done_testing();
```

`pgvector_force_merge('t_v_idx')` — add this as a SQL-callable helper in `vector.c` for tests. It triggers a merge of adjacent segments synchronously. (Search for existing `pgvector_*` test helpers and follow the pattern.) If a debug-only function feels invasive, alternative: poll until merge happens naturally based on `merge_threshold` config.

- [ ] **Step 2: Run, commit**

```bash
prove -I test/perl test/t/111_replication_segment_merge.pl
git add pgvector/test/t/111_replication_segment_merge.pl pgvector/src/vector.c
git commit -m "test: segment merge replicates and adopts on standby"
```

### Task 7.5: Test — queue durability across standby restart

**Files:**
- Create: `pgvector/test/t/112_replication_queue_restart.pl`

- [ ] **Step 1: Test**

```perl
use strict;
use warnings FATAL => 'all';
use lib 'test/perl';
use DpvReplication qw(setup_primary setup_standby wait_catchup);
use Test::More;

my $primary = setup_primary();
# Block the file server before bringing up the standby, so the standby
# enqueues fetches that can't complete.
# Easiest: don't start the primary's replication_server worker until later.
# v1 has no toggle; instead, stop the standby mid-fetch:

my $standby = setup_standby($primary);

# Drive a flush on the primary.
my $dim = 32;
my $array_sql = join(",", ('random()') x $dim);
$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');
$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    CREATE INDEX t_v_idx ON t USING <YOUR_LSM_AM_NAME> (v vector_l2_ops);
));
$primary->safe_psql('postgres',
    "INSERT INTO t SELECT i, ARRAY[$array_sql]::vector FROM generate_series(1, 50000) i;");
$primary->safe_psql('postgres', "SELECT pg_sleep(1);");
wait_catchup($primary, $standby);

# Immediately stop the standby (likely interrupts mid-fetch).
$standby->stop('immediate');

# Restart and verify the queue still has the pending entry, then drains.
$standby->start;

for (my $i = 0; $i < 30; $i++) {
    last if $standby->safe_psql('postgres',
        "SELECT count(*) FROM pg_ls_dir('pgvector_storage/_pending_fetches')") eq '0';
    sleep 1;
}

my $count = $standby->safe_psql('postgres', q(
    SET enable_seqscan = off;
    SELECT count(*) FROM (SELECT * FROM t ORDER BY v <-> '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector LIMIT 50000) sub;
));
is($count, '50000', "queue replays after standby restart");

done_testing();
```

- [ ] **Step 2: Run, commit**

```bash
prove -I test/perl test/t/112_replication_queue_restart.pl
git add pgvector/test/t/112_replication_queue_restart.pl
git commit -m "test: persistent fetch queue survives standby restart"
```

### Task 7.6: Test — out-of-order pulls handled by coverage rule

**Files:**
- Create: `pgvector/test/t/113_replication_out_of_order_pulls.pl`

- [ ] **Step 1: Test idea**

Create three flushed segments on the primary, then merge two of them. The standby has a fetch queue containing flush(A), flush(B), flush(C), merge(A∪B). If the fetcher processes merge first (which can happen with parallelism > 1 and faster path for smaller merged file), the coverage rule must:
- Adopt the merged segment from memtables A and B + segment C still around → wait, that's not the scenario. Re-read §8 walk-through.

Specifically test the "Out-of-order pulls" row: memtables 5,6,7; arrival order seg[5,6] then seg[5,5] then seg[6,6]:
- seg[5,6] arrives → coverage = mt5+mt6, adopt; replace mt5+mt6.
- seg[5,5] arrives → range is subset of existing seg[5,6] → stale, discard.
- seg[6,6] arrives → same → stale, discard.

```perl
use strict;
use warnings FATAL => 'all';
use lib 'test/perl';
use DpvReplication qw(setup_primary setup_standby wait_catchup);
use Test::More;

# This test is timing-sensitive. To deterministically exercise out-of-order
# adoption, we (a) configure the standby with fetch_parallelism=4 so multiple
# arrivals are in flight, and (b) instrument adoption to log every "stale" /
# "adopted" outcome. We then check the log post-hoc.

my $primary = setup_primary();
$primary->append_conf('postgresql.conf', "log_min_messages = DEBUG1");
$primary->restart;

my $standby = setup_standby($primary);
$standby->append_conf('postgresql.conf', "log_min_messages = DEBUG1");
$standby->restart;

# ... drive flushes and a merge on the primary ...
# ... wait for catchup ...
# ... grep the standby's logs for at least one "stale" outcome ...

my $log = $standby->logfile;
open my $fh, '<', $log or die $!;
my $found_stale = 0;
while (<$fh>) { $found_stale++ if /dpv adoption.*stale/i; }
close $fh;

ok($found_stale > 0, "saw at least one stale-discard outcome in the standby log");

done_testing();
```

This is "best-effort" — out-of-order isn't guaranteed without further determinism. If flaky, accept it as a soft test (mark `TODO` in the harness). The other tests cover the correctness path; this test specifically validates the discard path's existence.

- [ ] **Step 2: Run, commit**

```bash
prove -I test/perl test/t/113_replication_out_of_order_pulls.pl
git add pgvector/test/t/113_replication_out_of_order_pulls.pl
git commit -m "test: out-of-order pull arrivals discard stale entries"
```

---

## Self-review checklist for Plan 2

- [ ] **`SegmentCreated` emit follows the file write.** Verify in [lsmindex_io.c](../../../pgvector/src/lsmindex_io.c) that `dpv_emit_segment_created` is the last line of `flush_segment_to_disk` (after the metadata-rename `write_lsm_segment_metadata` call).
- [ ] **`SegmentReplaced` emit follows the merge/rebuild file write.** Both merge and rebuild paths emit it.
- [ ] **The fetcher's `pull_all_files` pulls metadata LAST.** Re-read [segment_fetcher.c](../../../pgvector/src/segment_fetcher.c): the `kinds[]` array ends with `DPV_FILE_METADATA`.
- [ ] **The fetcher writes each file via temp-then-rename.** Verify in `pull_one_file`.
- [ ] **The adoption rule holds `pool->seg_lock` + `slot->mt_lock` for the duration of the swap.** Verify in `dpv_attempt_adoption`.
- [ ] **Identity-translation limitation is documented in `segment_adoption.c`.** Plan 3 will replace it.
- [ ] **`VECTOR_STORAGE_BASE_DIR` is now a GUC.** Grep returns 0 hits for the old macro name.
- [ ] **All 4 integration tests pass.**
- [ ] **The smoke test from Plan 1 still passes.**

---

## End of Plan 2
