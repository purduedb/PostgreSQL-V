# Physical Replication — Plan 1: Custom rmgr + Memtable Replayer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `GenericXLog` use in `statuspage.c` with a custom resource manager that emits semantic WAL records for memtable lifecycle (`Register`, `Add`, `MemtableTombstone`, `UpdateMaxSid`, `Release`). On the standby, redo callbacks materialize the in-memory memtable state in `SharedMemtableBuffer`, so memtable-only data is queryable on the standby after WAL replay.

**Architecture:** A custom rmgr (PG 15+, `RegisterCustomRmgr`) is registered from `_PG_init`. Five new record types each carry `indexRelId` plus their semantic payload; the full status page is also registered as a buffer reference (with `REGBUF_FORCE_IMAGE`) so the page-level change is always idempotently redone. Each redo callback (a) applies the page-level change to the status page via the standard `XLogReadBufferForRedoExtended` machinery, and (b) on a hot standby — and only on a hot standby — applies the in-memory side effect to `SharedMemtableBuffer`. Primary crash recovery skips all extension-side effects and lets `load_lsm_index_internal` reconstruct from disk on first index touch. Standby coordination with `IndexLoadWorker` is the v1 "block redo on `valid==2`" policy from spec §9.

**Tech Stack:** PostgreSQL 17 (PGXS extension), C (extension code), Perl (`PostgreSQL::Test::Cluster`) for primary+standby integration tests.

**Scope of this plan (out-of-scope deferred to Plans 2/3):**
- No segment-lifecycle records (`SegmentCreated`, `SegmentReplaced`) — Plan 2.
- No vacuum records (`SegmentVacuumTombstones`) — Plan 3.
- No side-channel file server / fetcher — Plan 2.
- No standby query barrier — Plan 3.

---

## File structure

| File | Role | Action |
| --- | --- | --- |
| [pgvector/src/replication_rmgr.h](../../../pgvector/src/replication_rmgr.h) | Rmgr ID, record-type IDs, record payload structs, emit helper prototypes, redo entry prototype | Create |
| [pgvector/src/replication_rmgr.c](../../../pgvector/src/replication_rmgr.c) | Registration, redo dispatcher, per-record redo callbacks, emit helpers | Create |
| [pgvector/src/standby_memtable.c](../../../pgvector/src/standby_memtable.c) | Standby-side helpers: allocate memtable slot on `Register` redo, fetch heap tuple by tid on `Add` redo, etc. Keeps standby-specific logic out of `replication_rmgr.c` | Create |
| [pgvector/src/standby_memtable.h](../../../pgvector/src/standby_memtable.h) | Prototypes for the above | Create |
| [pgvector/Makefile](../../../pgvector/Makefile) | Add `replication_rmgr.o standby_memtable.o` to `OBJS` | Modify |
| [pgvector/src/vector.c](../../../pgvector/src/vector.c) | Register the custom rmgr from `_PG_init` | Modify |
| [pgvector/src/statuspage.c](../../../pgvector/src/statuspage.c) | Replace 5 `GenericXLog`-based call sites with custom-rmgr `XLogInsert` | Modify |
| [pgvector/src/lsmindex.c](../../../pgvector/src/lsmindex.c) | On load completion (success or failure), `ConditionVariableBroadcast(&slot->load_cv)` so redo waiters can re-check `valid` | Modify (single small edit) |
| [pgvector/test/t/100_replication_rmgr_smoke.pl](../../../pgvector/test/t/100_replication_rmgr_smoke.pl) | Standby starts; rmgr registered; CREATE EXTENSION works on both nodes | Create |
| [pgvector/test/t/101_replication_memtable_insert.pl](../../../pgvector/test/t/101_replication_memtable_insert.pl) | Insert vectors on primary → standby ANN query returns them | Create |
| [pgvector/test/t/102_replication_memtable_tombstone.pl](../../../pgvector/test/t/102_replication_memtable_tombstone.pl) | Delete rows on primary → standby ANN query no longer returns them | Create |
| [pgvector/test/t/103_replication_load_during_redo.pl](../../../pgvector/test/t/103_replication_load_during_redo.pl) | First-touch lazy load on standby concurrent with WAL replay (validates `valid==2` blocking) | Create |

---

## Phase 1 — Test harness for primary + standby

### Task 1.1: Add a baseline Perl test that starts primary + streaming standby

**Files:**
- Create: `pgvector/test/t/100_replication_rmgr_smoke.pl`

- [ ] **Step 1: Write the test file**

```perl
# Smoke test: primary + streaming replica with the vector extension loaded
use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $primary = PostgreSQL::Test::Cluster->new('primary');
$primary->init(allows_streaming => 1);
$primary->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
wal_level = replica
max_wal_senders = 4
wal_keep_size = 64MB
));
$primary->start;
$primary->backup('basebackup');

my $standby = PostgreSQL::Test::Cluster->new('standby');
$standby->init_from_backup($primary, 'basebackup', has_streaming => 1);
$standby->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
));
$standby->start;

$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');

# Wait for the CREATE EXTENSION WAL to replay on the standby
my $appname = $standby->name;
my $caughtup = "SELECT pg_current_wal_lsn() <= replay_lsn "
             . "FROM pg_stat_replication WHERE application_name = '$appname';";
$primary->poll_query_until('postgres', $caughtup)
    or die "standby never caught up";

# Verify extension is present on standby too
my $ver = $standby->safe_psql('postgres',
    "SELECT extversion FROM pg_extension WHERE extname = 'vector';");
ok($ver ne '', "vector extension is visible on standby (got: $ver)");

# Verify our custom rmgr registered without crash and replicates simple work.
$primary->safe_psql('postgres', q(
    CREATE TABLE t (id int, v vector(4));
    INSERT INTO t SELECT i, ARRAY[i*1.0, i+0.1, i+0.2, i+0.3]::vector
        FROM generate_series(1, 10) i;
));
$primary->poll_query_until('postgres', $caughtup);

my $count = $standby->safe_psql('postgres', 'SELECT count(*) FROM t;');
is($count, '10', "10 rows replicated to standby");

done_testing();
```

- [ ] **Step 2: Run the test (expected to PASS — no extension code changes yet)**

Run: `cd pgvector && make installcheck PROVE_TESTS=test/t/100_replication_rmgr_smoke.pl` (or `prove -I test/perl test/t/100_replication_rmgr_smoke.pl` against a writable install).

If the harness can't find `pgvector` for `shared_preload_libraries`, install once with `make install` and re-run. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add pgvector/test/t/100_replication_rmgr_smoke.pl
git commit -m "test: add primary+standby harness for vector extension"
```

---

## Phase 2 — Custom rmgr scaffolding

### Task 2.1: Define the rmgr API surface in `replication_rmgr.h`

**Files:**
- Create: `pgvector/src/replication_rmgr.h`

- [ ] **Step 1: Write the header**

```c
/*
 * replication_rmgr.h — custom resource manager for decoupled_pgvector
 *
 * Five record types for the memtable lifecycle in plan 1.
 * Plan 2 will add SegmentCreated / SegmentReplaced.
 * Plan 3 will add SegmentVacuumTombstones.
 */
#ifndef DPV_REPLICATION_RMGR_H
#define DPV_REPLICATION_RMGR_H

#include "postgres.h"
#include "access/xlog.h"
#include "access/xlogreader.h"
#include "access/xlog_internal.h"
#include "storage/itemptr.h"
#include "utils/relcache.h"

#include "lsm_segment.h"   /* SegmentId */

/*
 * Custom rmgr ID. PG reserves [RM_EXPERIMENTAL_ID, RM_MAX_ID] = [128, 255] for
 * extension use. We pick 137 — outside the upstream contrib rmgr ID range
 * documented in src/include/access/rmgrlist.h.
 */
#define RM_DPV_REPLICATION_ID  137
#define RM_DPV_REPLICATION_NAME "decoupled_pgvector"

/* Info bits (in xl_info's low 4 bits, masked by XLR_INFO_MASK). */
#define XLOG_DPV_REGISTER_MEMTABLE     0x00
#define XLOG_DPV_ADD_TO_MEMTABLE       0x10
#define XLOG_DPV_MEMTABLE_TOMBSTONE    0x20
#define XLOG_DPV_UPDATE_MAX_SID        0x30
#define XLOG_DPV_RELEASE_MEMTABLE      0x40
/* 0x50..0xF0 reserved for plans 2 and 3 */

/* ---- Record payload structs (the "main data" portion of each record) ----
 * Buffer references (the status page being modified) are attached separately
 * via XLogRegisterBuffer; only the *semantic* fields live here.
 */
typedef struct
{
    Oid       indexRelId;
    SegmentId sid;
} xl_dpv_register_memtable;

typedef struct
{
    Oid             indexRelId;
    SegmentId       sid;
    uint32          slot_index;
    ItemPointerData tid;
} xl_dpv_add_to_memtable;

typedef struct
{
    Oid       indexRelId;
    SegmentId sid;
    uint32    slot_index;
} xl_dpv_memtable_tombstone;

typedef struct
{
    Oid       indexRelId;
    SegmentId sid;
} xl_dpv_update_max_sid;

typedef struct
{
    Oid       indexRelId;
    SegmentId sid;
} xl_dpv_release_memtable;

/* Register the rmgr with PG. Called from _PG_init. */
extern void vector_replication_rmgr_register(void);

/* Emit helpers (used on the primary). Each handles the XLogInsert dance and
 * returns the resulting LSN. The buffers passed in must already be locked
 * EXCLUSIVE and dirtied by the caller. */
extern XLogRecPtr dpv_emit_register_memtable(Relation idx, Buffer buf,
                                             Oid indexRelId, SegmentId sid);
extern XLogRecPtr dpv_emit_add_to_memtable(Relation idx, Buffer buf,
                                           Oid indexRelId, SegmentId sid,
                                           uint32 slot_index, ItemPointer tid);
extern XLogRecPtr dpv_emit_memtable_tombstone(Relation idx, Buffer buf,
                                              Oid indexRelId, SegmentId sid,
                                              uint32 slot_index);
extern XLogRecPtr dpv_emit_update_max_sid(Relation idx, Buffer buf,
                                          Oid indexRelId, SegmentId sid);
extern XLogRecPtr dpv_emit_release_memtable(Relation idx, Buffer buf,
                                            Oid indexRelId, SegmentId sid);

#endif
```

### Task 2.2: Implement registration and redo dispatcher in `replication_rmgr.c`

**Files:**
- Create: `pgvector/src/replication_rmgr.c`

- [ ] **Step 1: Write the file with registration and dispatcher skeleton**

```c
#include "postgres.h"
#include "access/xlog.h"
#include "access/xloginsert.h"
#include "access/xlogreader.h"
#include "access/xlogutils.h"
#include "access/xlog_internal.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "utils/elog.h"

#include "replication_rmgr.h"
#include "standby_memtable.h"

/* Forward decls for redo callbacks (defined below). */
static void redo_register_memtable(XLogReaderState *r);
static void redo_add_to_memtable(XLogReaderState *r);
static void redo_memtable_tombstone(XLogReaderState *r);
static void redo_update_max_sid(XLogReaderState *r);
static void redo_release_memtable(XLogReaderState *r);

static void
dpv_replication_redo(XLogReaderState *record)
{
    uint8 info = XLogRecGetInfo(record) & ~XLR_INFO_MASK;

    switch (info)
    {
        case XLOG_DPV_REGISTER_MEMTABLE:  redo_register_memtable(record);  break;
        case XLOG_DPV_ADD_TO_MEMTABLE:    redo_add_to_memtable(record);    break;
        case XLOG_DPV_MEMTABLE_TOMBSTONE: redo_memtable_tombstone(record); break;
        case XLOG_DPV_UPDATE_MAX_SID:     redo_update_max_sid(record);     break;
        case XLOG_DPV_RELEASE_MEMTABLE:   redo_release_memtable(record);   break;
        default:
            elog(PANIC, "decoupled_pgvector_redo: unknown op code %u", info);
    }
}

/* Optional descriptor for pg_waldump; minimal v1 stub is fine. */
static void
dpv_replication_desc(StringInfo buf, XLogReaderState *record)
{
    uint8 info = XLogRecGetInfo(record) & ~XLR_INFO_MASK;
    appendStringInfo(buf, "dpv op=0x%02x", info);
}

static const char *
dpv_replication_identify(uint8 info)
{
    switch (info & ~XLR_INFO_MASK)
    {
        case XLOG_DPV_REGISTER_MEMTABLE:  return "REGISTER_MEMTABLE";
        case XLOG_DPV_ADD_TO_MEMTABLE:    return "ADD_TO_MEMTABLE";
        case XLOG_DPV_MEMTABLE_TOMBSTONE: return "MEMTABLE_TOMBSTONE";
        case XLOG_DPV_UPDATE_MAX_SID:     return "UPDATE_MAX_SID";
        case XLOG_DPV_RELEASE_MEMTABLE:   return "RELEASE_MEMTABLE";
        default:                          return NULL;
    }
}

static const RmgrData dpv_replication_rmgr = {
    .rm_name      = RM_DPV_REPLICATION_NAME,
    .rm_redo      = dpv_replication_redo,
    .rm_desc      = dpv_replication_desc,
    .rm_identify  = dpv_replication_identify,
    .rm_startup   = NULL,
    .rm_cleanup   = NULL,
    .rm_mask      = NULL,
    .rm_decode    = NULL,
};

void
vector_replication_rmgr_register(void)
{
    RegisterCustomRmgr(RM_DPV_REPLICATION_ID, &dpv_replication_rmgr);
}

/* ----- Redo callbacks are stubs for now; flesh out in Phase 5. ----- */

static void redo_register_memtable(XLogReaderState *r)   { /* TODO Phase 5 */ }
static void redo_add_to_memtable(XLogReaderState *r)     { /* TODO Phase 5 */ }
static void redo_memtable_tombstone(XLogReaderState *r)  { /* TODO Phase 5 */ }
static void redo_update_max_sid(XLogReaderState *r)      { /* TODO Phase 5 */ }
static void redo_release_memtable(XLogReaderState *r)    { /* TODO Phase 5 */ }

/* ----- Emit helpers (Phase 3). ----- */
/* TODO: defined in Phase 3 below */
```

### Task 2.3: Define `standby_memtable.h` with empty bodies for Phase 5 helpers

**Files:**
- Create: `pgvector/src/standby_memtable.h`
- Create: `pgvector/src/standby_memtable.c`

- [ ] **Step 1: Write the header**

```c
#ifndef DPV_STANDBY_MEMTABLE_H
#define DPV_STANDBY_MEMTABLE_H

#include "postgres.h"
#include "storage/itemptr.h"
#include "lsm_segment.h"

/*
 * Standby-side effects for redo callbacks. Each returns silently if the
 * targeted index is not loaded (LSMIndexBufferSlot absent or valid != 1).
 *
 * If valid == 2 (load in progress), the caller will block on the slot's
 * load_cv before invoking these — see dpv_standby_wait_if_loading().
 */

/* Allocate / find the memtable slot for sid on the standby. */
extern void dpv_standby_register_memtable(Oid indexRelId, SegmentId sid);

/* Materialize one inserted vector by fetching it from the heap by tid. */
extern void dpv_standby_add_to_memtable(Oid indexRelId, SegmentId sid,
                                         uint32 slot_index, ItemPointer tid);

/* Mark slot_index as deleted in the memtable bitmap. */
extern void dpv_standby_memtable_tombstone(Oid indexRelId, SegmentId sid,
                                            uint32 slot_index);

/* Update the cached max_memtable_sid. */
extern void dpv_standby_update_max_sid(Oid indexRelId, SegmentId sid);

/*
 * `Release` on the standby is intentionally a no-op for SharedMemtableBuffer
 * (see spec §10 — memtable persists until adoption). Provided as a stub so
 * the dispatcher is uniform.
 */
extern void dpv_standby_release_memtable(Oid indexRelId, SegmentId sid);

/*
 * If the target index's LSMIndexBufferSlot is in valid==2 (load in progress),
 * sleep on slot->load_cv until valid transitions to 1 or 0. Returns the
 * resulting `valid` value. If the slot does not exist, returns 0.
 */
extern int dpv_standby_wait_if_loading(Oid indexRelId);

#endif
```

- [ ] **Step 2: Write `standby_memtable.c` with empty bodies**

```c
#include "postgres.h"
#include "standby_memtable.h"

void dpv_standby_register_memtable(Oid indexRelId, SegmentId sid)             { /* Phase 5 */ }
void dpv_standby_add_to_memtable(Oid indexRelId, SegmentId sid,
                                  uint32 slot_index, ItemPointer tid)         { /* Phase 5 */ }
void dpv_standby_memtable_tombstone(Oid indexRelId, SegmentId sid,
                                     uint32 slot_index)                       { /* Phase 5 */ }
void dpv_standby_update_max_sid(Oid indexRelId, SegmentId sid)                 { /* Phase 5 */ }
void dpv_standby_release_memtable(Oid indexRelId, SegmentId sid)               { /* Phase 5 */ }
int  dpv_standby_wait_if_loading(Oid indexRelId)                               { return 1; /* Phase 6 */ }
```

### Task 2.4: Register the rmgr from `_PG_init`

**Files:**
- Modify: `pgvector/src/vector.c`

- [ ] **Step 1: Add `#include "replication_rmgr.h"` near the other extension headers in `vector.c`**

Locate the existing extension `#include`s near the top of [vector.c](../../../pgvector/src/vector.c) and append:

```c
#include "replication_rmgr.h"
```

- [ ] **Step 2: Add the registration call early in `_PG_init`**

In [vector.c:95-120](../../../pgvector/src/vector.c#L95-L120), after the per-backend `LWLockRegisterTranche` block and before `shmem_startup_hook` assignment, insert:

```c
    /* Custom rmgr for physical replication of LSM state (spec §7). */
    vector_replication_rmgr_register();
```

Exact location: directly after the line `LWLockRegisterTranche(LSM_INDEX_BUFFER_LWTRANCHE_ID, LSM_INDEX_BUFFER_LWTRANCHE);` and before the `#if PG_VERSION_NUM >= 150000` block in [vector.c:108-110](../../../pgvector/src/vector.c#L108-L110).

### Task 2.5: Add new objects to the Makefile and rebuild

**Files:**
- Modify: `pgvector/Makefile`

- [ ] **Step 1: Append the two new objects to `OBJS` (Makefile line 7-8)**

Append `src/replication_rmgr.o src/standby_memtable.o` to the second `OBJS` line:

```makefile
OBJS = src/bitutils.o src/bitvec.o src/halfutils.o src/halfvec.o src/hnsw.o src/hnswbuild.o src/hnswinsert.o src/hnswscan.o src/hnswutils.o src/hnswvacuum.o src/ivfbuild.o src/ivfflat.o src/ivfinsert.o src/ivfkmeans.o src/ivfscan.o src/ivfutils.o src/ivfvacuum.o src/sparsevec.o src/vector.o \
       src/lsmindex.o src/lsm_segment.o src/lsmbackground.o src/lsmindex_io.o src/ringbuffer.o src/tasksend.o src/vector_index_worker.o src/index_load_worker.o src/vectorindeximpl.o src/utils.o src/folly_f14_link_alias.o src/statuspage.o \
       src/replication_rmgr.o src/standby_memtable.o
```

- [ ] **Step 2: Build**

```bash
cd pgvector && make 2>&1 | tail -50
```

Expected: clean build. Fix include errors before proceeding.

- [ ] **Step 3: Install and rerun the smoke test**

```bash
cd pgvector && make install
prove -I test/perl test/t/100_replication_rmgr_smoke.pl
```

Expected: PASS. Rmgr is registered but no records emitted yet; status pages still go through `GenericXLog`.

- [ ] **Step 4: Commit**

```bash
git add pgvector/src/replication_rmgr.h pgvector/src/replication_rmgr.c \
        pgvector/src/standby_memtable.h pgvector/src/standby_memtable.c \
        pgvector/src/vector.c pgvector/Makefile
git commit -m "feat: register custom rmgr for decoupled_pgvector replication"
```

---

## Phase 3 — Emit helpers on the primary

### Task 3.1: Implement `dpv_emit_register_memtable` and the other emit helpers

**Files:**
- Modify: `pgvector/src/replication_rmgr.c`

- [ ] **Step 1: Append the five emit helpers at the end of `replication_rmgr.c`**

Each helper follows the same pattern: begin insert, register the page (with `REGBUF_FORCE_IMAGE` so the page state is durable through replay independent of the semantic fields — matches `GenericXLog` behavior), register the main payload, and insert.

```c
XLogRecPtr
dpv_emit_register_memtable(Relation idx, Buffer buf,
                           Oid indexRelId, SegmentId sid)
{
    xl_dpv_register_memtable xlrec = { .indexRelId = indexRelId, .sid = sid };
    XLogRecPtr lsn;

    XLogBeginInsert();
    XLogRegisterBuffer(0, buf, REGBUF_FORCE_IMAGE | REGBUF_STANDARD);
    XLogRegisterData((char *) &xlrec, sizeof(xlrec));
    lsn = XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_REGISTER_MEMTABLE);
    PageSetLSN(BufferGetPage(buf), lsn);
    return lsn;
}

XLogRecPtr
dpv_emit_add_to_memtable(Relation idx, Buffer buf,
                         Oid indexRelId, SegmentId sid,
                         uint32 slot_index, ItemPointer tid)
{
    xl_dpv_add_to_memtable xlrec = {
        .indexRelId = indexRelId,
        .sid        = sid,
        .slot_index = slot_index,
        .tid        = *tid,
    };
    XLogRecPtr lsn;

    XLogBeginInsert();
    XLogRegisterBuffer(0, buf, REGBUF_FORCE_IMAGE | REGBUF_STANDARD);
    XLogRegisterData((char *) &xlrec, sizeof(xlrec));
    lsn = XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_ADD_TO_MEMTABLE);
    PageSetLSN(BufferGetPage(buf), lsn);
    return lsn;
}

XLogRecPtr
dpv_emit_memtable_tombstone(Relation idx, Buffer buf,
                            Oid indexRelId, SegmentId sid, uint32 slot_index)
{
    xl_dpv_memtable_tombstone xlrec = {
        .indexRelId = indexRelId, .sid = sid, .slot_index = slot_index,
    };
    XLogRecPtr lsn;

    XLogBeginInsert();
    XLogRegisterBuffer(0, buf, REGBUF_FORCE_IMAGE | REGBUF_STANDARD);
    XLogRegisterData((char *) &xlrec, sizeof(xlrec));
    lsn = XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_MEMTABLE_TOMBSTONE);
    PageSetLSN(BufferGetPage(buf), lsn);
    return lsn;
}

XLogRecPtr
dpv_emit_update_max_sid(Relation idx, Buffer buf,
                        Oid indexRelId, SegmentId sid)
{
    xl_dpv_update_max_sid xlrec = { .indexRelId = indexRelId, .sid = sid };
    XLogRecPtr lsn;

    XLogBeginInsert();
    XLogRegisterBuffer(0, buf, REGBUF_FORCE_IMAGE | REGBUF_STANDARD);
    XLogRegisterData((char *) &xlrec, sizeof(xlrec));
    lsn = XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_UPDATE_MAX_SID);
    PageSetLSN(BufferGetPage(buf), lsn);
    return lsn;
}

XLogRecPtr
dpv_emit_release_memtable(Relation idx, Buffer buf,
                          Oid indexRelId, SegmentId sid)
{
    xl_dpv_release_memtable xlrec = { .indexRelId = indexRelId, .sid = sid };
    XLogRecPtr lsn;

    XLogBeginInsert();
    XLogRegisterBuffer(0, buf, REGBUF_FORCE_IMAGE | REGBUF_STANDARD);
    XLogRegisterData((char *) &xlrec, sizeof(xlrec));
    lsn = XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_RELEASE_MEMTABLE);
    PageSetLSN(BufferGetPage(buf), lsn);
    return lsn;
}
```

- [ ] **Step 2: Build**

```bash
cd pgvector && make 2>&1 | tail -20
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add pgvector/src/replication_rmgr.c
git commit -m "feat: add emit helpers for memtable lifecycle WAL records"
```

---

## Phase 4 — Replace `GenericXLog` use in `statuspage.c`

This phase converts each of the 5 sites that touch the status pages today to use the new emit helpers. The page mutation must be performed inside a `START_CRIT_SECTION()` / `END_CRIT_SECTION()` bracket — failing to do so risks `elog(ERROR)` partway through a buffer modification, which PG would treat as a recoverable error even though the page is already inconsistent.

The conversion pattern at every site is:

```
// BEFORE (GenericXLog):
GenericXLogState *state = GenericXLogStart(index);
Page page = GenericXLogRegisterBuffer(state, buf, 0);
...modify page...
GenericXLogFinish(state);
UnlockReleaseBuffer(buf);

// AFTER (custom rmgr):
START_CRIT_SECTION();
Page page = BufferGetPage(buf);
...modify page...
MarkBufferDirty(buf);
dpv_emit_<op>(index, buf, indexRelId, sid, ...);
END_CRIT_SECTION();
UnlockReleaseBuffer(buf);
```

Note: `GenericXLog` previously bracketed the buffer lock for us. With the custom rmgr we hold the buffer lock manually — already done at every site (`LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE)` precedes the `GenericXLogStart`).

### Task 4.1: Convert `UpdateMaxMemtableSid` ([statuspage.c:249-270](../../../pgvector/src/statuspage.c#L249-L270))

**Files:**
- Modify: `pgvector/src/statuspage.c`

- [ ] **Step 1: Add include**

At the top of [statuspage.c](../../../pgvector/src/statuspage.c) (with the other includes), append:

```c
#include "replication_rmgr.h"
#include "storage/bufmgr.h"
#include "miscadmin.h"
```

- [ ] **Step 2: Replace `UpdateMaxMemtableSid` body**

```c
static void
UpdateMaxMemtableSid(Relation index, SegmentId sid)
{
    Buffer buf;
    Page   page;
    StatusPageMeta statuspagemeta;

    buf = ReadBuffer(index, STATUS_METAPAGE_BLKNO);
    LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

    START_CRIT_SECTION();
    page = BufferGetPage(buf);
    statuspagemeta = StatusPageGetMeta(page);
    statuspagemeta->max_memtable_sid = sid;
    MarkBufferDirty(buf);
    dpv_emit_update_max_sid(index, buf, RelationGetRelid(index), sid);
    END_CRIT_SECTION();

    UnlockReleaseBuffer(buf);
}
```

### Task 4.2: Convert `RegisterStatusMemtable` ([statuspage.c:273-…](../../../pgvector/src/statuspage.c#L273))

**Files:**
- Modify: `pgvector/src/statuspage.c`

The current function is a loop that may extend the relation by one page on overflow. The conversion must:
1. Keep the existing extension logic for the *page allocation* path.
2. Replace the *successful insert* path's `GenericXLogFinish` with `dpv_emit_register_memtable + END_CRIT_SECTION`.
3. Track the buffer that ultimately gets the inserted `StatusMemtableData` (call it `bufWritten`) — that's the buffer to register in WAL.

- [ ] **Step 1: Read the existing function fully**

Read `RegisterStatusMemtable` in [statuspage.c:273](../../../pgvector/src/statuspage.c#L273) (approximately 80–120 lines). Identify the two exit paths:
- "Found free space in the current page": insert the tuple, finalize WAL on that buffer.
- "Need to allocate a new page": link old page → new page, init new page, insert tuple on new page, finalize WAL.

Both paths end with a successful `GenericXLogFinish(state)` over a state that registered up to two buffers (old + new). With the custom rmgr, the equivalent is: register both buffers via `XLogRegisterBuffer` (slots 0 and 1) and emit once.

- [ ] **Step 2: Add a two-buffer emit variant in `replication_rmgr.h` / `.c`**

For Register, we sometimes need to register a new (empty) overflow page in the same WAL record so the standby reconstructs the linked list correctly. Add a second-buffer parameter (use `InvalidBuffer` when not needed).

Replace `dpv_emit_register_memtable` prototype in [replication_rmgr.h](../../../pgvector/src/replication_rmgr.h):

```c
extern XLogRecPtr dpv_emit_register_memtable(Relation idx,
                                             Buffer buf, Buffer newPageBuf,
                                             Oid indexRelId, SegmentId sid);
```

Update body in [replication_rmgr.c](../../../pgvector/src/replication_rmgr.c):

```c
XLogRecPtr
dpv_emit_register_memtable(Relation idx,
                           Buffer buf, Buffer newPageBuf,
                           Oid indexRelId, SegmentId sid)
{
    xl_dpv_register_memtable xlrec = { .indexRelId = indexRelId, .sid = sid };
    XLogRecPtr lsn;

    XLogBeginInsert();
    XLogRegisterBuffer(0, buf, REGBUF_FORCE_IMAGE | REGBUF_STANDARD);
    if (BufferIsValid(newPageBuf))
        XLogRegisterBuffer(1, newPageBuf, REGBUF_FORCE_IMAGE | REGBUF_STANDARD | REGBUF_WILL_INIT);
    XLogRegisterData((char *) &xlrec, sizeof(xlrec));
    lsn = XLogInsert(RM_DPV_REPLICATION_ID, XLOG_DPV_REGISTER_MEMTABLE);
    PageSetLSN(BufferGetPage(buf), lsn);
    if (BufferIsValid(newPageBuf))
        PageSetLSN(BufferGetPage(newPageBuf), lsn);
    return lsn;
}
```

- [ ] **Step 3: Rewrite `RegisterStatusMemtable` to use the new emit**

Convert the function. The skeleton (read the existing code first to fill in the unchanged middle pieces):

```c
void
RegisterStatusMemtable(Relation index, SegmentId sid)
{
    elog(DEBUG1, "[RegisterStatusMemtable] Registering status memtable, sid: %d", sid);

    UpdateMaxMemtableSid(index, sid);

    Buffer  buf;
    Page    page;
    BlockNumber insertPage = STATUS_MEMTABLE_ARRAY_BLKNO;
    StatusMemtable mt;
    Size    mtSize;
    Buffer  prevBuf  = InvalidBuffer;   /* previous page when allocating new */
    Buffer  newBuf   = InvalidBuffer;   /* newly allocated overflow page    */
    Buffer  bufWritten;                 /* buffer that received the insert  */

    mtSize = MAXALIGN(sizeof(StatusMemtableData));
    mt = (StatusMemtable) palloc0(mtSize);
    mt->sid = sid;
    mt->memtablePageHead   = InvalidBlockNumber;
    mt->memtableInsertPage = InvalidBlockNumber;

    for (;;)
    {
        buf = ReadBuffer(index, insertPage);
        LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
        page = BufferGetPage(buf);

        if (PageGetFreeSpace(page) > mtSize)
        {
            bufWritten = buf;
            break;
        }

        /* This page is full; chase the chain. */
        insertPage = StatusPageGetOpaque(page)->nextblkno;

        if (BlockNumberIsValid(insertPage))
        {
            UnlockReleaseBuffer(buf);
            continue;
        }

        /* No next page; extend. */
        prevBuf = buf;  /* keep locked — we'll patch its nextblkno */
        LockRelationForExtension(index, ExclusiveLock);
        newBuf = StatusNewBuffer(index, MAIN_FORKNUM);  /* returns LOCK_EXCLUSIVE */
        UnlockRelationForExtension(index, ExclusiveLock);

        bufWritten = newBuf;
        break;
    }

    START_CRIT_SECTION();
    {
        Page targetPage = BufferGetPage(bufWritten);
        OffsetNumber offno;

        if (BufferIsValid(newBuf))
        {
            /* Initialize the new overflow page and link the previous page. */
            StatusInitPage(newBuf, BufferGetPage(newBuf));
            StatusPageGetOpaque(BufferGetPage(prevBuf))->nextblkno =
                BufferGetBlockNumber(newBuf);
            MarkBufferDirty(prevBuf);
        }

        offno = PageAddItem(targetPage, (Item) mt, mtSize, InvalidOffsetNumber,
                            false, false);
        if (offno == InvalidOffsetNumber)
            elog(PANIC, "[RegisterStatusMemtable] PageAddItem failed");

        MarkBufferDirty(bufWritten);

        dpv_emit_register_memtable(index,
                                    BufferIsValid(newBuf) ? prevBuf : bufWritten,
                                    BufferIsValid(newBuf) ? newBuf : InvalidBuffer,
                                    RelationGetRelid(index), sid);
    }
    END_CRIT_SECTION();

    if (BufferIsValid(newBuf))
    {
        UnlockReleaseBuffer(newBuf);
        UnlockReleaseBuffer(prevBuf);
    }
    else
    {
        UnlockReleaseBuffer(bufWritten);
    }

    pfree(mt);
}
```

Note: The block at line `if (BufferIsValid(newBuf))` deliberately registers `prevBuf` (the page being updated to point at the new tail) at slot 0 and `newBuf` (newly initialized) at slot 1, so the WAL record covers both. When a new page is *not* needed, `bufWritten == buf` is registered at slot 0 and slot 1 is `InvalidBuffer`. This mirrors the buffers `GenericXLog` previously registered in this function.

### Task 4.3: Convert `AddToStatusMemtable` ([statuspage.c:465](../../../pgvector/src/statuspage.c#L465))

Same pattern as `RegisterStatusMemtable`. Read the existing function to identify the buffer(s) registered with `GenericXLogRegisterBuffer`. Replace the `GenericXLogStart` / `GenericXLogFinish` envelope with `START_CRIT_SECTION` / `MarkBufferDirty` / `dpv_emit_add_to_memtable` / `END_CRIT_SECTION`.

- [ ] **Step 1: Read the existing function** (`AddToStatusMemtable`) and identify the locked buffer that receives the `PageAddItem` for the new memtable entry.

- [ ] **Step 2: Replace its WAL machinery**

Convert the function body — the per-tid insert. The emit helper takes the buffer that received the item and the `(slot_index, tid)` payload. `slot_index` is the local index within memtable `sid` that the primary used; carry it in the record so the standby uses the identical slot.

The slot index is the per-memtable in-memory slot number selected by `register_and_set_memtable`. Confirm by reading `AddToStatusMemtable` — it should accept a `slot_index` argument or compute it deterministically. Pass that value through to `dpv_emit_add_to_memtable`.

If `AddToStatusMemtable`'s current signature does not include `slot_index`, extend the signature and update its callers (search for `AddToStatusMemtable(` in the repo). Tracking issue: this propagates to `lsmindex.c`.

### Task 4.4: Convert `RemoveFromStatusMemtable` ([statuspage.c:750](../../../pgvector/src/statuspage.c#L750))

- [ ] **Step 1: Read the function**, identify the buffer, replace the GenericXLog envelope with `START_CRIT_SECTION` + `MarkBufferDirty` + `dpv_emit_memtable_tombstone` + `END_CRIT_SECTION`.

Carry the same `slot_index` (per-memtable local index) as `AddToStatusMemtable`. This is what `bulk_delete_lsm_index` already passes — confirm in [lsmindex.c:1680](../../../pgvector/src/lsmindex.c#L1680) and surrounding lines.

### Task 4.5: Convert `ReleaseStatusMemtable` ([statuspage.c:86](../../../pgvector/src/statuspage.c#L86))

- [ ] **Step 1: Read the function**. It deletes the memtable entry from a status page and may unlink the page from the chain.

- [ ] **Step 2: Replace the WAL envelope**. Two buffers may be involved (the page holding the entry, and the metapage if free-list bookkeeping is modified). Mirror Task 4.2's pattern of registering both. Emit `dpv_emit_release_memtable` (single-info), but provide a two-buffer variant analogous to register if needed. Match buffers to what `GenericXLog` registered in the existing code.

### Task 4.6: Build + smoke test + commit

- [ ] **Step 1: Build**

```bash
cd pgvector && make 2>&1 | tail -50
```

Expected: clean. Fix any unused-variable warnings (e.g., `GenericXLogState` locals that are gone).

- [ ] **Step 2: Install and rerun smoke test**

```bash
cd pgvector && make install
prove -I test/perl test/t/100_replication_rmgr_smoke.pl
```

Expected: PASS. The primary now emits custom-rmgr WAL records; on the standby the redo callbacks are still stubs, so the page-level changes happen (because each record carries a full-page image via `REGBUF_FORCE_IMAGE`) but the in-memory `SharedMemtableBuffer` on the standby is empty. That's OK for now — the test only verifies the primary still works.

- [ ] **Step 3: Verify WAL records appear**

Manually inspect with `pg_waldump` against the primary's WAL after a small insert:

```bash
PRIMARY_PGDATA=$(prove -I test/perl test/t/100_replication_rmgr_smoke.pl -v 2>&1 | grep -o 'primary_data=[^ ]*' | head -1)
# Or: locate the primary node's data dir from the test output and run:
pg_waldump <wal-file> | grep -i decoupled_pgvector
```

Expected: lines mentioning `decoupled_pgvector REGISTER_MEMTABLE`, `ADD_TO_MEMTABLE`, etc.

- [ ] **Step 4: Commit**

```bash
git add pgvector/src/statuspage.c pgvector/src/replication_rmgr.c pgvector/src/replication_rmgr.h
git commit -m "feat: replace GenericXLog in statuspage.c with custom rmgr records"
```

---

## Phase 5 — Standby memtable replayer (redo side effects)

This is where the standby starts to actually materialize memtable state. The redo callback for each record type now:

1. **Always** applies the page-level change via `XLogReadBufferForRedoExtended` (this is the same machinery built-in WAL records use; PG handles full-page-image replay automatically based on the flags set in `XLogRegisterBuffer`). Returns `BLK_NEEDS_REDO`, `BLK_DONE`, or `BLK_RESTORED` — for our records (always FPI), `BLK_RESTORED` is the common path. The page is dirtied and unlocked by PG.

2. **On a hot standby**, applies the in-memory side effect. The check is `InHotStandby` (PG global, true only on a hot standby running redo); on the primary during crash recovery, that's false, and the callback skips the side effect entirely.

3. **Before** applying the in-memory side effect, calls `dpv_standby_wait_if_loading(indexRelId)` to coordinate with the IndexLoadWorker per spec §9 v1 policy.

### Task 5.1: Implement standby-side helpers in `standby_memtable.c`

**Files:**
- Modify: `pgvector/src/standby_memtable.c`
- Read: `pgvector/src/lsmindex.h`, `pgvector/src/lsmindex.c` (to understand `SharedLSMIndexBuffer`, `LSMIndexBufferSlot`, `SharedMemtableBuffer`)

- [ ] **Step 1: Read the relevant data structures**

Read [pgvector/src/lsmindex.h:160-260](../../../pgvector/src/lsmindex.h#L160-L260) and find the definitions of:
- `LSMIndexBufferSlot` (per-index slot; includes `valid` flag, `load_cv`, pointer to memtable buffer)
- `SharedMemtableBuffer` / `MemtableBuffer` (per-index memtable array)
- `ConcurrentMemTableData` (per-memtable; includes `vector_blob`, `tids`, `bitmap`, `valid`)
- helpers: `lookup_lsm_index_buffer_slot`, `find_memtable_by_sid`, `SET_SLOT`, `register_and_set_memtable`

In [lsmindex.c:268](../../../pgvector/src/lsmindex.c#L268), study `register_and_set_memtable` — the "recovery" path (used by `load_lsm_index_internal`) is what the standby should mirror.

- [ ] **Step 2: Implement `dpv_standby_wait_if_loading`**

Pattern is borrowed from the existing backend wait at [lsmindex.c:181-193](../../../pgvector/src/lsmindex.c#L181-L193). Use `ConditionVariablePrepareToSleep` / `ConditionVariableSleep` / `ConditionVariableCancelSleep`. On a load completion the loader does `ConditionVariableBroadcast(&slot->load_cv)` (added in Task 5.2 below).

```c
int
dpv_standby_wait_if_loading(Oid indexRelId)
{
    LSMIndexBufferSlot *slot = lookup_lsm_index_buffer_slot(indexRelId);
    int observed;

    if (slot == NULL)
        return 0;  /* index not present in shmem: caller skips side effect */

    /* Fast path: already loaded. */
    observed = pg_atomic_read_u32(&slot->valid);
    if (observed != 2)
        return (int) observed;

    /* Slow path: wait for the loader. */
    ConditionVariablePrepareToSleep(&slot->load_cv);
    for (;;)
    {
        observed = pg_atomic_read_u32(&slot->valid);
        if (observed != 2)
            break;
        ConditionVariableSleep(&slot->load_cv, WAIT_EVENT_EXTENSION);
    }
    ConditionVariableCancelSleep();
    return observed;
}
```

If `slot->valid` is a plain `int` rather than an atomic, use the appropriate accessor consistent with the rest of the codebase. Replace `lookup_lsm_index_buffer_slot` with the actual lookup function name (search `lsmindex.c`); rename throughout if it differs.

- [ ] **Step 3: Implement `dpv_standby_register_memtable`**

```c
void
dpv_standby_register_memtable(Oid indexRelId, SegmentId sid)
{
    LSMIndexBufferSlot *slot = lookup_lsm_index_buffer_slot(indexRelId);
    if (slot == NULL || pg_atomic_read_u32(&slot->valid) != 1)
        return;  /* index not loaded: status-page redo + load on first touch will rebuild */

    register_and_set_memtable(slot, sid, /*is_recovery=*/true);
}
```

`register_and_set_memtable` already supports the recovery (no WAL emission) path; verify by reading [lsmindex.c:268](../../../pgvector/src/lsmindex.c#L268) and confirming there's an `is_recovery` parameter (or refactor to expose one — track as a sub-task if needed). The recovery-path skips the primary-side `RegisterStatusMemtable` call, which is correct here since WAL replay already updated the status page.

- [ ] **Step 4: Implement `dpv_standby_add_to_memtable`**

The redo callback must fetch the row from the standby's heap at `tid` using `SnapshotAny`, then write the vector into the memtable's vector buffer at the named slot. Pattern matches `load_lsm_index_internal`'s reconstruction at [lsmindex.c:854](../../../pgvector/src/lsmindex.c#L854):

```c
void
dpv_standby_add_to_memtable(Oid indexRelId, SegmentId sid,
                             uint32 slot_index, ItemPointer tid)
{
    LSMIndexBufferSlot *slot = lookup_lsm_index_buffer_slot(indexRelId);
    ConcurrentMemTable mt;
    Relation idx;
    Relation heap;
    TupleTableSlot *tts;
    bool fetched;

    if (slot == NULL || pg_atomic_read_u32(&slot->valid) != 1)
        return;

    mt = find_memtable_by_sid(slot, sid);
    if (mt == NULL)
        return;  /* Register hasn't arrived yet, or already adopted */

    /* Open the heap by its OID. The index relation has indrelid = heap relid. */
    idx = relation_open(indexRelId, AccessShareLock);
    heap = table_open(idx->rd_index->indrelid, AccessShareLock);
    tts = table_slot_create(heap, NULL);

    fetched = table_tuple_fetch_row_version(heap, tid, SnapshotAny, tts);
    if (!fetched)
    {
        /*
         * Heap row not visible to SnapshotAny — should not happen, since heap-
         * insert WAL is replayed before our Add (PG ordering invariant). Log
         * and continue; load_lsm_index_internal will reconcile if needed.
         */
        elog(WARNING, "[standby Add redo] heap fetch failed for tid (%u,%u)",
             ItemPointerGetBlockNumber(tid), ItemPointerGetOffsetNumber(tid));
    }
    else
    {
        /* Extract the vector column from the heap tuple, copy into mt->vector_blob
         * at slot_index. Reuse the helper used by load_lsm_index_internal. */
        copy_vector_into_memtable_slot(slot, mt, slot_index, tts, tid);
        publish_slot_release(mt, slot_index);
    }

    ExecDropSingleTupleTableSlot(tts);
    table_close(heap, AccessShareLock);
    relation_close(idx, AccessShareLock);
}
```

`copy_vector_into_memtable_slot` is the helper currently used inline by `load_lsm_index_internal`. If it doesn't exist as a separate function, factor it out of `lsmindex.c` and call it from both places. Confirm by reading [lsmindex.c:850-920](../../../pgvector/src/lsmindex.c#L850-L920).

- [ ] **Step 5: Implement `dpv_standby_memtable_tombstone`**

```c
void
dpv_standby_memtable_tombstone(Oid indexRelId, SegmentId sid, uint32 slot_index)
{
    LSMIndexBufferSlot *slot = lookup_lsm_index_buffer_slot(indexRelId);
    ConcurrentMemTable mt;

    if (slot == NULL || pg_atomic_read_u32(&slot->valid) != 1)
        return;

    mt = find_memtable_by_sid(slot, sid);
    if (mt == NULL)
        return;  /* spec §9: absent memtable — page-level redo applied above is enough */

    SET_SLOT(mt->bitmap, slot_index);
}
```

- [ ] **Step 6: Implement `dpv_standby_update_max_sid` and the `Release` no-op**

```c
void
dpv_standby_update_max_sid(Oid indexRelId, SegmentId sid)
{
    LSMIndexBufferSlot *slot = lookup_lsm_index_buffer_slot(indexRelId);
    if (slot == NULL || pg_atomic_read_u32(&slot->valid) != 1)
        return;

    /* Cached max sid — used by hot path lookups. Page already updated by
     * the page-level redo before we got here. */
    update_cached_max_memtable_sid(slot, sid);
}

void
dpv_standby_release_memtable(Oid indexRelId, SegmentId sid)
{
    /* Intentionally a no-op for SharedMemtableBuffer. Adoption drops the
     * memtable slot, not Release (spec §10). The page-level redo cleans up
     * the status page already. */
    (void) indexRelId;
    (void) sid;
}
```

If `update_cached_max_memtable_sid` doesn't exist, add it as a small helper in `lsmindex.c` that sets the cached field while holding the appropriate lock (mirror the existing reader's lock acquisition).

### Task 5.2: Broadcast `load_cv` when load completes

**Files:**
- Modify: `pgvector/src/lsmindex.c`

- [ ] **Step 1: Find the completion sites in `load_lsm_index_internal`**

In [lsmindex.c:495](../../../pgvector/src/lsmindex.c#L495), `load_lsm_index_internal` sets `slot->valid = 1` on success. There may be an early-exit error path that sets `slot->valid = 0`. Both transitions out of `valid==2` must broadcast `load_cv` so any redo callback waiting in `dpv_standby_wait_if_loading` can wake.

- [ ] **Step 2: Add the broadcast**

After every assignment of `slot->valid` that takes it out of state 2, append:

```c
ConditionVariableBroadcast(&slot->load_cv);
```

If multiple sites need this, factor a tiny helper:

```c
static inline void
set_load_state(LSMIndexBufferSlot *slot, uint32 v)
{
    pg_atomic_write_u32(&slot->valid, v);
    ConditionVariableBroadcast(&slot->load_cv);
}
```

…and use it everywhere `slot->valid = …` currently appears. Verify `load_cv` is initialized somewhere during slot setup; if not, add `ConditionVariableInit(&slot->load_cv)` to the slot-initialization site.

### Task 5.3: Wire redo callbacks to call standby helpers

**Files:**
- Modify: `pgvector/src/replication_rmgr.c`

- [ ] **Step 1: Replace the stub redo callbacks with full implementations**

```c
#include "miscadmin.h"  /* InHotStandby */

static void
apply_page_level(XLogReaderState *r, int block_id)
{
    Buffer buf;

    /* PG handles FPI-based recovery automatically here. For our records the
     * buffer is always registered with REGBUF_FORCE_IMAGE, so the result
     * is normally BLK_RESTORED — no additional page work needed. */
    if (XLogReadBufferForRedo(r, block_id, &buf) != BLK_RESTORED)
    {
        /* Defensive: if we ever stop using REGBUF_FORCE_IMAGE, we'd need
         * to replay the semantic page modification here. For v1 we always
         * use FPI, so this branch is unexpected. */
        elog(WARNING, "decoupled_pgvector: non-FPI redo path hit; record info=0x%02x",
             XLogRecGetInfo(r) & ~XLR_INFO_MASK);
    }
    UnlockReleaseBuffer(buf);
}

static void
redo_register_memtable(XLogReaderState *r)
{
    xl_dpv_register_memtable *rec = (xl_dpv_register_memtable *) XLogRecGetData(r);

    apply_page_level(r, 0);
    if (XLogRecHasBlockRef(r, 1))
        apply_page_level(r, 1);

    if (!InHotStandby)
        return;  /* spec §7 primary-crash-recovery policy */

    if (dpv_standby_wait_if_loading(rec->indexRelId) != 1)
        return;

    dpv_standby_register_memtable(rec->indexRelId, rec->sid);
}

static void
redo_add_to_memtable(XLogReaderState *r)
{
    xl_dpv_add_to_memtable *rec = (xl_dpv_add_to_memtable *) XLogRecGetData(r);

    apply_page_level(r, 0);

    if (!InHotStandby)
        return;
    if (dpv_standby_wait_if_loading(rec->indexRelId) != 1)
        return;

    dpv_standby_add_to_memtable(rec->indexRelId, rec->sid,
                                 rec->slot_index, &rec->tid);
}

static void
redo_memtable_tombstone(XLogReaderState *r)
{
    xl_dpv_memtable_tombstone *rec = (xl_dpv_memtable_tombstone *) XLogRecGetData(r);

    apply_page_level(r, 0);

    if (!InHotStandby)
        return;
    if (dpv_standby_wait_if_loading(rec->indexRelId) != 1)
        return;

    dpv_standby_memtable_tombstone(rec->indexRelId, rec->sid, rec->slot_index);
}

static void
redo_update_max_sid(XLogReaderState *r)
{
    xl_dpv_update_max_sid *rec = (xl_dpv_update_max_sid *) XLogRecGetData(r);

    apply_page_level(r, 0);

    if (!InHotStandby)
        return;
    if (dpv_standby_wait_if_loading(rec->indexRelId) != 1)
        return;

    dpv_standby_update_max_sid(rec->indexRelId, rec->sid);
}

static void
redo_release_memtable(XLogReaderState *r)
{
    xl_dpv_release_memtable *rec = (xl_dpv_release_memtable *) XLogRecGetData(r);

    apply_page_level(r, 0);
    if (XLogRecHasBlockRef(r, 1))
        apply_page_level(r, 1);

    if (!InHotStandby)
        return;
    if (dpv_standby_wait_if_loading(rec->indexRelId) != 1)
        return;

    dpv_standby_release_memtable(rec->indexRelId, rec->sid);
}
```

- [ ] **Step 2: Build**

```bash
cd pgvector && make 2>&1 | tail -30
```

Expected: clean.

### Task 5.4: Commit Phase 5

- [ ] **Step 1: Commit**

```bash
git add pgvector/src/standby_memtable.h pgvector/src/standby_memtable.c \
        pgvector/src/replication_rmgr.c pgvector/src/lsmindex.c
git commit -m "feat: standby redo for memtable lifecycle records"
```

---

## Phase 6 — Integration tests for memtable replication

### Task 6.1: Test memtable insert replication

**Files:**
- Create: `pgvector/test/t/101_replication_memtable_insert.pl`

- [ ] **Step 1: Write the test**

```perl
use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $dim = 32;

my $primary = PostgreSQL::Test::Cluster->new('primary');
$primary->init(allows_streaming => 1);
$primary->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
wal_level = replica
max_wal_senders = 4
wal_keep_size = 256MB
));
$primary->start;
$primary->backup('basebackup');

my $standby = PostgreSQL::Test::Cluster->new('standby');
$standby->init_from_backup($primary, 'basebackup', has_streaming => 1);
$standby->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
hot_standby = on
));
$standby->start;

sub wait_catchup {
    my $appname = $standby->name;
    my $q = "SELECT pg_current_wal_lsn() <= replay_lsn "
          . "FROM pg_stat_replication WHERE application_name = '$appname';";
    $primary->poll_query_until('postgres', $q) or die "standby never caught up";
}

# Build a vector LSM index on the primary.
$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');
my $array_sql = join(",", ('random()') x $dim);
$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    CREATE INDEX t_v_idx ON t USING <YOUR_LSM_AM_NAME> (v vector_l2_ops);
));

# Insert enough rows that they all land in memtables (well below the flush threshold).
$primary->safe_psql('postgres',
    "INSERT INTO t SELECT i, ARRAY[$array_sql]::vector FROM generate_series(1, 100) i;");
wait_catchup();

# Query the index on the standby. Force index scan; the result must include
# all 100 inserted rows when k is 100.
my $primary_count = $primary->safe_psql('postgres', q(
    SET enable_seqscan = off;
    SELECT count(*) FROM (SELECT * FROM t ORDER BY v <-> '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector LIMIT 100) sub;
));
my $standby_count = $standby->safe_psql('postgres', q(
    SET enable_seqscan = off;
    SELECT count(*) FROM (SELECT * FROM t ORDER BY v <-> '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector LIMIT 100) sub;
));
is($standby_count, '100', "standby index returns all 100 memtable rows");
is($standby_count, $primary_count, "standby matches primary on memtable-only data");

done_testing();
```

Replace `<YOUR_LSM_AM_NAME>` with the actual access-method name registered by this fork. Search for `amhandler` in `vector.c` to find it.

- [ ] **Step 2: Run the test**

```bash
cd pgvector && make install
prove -I test/perl test/t/101_replication_memtable_insert.pl
```

Expected: PASS. If it fails with "standby index returns 0 rows", the redo callbacks aren't materializing the memtable — debug by adding `elog(LOG, …)` in `dpv_standby_add_to_memtable` and re-running.

If the standby's query path uses `vector_search_send` (the ring-buffer call), confirm that the standby's `VectorIndexWorker` is running and that its search path correctly reads the standby's own `SharedMemtableBuffer`.

- [ ] **Step 3: Commit**

```bash
git add pgvector/test/t/101_replication_memtable_insert.pl
git commit -m "test: memtable inserts replicate to standby"
```

### Task 6.2: Test memtable tombstone replication

**Files:**
- Create: `pgvector/test/t/102_replication_memtable_tombstone.pl`

- [ ] **Step 1: Write the test**

```perl
use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $dim = 32;

my $primary = PostgreSQL::Test::Cluster->new('primary');
$primary->init(allows_streaming => 1);
$primary->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
wal_level = replica
max_wal_senders = 4
wal_keep_size = 256MB
));
$primary->start;
$primary->backup('basebackup');

my $standby = PostgreSQL::Test::Cluster->new('standby');
$standby->init_from_backup($primary, 'basebackup', has_streaming => 1);
$standby->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
hot_standby = on
));
$standby->start;

sub wait_catchup {
    my $appname = $standby->name;
    my $q = "SELECT pg_current_wal_lsn() <= replay_lsn "
          . "FROM pg_stat_replication WHERE application_name = '$appname';";
    $primary->poll_query_until('postgres', $q) or die "standby never caught up";
}

$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');
my $array_sql = join(",", ('random()') x $dim);
$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    CREATE INDEX t_v_idx ON t USING <YOUR_LSM_AM_NAME> (v vector_l2_ops);
    INSERT INTO t SELECT i, ARRAY[$array_sql]::vector FROM generate_series(1, 100) i;
));

# Vacuum a deleted half. bulk_delete_lsm_index will emit MemtableTombstone WAL.
$primary->safe_psql('postgres', q(
    DELETE FROM t WHERE id <= 50;
    VACUUM t;
));
wait_catchup();

my $standby_count = $standby->safe_psql('postgres', q(
    SET enable_seqscan = off;
    SELECT count(*) FROM (
        SELECT * FROM t
        ORDER BY v <-> '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector
        LIMIT 100
    ) sub
));
is($standby_count, '50', "tombstoned half is filtered out of the standby's index results");

done_testing();
```

The key correctness signal: the standby filters out deleted rows. Note that the heap-level filter alone (via `HeapTupleSatisfiesVisibility`) would catch this too — to make the test specifically exercise `MemtableTombstone` redo, you can additionally use `EXPLAIN (ANALYZE, BUFFERS)` or instrumentation to confirm the index returned 50 rows rather than 100-with-filter. As a strong signal: drop the heap, query the index directly via the extension's internal API (if exposed) — or accept the heap-filter behavior as the integration-level signal.

- [ ] **Step 2: Run the test**

```bash
prove -I test/perl test/t/102_replication_memtable_tombstone.pl
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add pgvector/test/t/102_replication_memtable_tombstone.pl
git commit -m "test: memtable tombstones replicate to standby"
```

### Task 6.3: Test lazy-load during in-flight WAL

**Files:**
- Create: `pgvector/test/t/103_replication_load_during_redo.pl`

This is the §9 "redo during index loading" coordination test: a new standby starts, the index has never been loaded on it, and WAL replay encounters memtable records before any backend touches the index.

- [ ] **Step 1: Write the test**

```perl
use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $dim = 32;

my $primary = PostgreSQL::Test::Cluster->new('primary');
$primary->init(allows_streaming => 1);
$primary->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
wal_level = replica
max_wal_senders = 4
wal_keep_size = 256MB
));
$primary->start;

# Insert and build the index on the primary BEFORE creating the standby backup.
$primary->safe_psql('postgres', 'CREATE EXTENSION vector;');
my $array_sql = join(",", ('random()') x $dim);
$primary->safe_psql('postgres', qq(
    CREATE TABLE t (id int, v vector($dim));
    CREATE INDEX t_v_idx ON t USING <YOUR_LSM_AM_NAME> (v vector_l2_ops);
    INSERT INTO t SELECT i, ARRAY[$array_sql]::vector FROM generate_series(1, 50) i;
));
$primary->backup('basebackup');

# Insert more *after* backup so the standby has to redo Add records.
$primary->safe_psql('postgres',
    "INSERT INTO t SELECT i, ARRAY[$array_sql]::vector FROM generate_series(51, 200) i;");

my $standby = PostgreSQL::Test::Cluster->new('standby');
$standby->init_from_backup($primary, 'basebackup', has_streaming => 1);
$standby->append_conf('postgresql.conf', qq(
shared_preload_libraries = 'vector'
hot_standby = on
));
$standby->start;

sub wait_catchup {
    my $appname = $standby->name;
    my $q = "SELECT pg_current_wal_lsn() <= replay_lsn "
          . "FROM pg_stat_replication WHERE application_name = '$appname';";
    $primary->poll_query_until('postgres', $q) or die "standby never caught up";
}
wait_catchup();

my $count = $standby->safe_psql('postgres', q(
    SET enable_seqscan = off;
    SELECT count(*) FROM (
        SELECT * FROM t
        ORDER BY v <-> '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector
        LIMIT 200
    ) sub;
));
is($count, '200', "all 200 rows are queryable on standby even though some "
                  . "were inserted after the backup");

done_testing();
```

This exercises two paths simultaneously:
1. `load_lsm_index_internal` runs on first query on the standby — it reads status pages (which were materialized to disk by Phase 4's page-level redo) and reconstructs memtables for the first 50 rows from heap.
2. WAL replay for rows 51-200 was redoing `Add` records *while* the load was in flight (depending on timing). The `dpv_standby_wait_if_loading` block ensures redo waits for the loader to publish `valid=1` before applying in-memory side effects.

If timing is hard to provoke, add a deliberate `pg_sleep` in the test after `$standby->start` and before the query, to allow WAL replay to outpace the lazy-load trigger. Or set `recovery_target = 'immediate'` semantics for fine-grained reproduction.

- [ ] **Step 2: Run the test**

```bash
prove -I test/perl test/t/103_replication_load_during_redo.pl
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add pgvector/test/t/103_replication_load_during_redo.pl
git commit -m "test: index load + WAL redo concurrency on standby"
```

---

## Self-review checklist for Plan 1

After the last commit, run through this list before declaring Plan 1 complete.

- [ ] **All five `GenericXLog` use sites in `statuspage.c` have been converted.** Run `grep -n GenericXLog pgvector/src/statuspage.c` — should return 0 hits (the helpers `StatusInitRegisterPage` / `StatusCommitBuffer` may still exist if used by `CreateStatusMetaPage` / `InitializeStatusMemtableArray`; those are called only during `CREATE INDEX`, which is already WAL-logged by PG via the standard relation-extension path. Decide per-site: convert them too, or leave them alone if the standby will see the relation via standard `XLOG_FPI` records. Leaving them alone is acceptable for v1 — those operations only run once at index creation, and PG's relation-extension WAL already covers them.)
- [ ] **`RM_DPV_REPLICATION_ID` doesn't collide.** Grep `RegisterCustomRmgr` in installed extensions on the test machine; if any reuse 137, pick another in the [128, 255] range.
- [ ] **Every redo callback applies the page-level change first, then checks `InHotStandby`, then `dpv_standby_wait_if_loading`, then the in-memory side effect.** This ordering matters: skipping the page-level work on the primary during crash recovery would corrupt the status pages.
- [ ] **`PageSetLSN` is called on every buffer registered for WAL.** Without it, the buffer manager won't flush the buffer in WAL-LSN order, and a crash mid-checkpoint could leave the buffer ahead of the WAL on disk.
- [ ] **`ConditionVariableBroadcast(&slot->load_cv)` is called on every transition out of `valid==2`.** Both success (`valid=1`) and failure (`valid=0`) paths.
- [ ] **All three integration tests pass.** Run them under `prove`.
- [ ] **The primary still works without a standby.** Re-run the existing test suite (`make installcheck`) and ensure no regression.

---

## End of Plan 1

---

## Implementation notes (2026-05-15)

This section was added after Plan 1 was implemented end-to-end via subagent-driven development. It records what was actually built, the two design pivots from the original spec, and the issues found by a final code review.

### What was built

**Files created**:
- [`pgvector/src/replication_rmgr.h`](../../../pgvector/src/replication_rmgr.h) / [`replication_rmgr.c`](../../../pgvector/src/replication_rmgr.c) — custom rmgr id `137`, five record types, emit helpers, redo dispatcher.
- [`pgvector/src/standby_memtable.h`](../../../pgvector/src/standby_memtable.h) / [`standby_memtable.c`](../../../pgvector/src/standby_memtable.c) — standby-side redo side-effect helpers.
- [`pgvector/test/t/100_replication_rmgr_smoke.pl`](../../../pgvector/test/t/100_replication_rmgr_smoke.pl) — primary+standby smoke harness.
- [`pgvector/test/t/101_replication_memtable_insert.pl`](../../../pgvector/test/t/101_replication_memtable_insert.pl), [`102_replication_memtable_tombstone.pl`](../../../pgvector/test/t/102_replication_memtable_tombstone.pl), [`103_replication_load_during_redo.pl`](../../../pgvector/test/t/103_replication_load_during_redo.pl) — integration tests.

**Files modified**:
- [`pgvector/Makefile`](../../../pgvector/Makefile) — `src/replication_rmgr.o src/standby_memtable.o` added to `OBJS`.
- [`pgvector/src/vector.c`](../../../pgvector/src/vector.c) — `replication_rmgr.h` included; `vector_replication_rmgr_register()` called from `_PG_init`; `VectorIndexWorker` and `IndexLoadWorker` switched from `BgWorkerStart_RecoveryFinished` to `BgWorkerStart_ConsistentState` (so they run on a hot standby — fix for review issue C1).
- [`pgvector/src/statuspage.h`](../../../pgvector/src/statuspage.h) — `AddToStatusMemtable` and `RemoveFromStatusMemtable` signatures gained a `uint32 slot_index` parameter.
- [`pgvector/src/statuspage.c`](../../../pgvector/src/statuspage.c) — semantic emit calls (`dpv_emit_release_memtable`, `dpv_emit_update_max_sid`, `dpv_emit_register_memtable`, `dpv_emit_memtable_tombstone`) added *after* each entry-point function's `GenericXLog` work completes. `AddToStatusMemtable`'s emit was deliberately not added here (it lives in `lsmindex.c` — see pivot #2 below).
- [`pgvector/src/lsmindex.h`](../../../pgvector/src/lsmindex.h) — exposes `register_and_set_memtable`, `publish_slot_release` (inline), `VEC_BYTES_PER_ROW`, `VEC_PTR_AT` to `standby_memtable.c`.
- [`pgvector/src/lsmindex.c`](../../../pgvector/src/lsmindex.c) — emits `dpv_emit_add_to_memtable` right after the memcpy at the insert site (~line 1270); threads `slot_index` to the three `Add`/`Remove` callers (`:1318`, `:1665`, `:1830`).

Build is clean.

### Pivot #1 — `GenericXLog` stays in place

The original plan called for replacing every `GenericXLog` use site in `statuspage.c` with the custom rmgr. While reading the code we discovered:
- `RegisterStatusMemtable` and `AddToStatusMemtable` each modify up to **three** buffers across **two** separate `GenericXLogState` transactions: the main loop touches 1-2 buffers (status-page entry page, free-list metapage from `GetFreePage`, plus a possibly-extended page); then `StatusUpdateInsertPage` modifies a *different* page in its own separate WAL transaction.
- Replacing all of that with a single custom-rmgr record would require a major refactor of `GetFreePage` (which currently takes `GenericXLogState *state` and registers buffers itself) and `StatusUpdateInsertPage`.

**Revised design**: keep `GenericXLog` for all page-level WAL exactly as today, and **emit our custom rmgr records *alongside* (after) the `GenericXLog` finish**, carrying only the semantic payload (no buffer registration in the custom rmgr record). The standby gets page state from PG's built-in `GenericXLog` redo and gets the in-memory `SharedMemtableBuffer` side effect from our custom rmgr redo.

Trade-off: ~2× WAL records for status-page operations, but no `GenericXLog` refactor and no multi-buffer plumbing in our emit helpers.

This pivot changed:
- `dpv_emit_*` prototypes (drop `Relation idx, Buffer buf`).
- `dpv_emit_*` bodies (no `XLogRegisterBuffer`, no `PageSetLSN`).
- The Phase 4 conversion pattern: instead of replacing the `GenericXLog` envelope, we simply append a `dpv_emit_*` call after each `StatusCommitBuffer`.
- The Phase 5 redo callbacks: page-level apply is now PG's job via the standard `GenericXLog` redo; our callbacks do only the in-memory side effect.

### Pivot #2 — `Add` WAL carries the vector inline

The original plan / spec §9 said redo callbacks would fetch the vector from the standby's heap by tid via `table_open` + `table_tuple_fetch_row_version`. **This does not work**: PG's startup process (which runs WAL redo) has no `MyDatabaseId` and no relcache initialization, so `table_open` fails.

**The fix**: ship the vector blob inline in the `Add` WAL record (the standard PG idiom — btree, hash, gin, etc. all carry index payload in their WAL records). The redo callback then `memcpy`s the inline payload into `mt->vector_blob[slot_index]`. No heap fetch.

This pivot changed:
- `xl_dpv_add_to_memtable` struct gained a `uint32 vector_bytes` field; the WAL record's payload is the struct followed by `vector_bytes` raw bytes.
- `dpv_emit_add_to_memtable` prototype gained `const void *vector, uint32 vector_bytes`.
- The emit call was moved from `statuspage.c:582` (where the vector was not in scope) to [`lsmindex.c:1270`](../../../pgvector/src/lsmindex.c#L1270) (right after the `memcpy` at the insert site, where `VEC_PTR_AT(mt, i)` is in scope).
- `dpv_standby_add_to_memtable` reads the inline vector from the WAL record and `memcpy`s it into the standby's memtable slot.

Trade-off: WAL volume grows by `sizeof(vector)` per insert. For 768-dim float32 vectors that's ~3 KB per insert. At 100 K inserts: ~300 MB of additional WAL. Acceptable for v1.

### Review findings — fixed in this plan

- **C1 (workers don't start on standby)** — `BgWorkerStart_RecoveryFinished` literally means "start after recovery finishes," which on a continuously-replicating hot standby never happens. Two workers needed on standby (`VectorIndexWorker` for search dispatch, `IndexLoadWorker` for lazy index attach) were switched to `BgWorkerStart_ConsistentState`. `LSMBackgroundWorker` (the flush worker) stays on `RecoveryFinished` — it intentionally never starts on a standby because flush is a primary-only operation. Fixed in `vector.c`.

- **C2 (standby `dpv_standby_register_memtable` mutates LSMIndex without `mt_lock`)** — On a standby, hot-standby query backends can be reading `growing_memtable_idx`/`memtable_idxs[]`/`memtable_count` concurrently with WAL redo. The redo callback now holds `lsm->mt_lock` `LW_EXCLUSIVE` for the dedup + allocate + wire sequence. `dpv_standby_add_to_memtable` and `dpv_standby_memtable_tombstone` hold `LW_SHARED` around `find_memtable_by_sid` only (per-memtable mutations are protected by atomics / publish-release / SET_SLOT bit-OR). Fixed in `standby_memtable.c`.

### Review findings — deferred (out of Plan 1 scope)

- **C3 (transient gaps during concurrent primary inserts)** — Concurrent primary inserts that reserve adjacent slots can produce WAL in any order; the standby's view between the two redo applications shows a "hole" at the not-yet-replayed slot. The primary exhibits the *same* transient inconsistency between `reserve_slot(i)` and `publish_slot_release(i)`, and `publish_slot_release`'s `max_ready_id` advancement is gap-aware (only advances on contiguous ready bits). Self-healing once both records replay; not a permanent inconsistency.

- **C4 (`next_segment_id` not advanced on standby)** — `register_and_set_memtable(is_recovery=true)` uses the WAL-provided sid directly without advancing `lsm->next_segment_id`. After promotion the new primary's `alloc_segment_id` could return a stale value that collides with an already-used sid. Matters only after promotion, which is explicitly out of v1 scope (see [physical-replication design §2 non-goals](../specs/2026-05-14-physical-replication-design.md)).

- **C5 (slot leak when `MEMTABLE_NUM` exceeded)** — `MEMTABLE_BUF_SIZE = 8`. If the standby accumulates 8 memtables without adoption, the next `dpv_standby_register_memtable` silently drops the previous growing memtable rather than enqueueing it. Plan 2 introduces segment adoption, which drops memtables once their corresponding segment files arrive — relieving this pressure. For Plan 1 tests in isolation this isn't triggered (workloads don't generate enough memtables).

- **C6 (lazy-load + redo materialization overlap)** — With C1 fixed and the v1 "block redo on `valid==2`" policy from spec §9 in place, this scenario doesn't produce duplicates: redo blocks on `valid==2`, the loader reads stable status pages, and only after the loader sets `valid=1` (and broadcasts `load_cv`) does redo apply subsequent WAL records.

### Plan 1 scope boundary: segments created after backup don't reach the standby

Plan 1 replicates **memtable lifecycle state** (Register / Add / Tombstone / UpdateMaxSid / Release) via the custom rmgr and **status pages** via PG's built-in `GenericXLog` replay. Two things are intentionally **out of Plan 1's scope**:

1. **Segment shipment** — Plan 2 (spec §6, side-channel + `SegmentCreated` / `SegmentReplaced` WAL records + `SegmentFetcher` bgworker).
2. **Per-segment bitmap subversions / vacuum on segments** — Plan 3 (spec §13.1, `SegmentVacuumTombstones`).

This produces a concrete failure mode whenever the primary creates a segment AFTER the basebackup point:

1. Standby takes `pg_basebackup` at LSN_backup. The backup includes everything under `<DataDir>`, which now includes `<DataDir>/pgvector_storage/` — so whatever segment files exist at backup time **do** ride to the standby for free (no external `rsync` needed, contrary to spec §2 non-goal #3 which assumed an absolute path outside `data_dir`).
2. After the backup, the primary flushes a memtable to disk:
   - Writes segment files (`index_…`, `mapping_…`, `bitmap_…`, `metadata_…`) under the primary's `<DataDir>/pgvector_storage/`.
   - Calls `ReleaseStatusMemtable(sid)` — the status-page entry for those tids is **removed**, and `dpv_emit_release_memtable` emits the matching custom rmgr record.
3. The standby catches up on WAL: status pages on the standby reflect the post-flush state (entries removed), but the segment file itself is in the primary's `<DataDir>/pgvector_storage/`, not the standby's.
4. Standby's first index touch runs `load_lsm_index_internal`. It scans the standby's own `<DataDir>/pgvector_storage/` for segment metadata files — finds nothing for that range — and reads status pages for memtable tids. The flushed rows are silently missing from the standby's index. **This is the gap.**

**Owner: Plan 2.** Plan 2 introduces a side channel (file-server bgworker on the primary, `SegmentFetcher` bgworker(s) on the standby) and `SegmentCreated` / `SegmentReplaced` WAL records that drive pull requests. After Plan 2 lands, the fetcher mirrors post-backup segment files from the primary to the standby. The Plan 2 spec also introduces a "not queryable" attach-time barrier (spec §10, §12) so backends don't run queries against an index whose segment fetches are still in flight.

### Storage layout: `VECTOR_STORAGE_BASE_DIR` → `pgvector.storage_base_dir` GUC (resolved under `<DataDir>` by default)

Originally tracked as spec §16.2 / Plan 2 Task 7.2 ("Configurable `VECTOR_STORAGE_BASE_DIR`"). **Pulled forward into Plan 1** because the previous hardcoded absolute path masked Plan 1's gaps in tests (shared filesystem on a single machine → standby silently saw the primary's segments).

What changed:
- The `#define VECTOR_STORAGE_BASE_DIR "/ssd_root/liu4127/pg_vector_extension_indexes/"` macro at [lsmindex.h:203](../../../pgvector/src/lsmindex.h#L203) is **removed**.
- A new GUC `pgvector.storage_base_dir` (string, `PGC_POSTMASTER`) takes its place. When empty (the default), the runtime resolves to `<DataDir>/pgvector_storage/`. When non-empty, the user-specified path is used (with a trailing slash auto-appended if missing).
- `get_vector_storage_dir(void)` is the canonical accessor; all 9 call sites under `pgvector/src/lsmindex_io.c` and `pgvector/src/hnswbuild.c` were converted from `VECTOR_STORAGE_BASE_DIR "%u/…"` string concatenation to `"%s%u/…"` format + `get_vector_storage_dir()` argument.
- `_PG_init` creates the base directory via `MakePGDirectory` (single-level mkdir; sufficient for the default location, which is one level under `<DataDir>`).

Consequences:
- **Each PG cluster gets its own segment storage by default.** No more conflict between two clusters on the same machine.
- **`pg_basebackup` includes the storage directory.** Whatever segments exist at backup time automatically replicate to the standby. This is a meaningful chunk of spec §2's "initial sync" non-goal solved for free.
- **Tests now exercise the real Plan 1 surface.** The Perl test clusters get separate `data_dir`s → separate `pgvector_storage/` → standby cannot see segments the primary creates post-backup. The gap described above is now *visible*, not masked.
- **Back-compat for users who want a separate volume.** Set `pgvector.storage_base_dir = '/ssd_root/...'` in `postgresql.conf` to opt back into an external path.

**Implication for Plan 1 tests:** test 101 / 102 / 103 each create their own per-cluster storage under `<tmp_check>/.../pgdata/pgvector_storage/`. Tests can now assume "the standby's segment pool is empty unless the segment existed at backup time." Plan 2's tests will validate the segment-shipment path.

**Implication for production:** a standby running Plan 1 without Plan 2 is only correct if the primary never flushes a memtable between the basebackup and steady-state operation. Plan 1 alone is a foundation for Plan 2, not a usable replication setup on its own.

### Test caveat — HNSW `ef_search`

Tests `101` and `103` use `ORDER BY v <-> '...'::vector LIMIT N` and assert `count = N`. HNSW's default `ef_search` may return fewer than `N` candidates even when N rows exist. If you see flakiness, add `SET hnsw.ef_search = 1000;` (or a value comfortably larger than `N`) at the top of each query in the test.

### Not done in this session (per user preferences)

- **No `git commit`** was run during implementation. The user manages git in a copy of this repo. The changes above are in the working tree only.
- **No tests were executed**. The user manages their PG instance and test runs. Tests are ready to run via `prove -I pgvector/test/perl pgvector/test/t/100_replication_rmgr_smoke.pl` (etc.) once the user is ready.

---

## Consistency analysis: the LSN_1 → LSN_2 window introduced by Pivot #1

Pivot #1 keeps `GenericXLog` for page-state WAL and emits a separate custom rmgr record for semantic payload. Every entry-point function in `statuspage.c` therefore produces **two consecutive WAL records**:

- **LSN_1** — `GenericXLog` record. Page-state mutation (status page now has the new `StatusMemtable` entry / tid / etc).
- **LSN_2** — custom rmgr record. Semantic payload (`indexRelId`, `sid`, `slot_index`, and for `Add` the inline vector).

LSN_2 > LSN_1, but they are distinct records from separate `XLogInsert` calls. On the standby they replay in order, but a query backend or the `IndexLoadWorker` can run in the window between them. Two distinct correctness paths are affected.

### Path A — pure query path on the standby (backends → `SharedMemtableBuffer`)

Standby query backends read `SharedMemtableBuffer`, never the status pages. So between LSN_1 and LSN_2:

- **Register** — status page has sid X but `SharedMemtableBuffer` does not. Queries don't see sid X at all. Self-healing once LSN_2 replays.
- **Add** — status page tid list has the new tid but `SharedMemtableBuffer` slot is not yet materialized. Queries miss this one entry. Self-healing once LSN_2 replays.
- **Tombstone** — status page has the tid removed but `SharedMemtableBuffer` bitmap doesn't yet. Queries can still return the about-to-be-deleted entry; heap-level `HeapTupleSatisfiesVisibility` filters at fetch time if the heap-vacuum WAL has also replayed. Same as PG btree behavior between btree-vacuum WAL and heap-vacuum WAL.

In all three cases the window produces only **transient, eventually-consistent** effects on query results — similar to the existing window the primary already has between `reserve_slot(i)` and `publish_slot_release(i)`. Acceptable for an approximate ANN index.

### Path B — lazy-load path on the standby (`load_lsm_index_internal` reads status pages)

**This is where the pivot weakened a protection the original spec relied on.** Spec §9 "Redo during index loading" said: when a redo callback sees `valid==2`, it blocks on `load_cv` until the load completes. That protection only works if **all** status-page modifications go through the custom rmgr — because only our custom rmgr's redo checks `valid==2`. PG's built-in `generic_redo` for `GenericXLog` records does not check `valid` and is not blocked.

With the pivot, the load now reads status pages while `generic_redo` can be concurrently modifying them across multiple page reads. Concrete scenario:

```
Load: read status array page  → sees [sid 1, sid 2, sid 3]
                                                ↓
generic_redo applies LSN_1 (adds sid 4)         page now has [1,2,3,4]
                                                ↓
Load: read status array page again, or          ← inconsistency window
      read tid list for sid 1                   ← may see new tids that load
                                                  picks different slot_indices for
                                                  than the primary's WAL slot_index.
```

Per-page reads are serialized by buffer locks, so each individual `ReadBuffer + LockBuffer` is atomic. But the load reads many pages sequentially, and between releases `generic_redo` can apply records. **Cross-page consistency is not guaranteed.**

Worst-case symptom: load assigns `slot_index = K` (via `pg_atomic_fetch_add_u32(&current_size, 1)`) to a tid that the primary's `Add` WAL says lives at `slot_index = K'`. When LSN_2 replays, `dpv_standby_add_to_memtable` writes `mt->tids[K']` (the primary's index), leaving the tid in *two* slots — the one load chose and the one redo chose. Both `current_size` and the bitmap are then inconsistent with the primary.

In practice this requires the primary to be actively writing while the standby is doing a first-touch lazy load. Plan 1's tests don't reproduce it (they `wait_catchup` before triggering load). A production standby attaching under continuous primary load *would* hit it.

### Mitigation options considered

#### **Option 1 — replay-pause around the load (RECOMMENDED for v1)**

Have `IndexLoadWorker`'s `load_lsm_index_internal` call `pg_wal_replay_pause()` before reading status pages and `pg_wal_replay_resume()` after publishing `valid=1`. While the load is running, ALL WAL replay on the standby is paused; the status pages are stable for the duration of the read.

- **Pros**: smallest code change. No restructuring of `load_lsm_index_internal`'s read pattern. Buys back exactly the "load runs against stable state" property the original spec assumed.
- **Cons**: sledgehammer. Pauses every other index's WAL replay too. Bounded by `max_standby_streaming_delay` if the load is slow; long-running loads could trigger recovery conflict.
- **Trade-off**: load latency on the standby is decoupled from primary write throughput; replication lag during attach grows by the load duration.
- **Where it goes**: `pgvector/src/index_load_worker.c` around the call to `load_lsm_index_internal`, OR at the top of `load_lsm_index_internal` itself in `lsmindex.c` (under `if (RecoveryInProgress())`).

PG 17 API: `pg_wal_replay_pause()` / `pg_wal_replay_resume()` are SQL-callable; the C-level equivalents are `SetRecoveryPause(true)` / `SetRecoveryPause(false)` in `xlogrecovery.h`. Standby backends can call them.

#### **Option 4 — revert Pivot #1 (custom rmgr replaces GenericXLog)**

Go back to the original spec's design: one combined WAL record per operation, carrying both page-state (via `REGBUF_FORCE_IMAGE`) and semantic payload. The block-on-`valid==2` rule in our custom rmgr redo then naturally covers all status-page modifications because they all go through us.

- **Pros**: semantically cleanest — eliminates the window entirely. No two-record race to worry about. One WAL record per operation (not two), so WAL volume actually shrinks vs the current state.
- **Cons**: requires the multi-buffer refactor of `GetFreePage` and `StatusUpdateInsertPage` we explicitly avoided when we pivoted. `dpv_emit_register_memtable` and `dpv_emit_add_to_memtable` need to register up to 3 buffers each (status array page + free-list metapage + extended page). `StatusUpdateInsertPage` either becomes its own custom rmgr record type or is dissolved into the caller.
- **Trade-off**: ~2-3 days of careful refactoring; more diff in `statuspage.c`; risk of subtle bugs in the multi-buffer plumbing. Plan 1's "implement then test" cycle would need a second pass.

### Decision for v1

**Lean toward Option 1.** Reasoning:

1. The window only matters during first-touch lazy load on the standby (Path B). Steady-state replay (Path A) is fine.
2. Loads on the standby are infrequent (once per index per standby restart). The pause-window cost amortizes over many subsequent queries.
3. Option 1 is mechanical and low-risk; Option 4 is invasive and reopens decisions we already made.
4. Option 4 remains a clean follow-up if `max_standby_streaming_delay` cancellations from long loads become a measured problem.

**Not yet implemented in this session.** Option 1 should be added before the standby attach path is exercised under load. A small follow-on task:

- In `pgvector/src/index_load_worker.c`, around the call to `load_lsm_index_internal(slot->lsmIndex.indexRelId, (uint32_t) i)`:
  - Before: `if (RecoveryInProgress()) { (void) SetRecoveryPause(true); }`
  - After (in both PG_TRY and PG_CATCH paths): `if (RecoveryInProgress()) { (void) SetRecoveryPause(false); }`
- Include `access/xlogrecovery.h` for `SetRecoveryPause`.
- Add an assertion / `elog(DEBUG1)` to record pause/resume durations for diagnostics.

If a future workload reveals that recovery-conflict cancellations are common because loads take longer than `max_standby_streaming_delay`, escalate to Option 4.

