/*
 * segment_fetcher.c
 *
 * Standby-side bgworker (Phase 6b of decoupled_pgvector Plan 2).
 *
 * The fetcher loops:
 *   1. Pop a pending entry from the persistent fetch queue
 *      (dpv_queue_pop_pending).
 *   2. Connect to the primary's file server (Phase 5) and pull the 5
 *      segment files: index, mapping, offset, bitmap, metadata
 *      (metadata LAST so the atomic-rename of metadata marks completion).
 *   3. Send an adopt task to vector_index_worker via
 *      dpv_send_adopt_task(...), if the index is currently loaded.
 *   4. Mark the queue entry done.
 *
 * N fetcher workers run concurrently, where N is
 * pgvector.replication_fetch_parallelism. The queue uses flock(2) so
 * pop_pending is safe to call from multiple processes.
 */

#include "postgres.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "postmaster/bgworker.h"
#include "postmaster/interrupt.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "utils/elog.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/timestamp.h"
#include "utils/wait_event.h"
#include "port/pg_bswap.h"

#include "lsmindex.h"
#include "lsmindex_io.h"
#include "pending_fetch_queue.h"
#include "replication_gucs.h"
#include "replication_server.h"
#include "segment_fetcher.h"
#include "tasksend.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifndef SOCK_CLOEXEC
#define SOCK_CLOEXEC 0
#endif

static volatile sig_atomic_t got_sigterm = false;
static volatile sig_atomic_t got_sighup  = false;

static void
sigterm_handler(SIGNAL_ARGS)
{
    int save_errno = errno;
    got_sigterm = true;
    SetLatch(MyLatch);
    errno = save_errno;
}

static void
sighup_handler(SIGNAL_ARGS)
{
    int save_errno = errno;
    got_sighup = true;
    SetLatch(MyLatch);
    errno = save_errno;
}

static bool
read_n(int fd, void *buf, size_t n)
{
    size_t got = 0;
    while (got < n)
    {
        ssize_t k = read(fd, (char *) buf + got, n - got);
        if (k == 0)
            return false;
        if (k < 0)
        {
            if (errno == EINTR)
                continue;
            return false;
        }
        got += (size_t) k;
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
            if (errno == EINTR)
                continue;
            return false;
        }
        sent += (size_t) k;
    }
    return true;
}

/*
 * Open a TCP connection to the configured primary host:port.
 * Returns -1 on failure (caller logs WARNING).
 */
static int
connect_primary(void)
{
    int s;
    struct sockaddr_in addr;

    if (dpv_replication_primary_host == NULL ||
        dpv_replication_primary_host[0] == '\0')
    {
        elog(WARNING, "[dpv fetcher] primary host not configured");
        return -1;
    }
    if (dpv_replication_primary_port <= 0)
    {
        elog(WARNING, "[dpv fetcher] primary port not configured");
        return -1;
    }

    s = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0)
    {
        elog(WARNING, "[dpv fetcher] socket: %m");
        return -1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(dpv_replication_primary_port);
    if (inet_pton(AF_INET, dpv_replication_primary_host, &addr.sin_addr) != 1)
    {
        elog(WARNING, "[dpv fetcher] inet_pton(\"%s\") failed",
             dpv_replication_primary_host);
        close(s);
        return -1;
    }

    if (connect(s, (struct sockaddr *) &addr, sizeof(addr)) < 0)
    {
        elog(WARNING, "[dpv fetcher] connect %s:%d: %m",
             dpv_replication_primary_host, dpv_replication_primary_port);
        close(s);
        return -1;
    }

    return s;
}

/*
 * Build the destination path for a given file kind, mirroring the path
 * computation the server uses on the primary so files land in the same
 * relative location on the standby.
 */
/*
 * Build the destination path for a pulled file on the standby. Must mirror
 * the server's path layout (see replication_server.c build_file_path):
 *
 *   - index/metadata: base path, no chunk suffix
 *   - mapping/offset/bitmap: written as <base>.0 because write_segment_file
 *     chunks at 1 GiB and read_segment_file expects `.N` suffixes. v1 pulls
 *     only chunk 0; multi-chunk transfer is deferred to v2.
 */
static void
build_dest_path(char *out, size_t out_size,
                Oid idx, uint32 start, uint32 end, uint32 version, DpvFileKind kind)
{
    char base[MAXPGPATH];

    switch (kind)
    {
        case DPV_FILE_INDEX:
            GetLSMIndexFilePathWithVersion(out, out_size, idx, start, end, version);
            break;
        case DPV_FILE_MAPPING:
            GetLSMMappingFilePathWithVersion(base, sizeof(base), idx, start, end, version);
            snprintf(out, out_size, "%s.0", base);
            break;
        case DPV_FILE_OFFSET:
            GetLSMOffsetFilePathWithVersion(base, sizeof(base), idx, start, end, version);
            snprintf(out, out_size, "%s.0", base);
            break;
        case DPV_FILE_BITMAP:
            GetLSMBitmapFilePathWithVersion(base, sizeof(base), idx, start, end, version);
            snprintf(out, out_size, "%s.0", base);
            break;
        case DPV_FILE_METADATA:
            GetLSMSegmentMetadataPathWithVersion(out, out_size, idx, start, end, version);
            break;
        case DPV_FILE_INDEX_METADATA:
            get_lsm_metadata_path(out, out_size, idx);
            break;
        default:
            if (out_size > 0)
                out[0] = '\0';
            break;
    }
}

/*
 * Pull a single file from the primary, writing it to out_path via
 * temp-then-rename. New connection per request (the Phase 5 server is
 * single-request-per-conn).
 *
 * The streaming buffer is malloc'd (NOT palloc'd) because the bgworker
 * memory context may not be initialized this early, and the size (1 MiB)
 * is comfortable for the heap. The buffer is freed on every exit path.
 */
static bool
pull_one_file(Oid idx, uint32 start, uint32 end, uint32 version,
              DpvFileKind kind, const char *out_path)
{
    int    sock = -1;
    int    tmp_fd = -1;
    char   tmp_path[MAXPGPATH];
    char  *buf = NULL;
    size_t buf_size = 1 << 20;   /* 1 MiB */
    uint32 proto_ver_be;
    uint32 secret_len_be;
    uint32 req_kind_be;
    uint32 idx_be, start_be, end_be, version_be;
    uint8  kind_byte;
    uint8  result;
    uint64 size_be;
    uint64 remaining;
    size_t secret_len;
    const char *secret = dpv_replication_shared_secret;

    if (secret == NULL)
        secret = "";
    secret_len = strlen(secret);

    if (snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", out_path) >= (int) sizeof(tmp_path))
    {
        elog(WARNING, "[dpv fetcher] tmp path overflow for \"%s\"", out_path);
        return false;
    }

    sock = connect_primary();
    if (sock < 0)
        return false;

    /* Request header. */
    proto_ver_be  = pg_hton32(1);
    secret_len_be = pg_hton32((uint32) secret_len);
    req_kind_be   = pg_hton32(DPV_REQ_GET_FILE);
    idx_be        = pg_hton32((uint32) idx);
    start_be      = pg_hton32(start);
    end_be        = pg_hton32(end);
    version_be    = pg_hton32(version);
    kind_byte     = (uint8) kind;

    if (!write_n(sock, &proto_ver_be,  4) ||
        !write_n(sock, &secret_len_be, 4) ||
        (secret_len > 0 && !write_n(sock, secret, secret_len)) ||
        !write_n(sock, &req_kind_be,   4) ||
        !write_n(sock, &idx_be,        4) ||
        !write_n(sock, &start_be,      4) ||
        !write_n(sock, &end_be,        4) ||
        !write_n(sock, &version_be,    4) ||
        !write_n(sock, &kind_byte,     1))
    {
        elog(WARNING, "[dpv fetcher] short write requesting kind=%d for idx=%u [%u,%u] v=%u",
             (int) kind, idx, start, end, version);
        close(sock);
        return false;
    }

    /* Result byte. */
    if (!read_n(sock, &result, 1))
    {
        elog(WARNING, "[dpv fetcher] no result byte for kind=%d idx=%u [%u,%u] v=%u",
             (int) kind, idx, start, end, version);
        close(sock);
        return false;
    }
    if (result != DPV_RESULT_OK)
    {
        elog(WARNING, "[dpv fetcher] server returned result=%u for kind=%d idx=%u [%u,%u] v=%u",
             (unsigned) result, (int) kind, idx, start, end, version);
        close(sock);
        return false;
    }

    /* File size (network byte order). */
    if (!read_n(sock, &size_be, 8))
    {
        elog(WARNING, "[dpv fetcher] no size for kind=%d idx=%u [%u,%u] v=%u",
             (int) kind, idx, start, end, version);
        close(sock);
        return false;
    }
    remaining = pg_ntoh64(size_be);

    /* Open the temp file for writing. */
    tmp_fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (tmp_fd < 0)
    {
        elog(WARNING, "[dpv fetcher] open(\"%s\"): %m", tmp_path);
        close(sock);
        return false;
    }

    buf = (char *) malloc(buf_size);
    if (buf == NULL)
    {
        elog(WARNING, "[dpv fetcher] malloc(%zu) failed", buf_size);
        close(tmp_fd);
        unlink(tmp_path);
        close(sock);
        return false;
    }

    while (remaining > 0)
    {
        size_t chunk = (remaining > (uint64) buf_size) ? buf_size : (size_t) remaining;
        ssize_t k;

        k = read(sock, buf, chunk);
        if (k == 0)
        {
            elog(WARNING,
                 "[dpv fetcher] unexpected EOF (still %llu bytes) kind=%d idx=%u [%u,%u] v=%u",
                 (unsigned long long) remaining, (int) kind, idx, start, end, version);
            free(buf);
            close(tmp_fd);
            unlink(tmp_path);
            close(sock);
            return false;
        }
        if (k < 0)
        {
            if (errno == EINTR)
                continue;
            elog(WARNING, "[dpv fetcher] read from primary: %m");
            free(buf);
            close(tmp_fd);
            unlink(tmp_path);
            close(sock);
            return false;
        }
        if (!write_n(tmp_fd, buf, (size_t) k))
        {
            elog(WARNING, "[dpv fetcher] write to \"%s\": %m", tmp_path);
            free(buf);
            close(tmp_fd);
            unlink(tmp_path);
            close(sock);
            return false;
        }
        remaining -= (uint64) k;
    }

    free(buf);
    buf = NULL;

    /* Persist before rename so metadata-LAST really marks completion. */
    if (pg_fsync(tmp_fd) != 0)
    {
        elog(WARNING, "[dpv fetcher] fsync \"%s\": %m", tmp_path);
        close(tmp_fd);
        unlink(tmp_path);
        close(sock);
        return false;
    }

    if (close(tmp_fd) != 0)
    {
        elog(WARNING, "[dpv fetcher] close \"%s\": %m", tmp_path);
        unlink(tmp_path);
        close(sock);
        return false;
    }
    tmp_fd = -1;
    close(sock);
    sock = -1;

    if (rename(tmp_path, out_path) != 0)
    {
        elog(WARNING, "[dpv fetcher] rename(\"%s\",\"%s\"): %m", tmp_path, out_path);
        unlink(tmp_path);
        return false;
    }

    return true;
}

/*
 * Pull all five segment files. Metadata is pulled LAST: its atomic rename
 * marks the segment as complete on disk for any later observer.
 *
 * Idempotence: if the metadata file already exists (e.g. from base-sync
 * rsync or a prior successful run that crashed before the queue mark was
 * persisted), skip the entire pull.
 */
static bool
pull_all_files(Oid idx, uint32 start, uint32 end, uint32 version)
{
    char path[MAXPGPATH];
    char per_idx_dir[MAXPGPATH];
    DpvFileKind kinds[] = {
        DPV_FILE_INDEX,
        DPV_FILE_MAPPING,
        DPV_FILE_OFFSET,
        DPV_FILE_BITMAP,
        DPV_FILE_METADATA,           /* MUST be last */
    };
    size_t i;

    /*
     * Ensure the per-index storage directory exists on the standby before
     * pulling. The primary creates this dir as a side-effect of writing its
     * own segments; on the standby nothing creates it until the fetcher
     * arrives. (storage_base_dir/ itself is created by _PG_init.)
     */
    snprintf(per_idx_dir, sizeof(per_idx_dir), "%s%u",
             get_vector_storage_dir(), idx);
    if (mkdir(per_idx_dir, S_IRWXU) != 0 && errno != EEXIST)
    {
        elog(WARNING, "[dpv fetcher] mkdir %s: %m", per_idx_dir);
        return false;
    }

    /*
     * The per-index overall metadata file (<oid>/metadata) is written once
     * at CREATE INDEX time on the primary; recover_lsm_index_internal reads it
     * to obtain (index_type, dim, elem_size). Pull it on every batch — it's
     * tiny (12 bytes) and overwriting it with identical content is harmless.
     * Do this BEFORE the per-segment files so that a successful pull leaves
     * the index loadable even if the segment metadata-last rename is
     * interrupted.
     */
    {
        char idx_meta_path[MAXPGPATH];
        get_lsm_metadata_path(idx_meta_path, sizeof(idx_meta_path), idx);
        if (!pull_one_file(idx, start, end, version,
                            DPV_FILE_INDEX_METADATA, idx_meta_path))
        {
            elog(WARNING,
                 "[dpv fetcher] failed to pull per-index metadata for idx=%u",
                 idx);
            return false;
        }
    }

    GetLSMSegmentMetadataPathWithVersion(path, sizeof(path), idx, start, end, version);
    if (access(path, F_OK) == 0)
    {
        elog(DEBUG1,
             "[dpv fetcher] metadata for idx=%u [%u,%u] v=%u already present, skipping pull",
             idx, start, end, version);
        return true;
    }

    for (i = 0; i < lengthof(kinds); i++)
    {
        build_dest_path(path, sizeof(path), idx, start, end, version, kinds[i]);
        if (path[0] == '\0')
        {
            elog(WARNING, "[dpv fetcher] could not build path for kind=%d", (int) kinds[i]);
            return false;
        }
        if (!pull_one_file(idx, start, end, version, kinds[i], path))
            return false;
    }
    return true;
}

/*
 * fetcher_release_covered_memtables — release all memtables whose
 * memtable_id falls in [start_sid, end_sid] from the LSM index slot.
 *
 * Called by segment_fetcher_main (a regular PG bgworker process) after
 * dpv_pool_adopt returns ADOPTED or INDEX_UNLOADED. Acquires lsm->mt_lock
 * LW_EXCLUSIVE; re-scans under the lock to avoid trusting the earlier
 * LW_SHARED snapshot.
 *
 * This is the ONLY path that acquires mt_lock for the adoption flow —
 * vector_index_worker pthreads no longer touch mt_lock.
 */
static void
fetcher_release_covered_memtables(int lsm_idx,
                                   SegmentId start_sid, SegmentId end_sid)
{
    LSMIndexBufferSlot *slot;
    LSMIndex lsm;
    int new_count = 0;
    int released = 0;
    uint32 i;

    if (lsm_idx < 0 || lsm_idx >= INDEX_BUF_SIZE)
        return;
    slot = &SharedLSMIndexBuffer->slots[lsm_idx];
    if (!is_writable(pg_atomic_read_u32(&slot->valid)))
    {
        elog(ERROR, "[segment_fetcher] lsm_idx %d is not writable", lsm_idx);
        return;
    }
    lsm = &slot->lsmIndex;

    LWLockAcquire(lsm->mt_lock, LW_EXCLUSIVE);

    /* Re-scan sealed memtables under the exclusive lock. */
    for (i = 0; i < lsm->memtable_count; i++)
    {
        int32 mt_idx = lsm->memtable_idxs[i];
        ConcurrentMemTable mt;

        if (mt_idx < 0)
            continue;
        mt = MT_FROM_SLOTIDX(mt_idx);
        if (mt->memtable_id >= start_sid && mt->memtable_id <= end_sid)
        {
            pg_atomic_add_fetch_u32(
                &SharedMemtableBuffer->slots[mt_idx].ref_count, (uint32) -1);
            released++;
            /* Do not copy into new_count position — effectively removes it. */
        }
        else
        {
            lsm->memtable_idxs[new_count++] = mt_idx;
        }
    }
    lsm->memtable_count = new_count;

    /* Check growing memtable. */
    if (lsm->growing_memtable_idx != MT_IDX_INVALID &&
        lsm->growing_memtable_idx != MT_IDX_ROTATING) // impossible to be MT_IDX_ROTATING
    {
        ConcurrentMemTable g = MT_FROM_SLOTIDX(lsm->growing_memtable_idx);
        if (g->memtable_id >= start_sid && g->memtable_id <= end_sid)
        {
            pg_atomic_add_fetch_u32(
                &SharedMemtableBuffer->slots[lsm->growing_memtable_idx].ref_count,
                (uint32) -1);
            lsm->growing_memtable_idx = MT_IDX_INVALID;
            lsm->growing_memtable_id = 0;
            released++;
        }
    }

    LWLockRelease(lsm->mt_lock);

    elog(DEBUG1, "[segment_fetcher] released %d memtables in range [%u,%u]",
         released, start_sid, end_sid);
}

void
segment_fetcher_main(Datum main_arg)
{
    int worker_id = DatumGetInt32(main_arg);

    pqsignal(SIGTERM, sigterm_handler);
    pqsignal(SIGHUP,  sighup_handler);
    BackgroundWorkerUnblockSignals();

    if (dpv_replication_role != DPV_ROLE_STANDBY)
    {
        elog(LOG, "dpv segment_fetcher (worker %d): not standby role, exiting", worker_id);
        proc_exit(0);
    }

    dpv_queue_init();
    /* Only worker 0 runs the orphan-recovery sweep. Other workers would race
     * with a still-running peer mid-pull: their recover-on-startup would demote
     * the peer's FETCHING entry back to PENDING, causing both to pull the same
     * files. Worker 0 also restarts on crash via bgw_restart_time; in that
     * window worker 1 may pick up new PENDING entries without recovery, which
     * is fine — orphan FETCHING entries from a prior crash will simply wait
     * until worker 0 comes back to demote them. */
    if (worker_id == 0)
        dpv_queue_recover_on_startup();

    elog(LOG, "dpv segment_fetcher (worker %d): started", worker_id);

    while (!got_sigterm)
    {
        DpvFetchEntry *e;
        bool           fetched;

        if (got_sighup)
        {
            got_sighup = false;
            ProcessConfigFile(PGC_SIGHUP);
        }

        e = dpv_queue_pop_pending();
        if (e == NULL)
        {
            (void) WaitLatch(MyLatch,
                             WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                             1000, PG_WAIT_EXTENSION);
            ResetLatch(MyLatch);
            continue;
        }

        elog(DEBUG1,
             "[dpv fetcher %d] pulling indexRelId=%u [%u,%u] v=%u kind=%u",
             worker_id, e->hdr.indexRelId,
             (uint32) e->hdr.start_sid, (uint32) e->hdr.end_sid,
             e->hdr.version, (unsigned) e->hdr.kind);

        fetched = pull_all_files(e->hdr.indexRelId,
                                 (uint32) e->hdr.start_sid,
                                 (uint32) e->hdr.end_sid,
                                 e->hdr.version);
        if (!fetched)
        {
            elog(WARNING,
                 "[dpv fetcher] pull failed for indexRelId=%u range=[%u,%u] v=%u",
                 e->hdr.indexRelId,
                 (uint32) e->hdr.start_sid,
                 (uint32) e->hdr.end_sid,
                 e->hdr.version);
            dpv_queue_mark(e->filename, DPV_FETCH_FAILED);
        }
        else
        {
            int lsm_idx = lookup_lsm_index_idx(e->hdr.indexRelId);

            if (lsm_idx < 0)
            {
                elog(DEBUG1,
                     "[dpv fetcher] index %u not loaded; files persist on disk for later adoption",
                     e->hdr.indexRelId);
            }
            else
            {
                int adopt_result;
                SegmentId          memtable_cover[MEMTABLE_NUM + 1];
                ConcurrentMemTable cover_mts[MEMTABLE_NUM + 1];
                int memtable_cover_count = 0;

                /*
                 * Pre-flight (Plan 3 Part C — race fix vs. vacuum redo):
                 *
                 * Under mt_lock SHARED, scan memtables in [start_sid, end_sid]
                 * AND take per-mt vacuum_lock SHARED on each cover memtable
                 * BEFORE releasing mt_lock. The vacuum_locks stay held until
                 * after fetcher_release_covered_memtables returns below.
                 *
                 * Lock order: mt_lock SHARED → vacuum_lock SHARED → mt_lock
                 * release → ... → mt_lock EXCLUSIVE (in release_covered) →
                 * mt_lock release → vacuum_lock release. The redo writer
                 * (apply_to_memtable_for_sid via dpv_apply_vacuum_tombstones)
                 * acquires vacuum_lock EXCLUSIVE AFTER taking its initial
                 * mt_lock SHARED, so the two paths cannot deadlock.
                 *
                 * Effect: while we hold these SHAREDs, redo writers trying to
                 * mutate mt->bitmap on cover memtables block. This guarantees
                 * the per-sid groups snapshot below is consistent with the
                 * adoption commit — no bit can be set on a cover memtable
                 * between snapshot and replace_flushed_segments_n.
                 */
                {
                    LSMIndex lsm_scan = &SharedLSMIndexBuffer->slots[lsm_idx].lsmIndex;
                    uint32 i;

                    LWLockAcquire(lsm_scan->mt_lock, LW_SHARED);

                    /* Sealed memtables */
                    for (i = 0; i < lsm_scan->memtable_count &&
                                memtable_cover_count < MEMTABLE_NUM + 1; i++)
                    {
                        int32 mt_idx = lsm_scan->memtable_idxs[i];
                        ConcurrentMemTable mt;

                        if (mt_idx < 0)
                            continue;
                        mt = MT_FROM_SLOTIDX(mt_idx);
                        if (mt->memtable_id >= e->hdr.start_sid &&
                            mt->memtable_id <= e->hdr.end_sid)
                        {
                            cover_mts[memtable_cover_count] = mt;
                            memtable_cover[memtable_cover_count] = mt->memtable_id;
                            memtable_cover_count++;
                        }
                    }

                    /* Growing memtable */
                    if (lsm_scan->growing_memtable_idx != MT_IDX_INVALID &&
                        lsm_scan->growing_memtable_idx != MT_IDX_ROTATING &&
                        memtable_cover_count < MEMTABLE_NUM + 1)
                    {
                        ConcurrentMemTable mt =
                            MT_FROM_SLOTIDX(lsm_scan->growing_memtable_idx);
                        if (mt->memtable_id >= e->hdr.start_sid &&
                            mt->memtable_id <= e->hdr.end_sid)
                        {
                            cover_mts[memtable_cover_count] = mt;
                            memtable_cover[memtable_cover_count] = mt->memtable_id;
                            memtable_cover_count++;
                        }
                    }

                    /* Take vacuum_lock SHARED on each cover memtable (held
                     * across the ADOPT flow). */
                    for (int ci = 0; ci < memtable_cover_count; ci++)
                        LWLockAcquire(&cover_mts[ci]->vacuum_lock, LW_SHARED);

                    LWLockRelease(lsm_scan->mt_lock);

                    /* Sort cover ascending (required by dpv_pool_adopt). */
                    /* Insertion sort — cover is at most MEMTABLE_NUM+1 = 5 elements. */
                    for (int si = 1; si < memtable_cover_count; si++)
                    {
                        SegmentId          v_sid = memtable_cover[si];
                        ConcurrentMemTable v_mt  = cover_mts[si];
                        int sj = si - 1;
                        while (sj >= 0 && memtable_cover[sj] > v_sid)
                        {
                            memtable_cover[sj + 1] = memtable_cover[sj];
                            cover_mts[sj + 1]      = cover_mts[sj];
                            sj--;
                        }
                        memtable_cover[sj + 1] = v_sid;
                        cover_mts[sj + 1]      = v_mt;
                    }
                }

                /*
                 * Plan 3 refactor — collect deletion bits as per-sid groups.
                 *
                 * vacuum_lock SHARED is already held on each cover memtable
                 * (from the pre-flight above); walk mt->bitmap directly. One
                 * group per cover memtable (each memtable carries exactly one
                 * sid). tids[] is in insertion order — matches the order they
                 * appear in the new segment's map_ptr for that sid.
                 */
                {
                DpvVacuumGroup *groups = (memtable_cover_count > 0)
                    ? (DpvVacuumGroup *) palloc(sizeof(DpvVacuumGroup) * memtable_cover_count)
                    : NULL;
                int64_t **per_group_tids =
                    (memtable_cover_count > 0)
                    ? (int64_t **) palloc(sizeof(int64_t *) * memtable_cover_count)
                    : NULL;
                int n_groups = 0;

                for (int mci = 0; mci < memtable_cover_count; mci++)
                {
                    ConcurrentMemTable mt = cover_mts[mci];
                    uint32 cur_size = pg_atomic_read_u32(&mt->current_size);
                    uint32 valid_size = (cur_size > mt->capacity)
                                        ? mt->capacity : cur_size;

                    /* First pass: count set bits to size the per-group buffer. */
                    int n_set = 0;
                    for (uint32 j = 0; j < valid_size; j++)
                        if (IS_SLOT_SET(mt->bitmap, j)) n_set++;

                    if (n_set == 0)
                        continue;

                    int64_t *g_tids = (int64_t *) palloc(sizeof(int64_t) * n_set);
                    int      g_n    = 0;
                    /* Second pass: copy tids in insertion order. */
                    for (uint32 j = 0; j < valid_size; j++)
                        if (IS_SLOT_SET(mt->bitmap, j))
                            g_tids[g_n++] = mt->tids[j];

                    groups[n_groups].sid    = mt->memtable_id;
                    groups[n_groups].n_tids = (uint32) g_n;
                    groups[n_groups].tids   = g_tids;
                    per_group_tids[n_groups] = g_tids;
                    n_groups++;
                }

                /* Submit ADOPT task to vector_index_worker. */
                {
                    TimestampTz dbg_t0 = GetCurrentTimestamp();
                    long        dbg_secs;
                    int         dbg_usecs;

                    adopt_result = dpv_send_adopt_task(lsm_idx,
                                                       e->hdr.indexRelId,
                                                       e->hdr.start_sid,
                                                       e->hdr.end_sid,
                                                       e->hdr.version,
                                                       memtable_cover,
                                                       memtable_cover_count,
                                                       groups,
                                                       n_groups);

                    TimestampDifference(dbg_t0, GetCurrentTimestamp(), &dbg_secs, &dbg_usecs);
                    /*
                     * ADOPT timing log: kept at LOG because a stalled wait
                     * here is the canonical symptom of a maintenance-worker
                     * wakeup loss (see test 112 history). One line per
                     * adopted/discarded segment is low volume.
                     */
                    elog(LOG,
                         "[dpv fetcher %d] ADOPT [%u,%u] v=%u -> %d in %ld.%06d s",
                         worker_id,
                         (uint32) e->hdr.start_sid, (uint32) e->hdr.end_sid,
                         e->hdr.version, adopt_result, dbg_secs, dbg_usecs);
                }

                if (groups != NULL)
                {
                    for (int g = 0; g < n_groups; g++)
                        if (per_group_tids[g] != NULL) pfree(per_group_tids[g]);
                    pfree(groups);
                }
                if (per_group_tids != NULL)
                    pfree(per_group_tids);
                }  /* close Plan 3 refactor groups block */

                /*
                 * Post-task memtable release:
                 *   0 = ADOPTED        → release memtables in range.
                 *   2 = INDEX_UNLOADED → release memtables; segment file is
                 *                        durable on disk regardless of pool state.
                 *   1 = STALE_DISCARD  → do NOT release; state is inconsistent.
                 */
                if (adopt_result == 0 || adopt_result == 2)
                {
                    fetcher_release_covered_memtables(lsm_idx,
                                                      e->hdr.start_sid,
                                                      e->hdr.end_sid);
                }
                else /* STALE_DISCARD */
                {
                    elog(WARNING,
                         "[segment_fetcher] STALE_DISCARD for [%u,%u] v%u — not releasing memtables",
                         (uint32) e->hdr.start_sid,
                         (uint32) e->hdr.end_sid,
                         e->hdr.version);
                }

                /*
                 * Plan 3 Part C — release per-mt vacuum_lock SHARED held since
                 * the pre-flight above. Released AFTER
                 * fetcher_release_covered_memtables so any vacuum-redo writer
                 * waiting on vacuum_lock EXCLUSIVE wakes after memtable_idxs[]
                 * has been cleared — its re-validation finds the memtable
                 * gone, returns MT_LOST_DURING_WAIT, and the dispatcher
                 * retry routes the tids to the now-adopted segment.
                 */
                for (int ci = 0; ci < memtable_cover_count; ci++)
                    LWLockRelease(&cover_mts[ci]->vacuum_lock);
            }
            dpv_queue_mark(e->filename, DPV_FETCH_DONE);
        }

        if (e->trailer)
            pfree(e->trailer);
        pfree(e);
    }

    elog(LOG, "dpv segment_fetcher (worker %d): SIGTERM, exiting", worker_id);
    proc_exit(0);
}
