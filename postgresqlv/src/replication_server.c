#include "postgres.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "postmaster/bgworker.h"
#include "postmaster/interrupt.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "utils/elog.h"
#include "utils/guc.h"
#include "utils/wait_event.h"
#include "port/pg_bswap.h"  /* pg_ntoh32, pg_hton32, pg_ntoh64, pg_hton64 */

#include "lsmindex.h"
#include "lsmindex_io.h"
#include "replication_gucs.h"
#include "replication_server.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

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

static int
open_listen_socket(int port)
{
    int s;
    int opt = 1;
    struct sockaddr_in addr;

    s = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0)
        ereport(FATAL, (errmsg("dpv replication server: socket: %m")));

    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons(port);

    if (bind(s, (struct sockaddr *) &addr, sizeof(addr)) < 0)
        ereport(FATAL, (errmsg("dpv replication server: bind(:%d): %m", port)));
    if (listen(s, 16) < 0)
        ereport(FATAL, (errmsg("dpv replication server: listen: %m")));
    return s;
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
 * Build the on-disk path for a requested file kind.
 *
 * `index` and `metadata` files live at the base path (no chunk suffix). The
 * three chunked file kinds (mapping, offset, bitmap) are stored on disk as
 * `<base>.0`, `<base>.1`, ... — write_segment_file() splits at 1 GiB per
 * chunk. For Plan 2 v1 the side channel pulls only the `.0` chunk and the
 * fetcher writes it back to `.0` on the standby, so reads on the standby
 * find the chunk where read_segment_file() expects it.
 *
 * Multi-chunk transfer (files > 1 GiB) is a v2 concern; document but defer.
 */
static void
build_file_path(char *out, size_t out_size,
                Oid indexRelId, uint32 start, uint32 end, uint32 version,
                uint8 kind)
{
    char base[MAXPGPATH];

    switch (kind)
    {
        case DPV_FILE_INDEX:
            GetLSMIndexFilePathWithVersion(out, out_size, indexRelId, start, end, version);
            break;
        case DPV_FILE_MAPPING:
            GetLSMMappingFilePathWithVersion(base, sizeof(base), indexRelId, start, end, version);
            snprintf(out, out_size, "%s.0", base);
            break;
        case DPV_FILE_OFFSET:
            GetLSMOffsetFilePathWithVersion(base, sizeof(base), indexRelId, start, end, version);
            snprintf(out, out_size, "%s.0", base);
            break;
        case DPV_FILE_BITMAP:
            GetLSMBitmapFilePathWithVersion(base, sizeof(base), indexRelId, start, end, version);
            snprintf(out, out_size, "%s.0", base);
            break;
        case DPV_FILE_METADATA:
            GetLSMSegmentMetadataPathWithVersion(out, out_size, indexRelId, start, end, version);
            break;
        case DPV_FILE_INDEX_METADATA:
            get_lsm_metadata_path(out, out_size, indexRelId);
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
    uint32 req_kind, idx, start_sid, end_sid, version;
    uint8  fkind, result;
    char   path[MAXPGPATH];
    int    fd;
    uint64 size_be;
    struct stat st;

    if (!read_n(conn, &proto_ver, 4))
        goto out;
    proto_ver = pg_ntoh32(proto_ver);
    if (proto_ver != 1)
        goto out;

    if (!read_n(conn, &secret_len, 4))
        goto out;
    secret_len = pg_ntoh32(secret_len);
    if (secret_len >= sizeof(secret))
        goto out;
    if (!read_n(conn, secret, secret_len))
        goto out;
    secret[secret_len] = '\0';

    if (dpv_replication_shared_secret == NULL ||
        strcmp(secret, dpv_replication_shared_secret) != 0)
    {
        result = DPV_RESULT_AUTH_FAIL;
        write_n(conn, &result, 1);
        goto out;
    }

    if (!read_n(conn, &req_kind,  4)) goto out;
    if (!read_n(conn, &idx,       4)) goto out;
    if (!read_n(conn, &start_sid, 4)) goto out;
    if (!read_n(conn, &end_sid,   4)) goto out;
    if (!read_n(conn, &version,   4)) goto out;
    if (!read_n(conn, &fkind,     1)) goto out;

    req_kind  = pg_ntoh32(req_kind);
    idx       = pg_ntoh32(idx);
    start_sid = pg_ntoh32(start_sid);
    end_sid   = pg_ntoh32(end_sid);
    version   = pg_ntoh32(version);
    if (req_kind != DPV_REQ_GET_FILE)
        goto out;

    build_file_path(path, sizeof(path), (Oid) idx, start_sid, end_sid, version, fkind);
    elog(DEBUG1,
         "[dpv replication_server] file request: indexRelId=%u [%u,%u] v=%u kind=%u",
         idx, start_sid, end_sid, version, (unsigned) fkind);
    if (path[0] == '\0')
    {
        result = DPV_RESULT_NOT_FOUND;
        write_n(conn, &result, 1);
        goto out;
    }

    fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        result = DPV_RESULT_NOT_FOUND;
        write_n(conn, &result, 1);
        goto out;
    }
    if (fstat(fd, &st) != 0)
    {
        close(fd);
        result = DPV_RESULT_IO_ERROR;
        write_n(conn, &result, 1);
        goto out;
    }

    result = DPV_RESULT_OK;
    if (!write_n(conn, &result, 1))
    {
        close(fd);
        goto out;
    }
    size_be = pg_hton64((uint64) st.st_size);
    if (!write_n(conn, &size_be, 8))
    {
        close(fd);
        goto out;
    }

    /* Stream the file in 1 MiB chunks. */
    {
        size_t  buf_size = 1 << 20;
        char   *buf = malloc(buf_size);
        ssize_t k;

        if (buf == NULL)
        {
            elog(WARNING, "dpv replication_server: malloc(%zu) failed for stream buffer", buf_size);
            close(fd);
            goto out;
        }

        while ((k = read(fd, buf, buf_size)) > 0)
        {
            if (!write_n(conn, buf, (size_t) k))
                break;
        }
        free(buf);
    }
    close(fd);
    elog(DEBUG1,
         "[dpv replication_server] served kind=%u idx=%u [%u,%u] v=%u size=%lld",
         (unsigned) fkind, idx, start_sid, end_sid, version,
         (long long) st.st_size);

out:
    close(conn);
}

void
replication_server_main(Datum main_arg)
{
    int listen_fd;

    pqsignal(SIGTERM, sigterm_handler);
    pqsignal(SIGHUP,  sighup_handler);
    BackgroundWorkerUnblockSignals();

    if (dpv_replication_role != DPV_ROLE_PRIMARY)
    {
        elog(LOG, "dpv replication_server: not primary role, exiting");
        proc_exit(0);
    }
    if (dpv_replication_primary_port <= 0)
    {
        elog(LOG, "dpv replication_server: port not configured (set pgvector.replication_primary_port), exiting");
        proc_exit(0);
    }
    if (dpv_replication_shared_secret == NULL || dpv_replication_shared_secret[0] == '\0')
    {
        elog(LOG, "dpv replication_server: shared secret empty (set pgvector.replication_shared_secret), exiting");
        proc_exit(0);
    }

    listen_fd = open_listen_socket(dpv_replication_primary_port);
    elog(LOG, "dpv replication_server: listening on port %d", dpv_replication_primary_port);

    while (!got_sigterm)
    {
        int rc;

        rc = WaitLatchOrSocket(MyLatch,
                               WL_LATCH_SET | WL_SOCKET_READABLE | WL_EXIT_ON_PM_DEATH,
                               listen_fd, -1, PG_WAIT_EXTENSION);
        ResetLatch(MyLatch);

        if (got_sighup)
        {
            got_sighup = false;
            ProcessConfigFile(PGC_SIGHUP);
        }

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
