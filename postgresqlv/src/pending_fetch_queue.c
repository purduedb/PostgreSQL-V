#include "postgres.h"
#include "miscadmin.h"
#include "storage/fd.h"
#include "storage/lwlock.h"
#include "utils/builtins.h"
#include "common/file_perm.h"
#include "common/string.h"     /* pg_str_endswith */

#include "lsm_segment.h"
#include "lsmindex.h"          /* get_vector_storage_dir() */
#include "pending_fetch_queue.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

/* Internal: build the queue dir path. get_vector_storage_dir() returns a
 * path that already has a trailing slash, so we concatenate as "%s%s". */
static void
queue_dir_path(char *buf, size_t buflen)
{
    snprintf(buf, buflen, "%s%s", get_vector_storage_dir(), DPV_FETCH_QUEUE_DIR);
}

static void
ensure_queue_dir(void)
{
    char dir[MAXPGPATH];

    queue_dir_path(dir, sizeof(dir));

    /* Ensure the parent (the vector storage dir) exists. It is created by
     * normal LSM setup, but on a fresh standby it may not yet exist. */
    {
        const char *parent = get_vector_storage_dir();
        char        parent_buf[MAXPGPATH];
        size_t      plen;

        strlcpy(parent_buf, parent, sizeof(parent_buf));
        plen = strlen(parent_buf);
        /* Strip trailing slash for mkdir. */
        if (plen > 1 && parent_buf[plen - 1] == '/')
            parent_buf[plen - 1] = '\0';

        if (mkdir(parent_buf, pg_dir_create_mode) != 0 && errno != EEXIST)
            elog(ERROR, "[dpv_queue] mkdir %s: %m", parent_buf);
    }

    if (mkdir(dir, pg_dir_create_mode) != 0 && errno != EEXIST)
        elog(ERROR, "[dpv_queue] mkdir %s: %m", dir);
}

void
dpv_queue_init(void)
{
    ensure_queue_dir();
}

typedef struct {
    Oid          indexRelId;
    SegmentId    start_sid;
    SegmentId    end_sid;
    uint32       version;
    DpvFetchKind kind;
} DpvFetchKey;

/* Filename is deterministic from the key — same key → same filename → atomic
 * rename gives natural deduplication. Total length is at most:
 *   8 (oid hex) + 1 + 10 + 1 + 10 + 1 + 10 + 1 + 1 (kind digit) + 6 (.entry) = 49
 * which fits comfortably in the 64-byte DpvFetchEntry.filename buffer. */
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
    DpvFetchKey key;
    int  fd;

    key.indexRelId = hdr->indexRelId;
    key.start_sid  = hdr->start_sid;
    key.end_sid    = hdr->end_sid;
    key.version    = hdr->version;
    key.kind       = hdr->kind;

    /* Lazily ensure the queue directory exists. The redo callback (running in
     * the startup process) reaches here before any fetcher bgworker has had a
     * chance to call dpv_queue_init(), so the directory may not yet exist on
     * a fresh standby. ensure_queue_dir() is idempotent (mkdir with EEXIST
     * tolerated). */
    ensure_queue_dir();

    queue_dir_path(dir, sizeof(dir));
    key_to_filename(&key, fname, sizeof(fname));
    snprintf(entry_path, sizeof(entry_path), "%s/%s", dir, fname);
    snprintf(tmp_path,   sizeof(tmp_path),   "%s/.%s.tmp", dir, fname);

    /* Atomic existence check via access(). If the final entry already
     * exists, the fetch was already enqueued (or is in flight). */
    if (access(entry_path, F_OK) == 0)
    {
        elog(DEBUG1, "[dpv queue] enqueue duplicate (already on disk): %s", fname);
        return false;
    }

    fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, pg_file_create_mode);
    if (fd < 0)
        elog(ERROR, "[dpv_queue] open %s: %m", tmp_path);

    if (write(fd, hdr, sizeof(*hdr)) != (ssize_t) sizeof(*hdr))
    {
        int save_errno = errno;
        close(fd);
        unlink(tmp_path);
        errno = save_errno;
        elog(ERROR, "[dpv_queue] write header %s: %m", tmp_path);
    }
    if (trailer_size > 0 &&
        write(fd, trailer, trailer_size) != (ssize_t) trailer_size)
    {
        int save_errno = errno;
        close(fd);
        unlink(tmp_path);
        errno = save_errno;
        elog(ERROR, "[dpv_queue] write trailer %s: %m", tmp_path);
    }

    if (pg_fsync(fd) != 0)
    {
        int save_errno = errno;
        close(fd);
        unlink(tmp_path);
        errno = save_errno;
        elog(ERROR, "[dpv_queue] fsync %s: %m", tmp_path);
    }
    close(fd);

    if (rename(tmp_path, entry_path) != 0)
    {
        int save_errno = errno;
        unlink(tmp_path);
        errno = save_errno;
        elog(ERROR, "[dpv_queue] rename %s -> %s: %m", tmp_path, entry_path);
    }

    /* fsync the directory so the rename is durable. */
    fsync_fname(dir, true);

    elog(DEBUG1, "[dpv queue] enqueued %s", fname);
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

        /* Skip "." / ".." and any hidden / tmp file. */
        if (de->d_name[0] == '.')
            continue;
        if (!pg_str_endswith(de->d_name, ".entry"))
            continue;

        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        fd = open(path, O_RDWR);
        if (fd < 0)
            continue;
        /* Try to claim this entry exclusively. If another worker already holds
         * the lock, skip — it is (or will be) claimed by them. The lock is
         * released when we close(fd) at the end of this function. */
        if (flock(fd, LOCK_EX | LOCK_NB) != 0)
        {
            if (errno == EWOULDBLOCK || errno == EAGAIN)
            {
                close(fd);
                continue;
            }
            close(fd);
            continue;
        }
        if (fstat(fd, &st) != 0)
        {
            close(fd);
            continue;
        }
        if ((size_t) st.st_size < sizeof(hdr))
        {
            close(fd);
            continue;
        }
        if (read(fd, &hdr, sizeof(hdr)) != (ssize_t) sizeof(hdr))
        {
            close(fd);
            continue;
        }
        if (hdr.status != DPV_FETCH_PENDING)
        {
            close(fd);
            continue;
        }

        /* Atomically claim by rewriting the header with status = FETCHING. */
        hdr.status = DPV_FETCH_FETCHING;
        if (pwrite(fd, &hdr, sizeof(hdr), 0) != (ssize_t) sizeof(hdr))
        {
            close(fd);
            continue;
        }
        if (pg_fsync(fd) != 0)
            elog(WARNING, "[dpv_queue] fsync %s: %m", path);

        trailer_size = (size_t) st.st_size - sizeof(hdr);
        out = (DpvFetchEntry *) palloc(sizeof(DpvFetchEntry));
        out->hdr = hdr;
        strlcpy(out->filename, de->d_name, sizeof(out->filename));
        out->trailer_size = trailer_size;
        out->trailer = trailer_size ? (char *) palloc(trailer_size) : NULL;
        if (trailer_size > 0 &&
            pread(fd, out->trailer, trailer_size, sizeof(hdr)) != (ssize_t) trailer_size)
        {
            close(fd);
            if (out->trailer)
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
        if (pwrite(fd, &hdr, sizeof(hdr), 0) != (ssize_t) sizeof(hdr))
            elog(WARNING, "[dpv_queue] pwrite %s: %m", path);
        if (pg_fsync(fd) != 0)
            elog(WARNING, "[dpv_queue] fsync %s: %m", path);
    }
    close(fd);
}

void
dpv_queue_recover_on_startup(void)
{
    /* On startup, FETCHING entries are orphans from a crashed worker. Demote
     * them back to PENDING so they get picked up again. Also clean orphan
     * temporary files left from interrupted enqueues. */
    char dir[MAXPGPATH];
    DIR *d;
    struct dirent *de;

    ensure_queue_dir();
    queue_dir_path(dir, sizeof(dir));

    d = AllocateDir(dir);
    if (!d)
        return;

    while ((de = ReadDir(d, dir)) != NULL)
    {
        char path[MAXPGPATH];
        int  fd;
        DpvFetchEntryHeader hdr;

        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;

        if (pg_str_endswith(de->d_name, ".tmp"))
        {
            /* Orphan tempfile from interrupted enqueue — unlink. */
            snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
            unlink(path);
            continue;
        }
        if (!pg_str_endswith(de->d_name, ".entry"))
            continue;

        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        fd = open(path, O_RDWR);
        if (fd < 0)
            continue;
        if (read(fd, &hdr, sizeof(hdr)) == (ssize_t) sizeof(hdr) &&
            hdr.status == DPV_FETCH_FETCHING)
        {
            hdr.status = DPV_FETCH_PENDING;
            if (pwrite(fd, &hdr, sizeof(hdr), 0) != (ssize_t) sizeof(hdr))
                elog(WARNING, "[dpv_queue] pwrite %s: %m", path);
            if (pg_fsync(fd) != 0)
                elog(WARNING, "[dpv_queue] fsync %s: %m", path);
        }
        close(fd);
    }
    FreeDir(d);
    fsync_fname(dir, true);
}
