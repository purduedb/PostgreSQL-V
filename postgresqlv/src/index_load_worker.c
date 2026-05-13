/*
 * index_load_worker.c
 *
 * Background worker that processes LSM index load requests.
 *
 * Backends that need to load an index into SharedLSMIndexBuffer claim a slot
 * (CAS 0→2), fill in request_db_oid / request_db_userid, signal this worker
 * via SharedIndexLoadCoordinator->worker_pgprocno, then sleep on the slot's
 * load_cv.  This worker scans for valid==2 slots, calls
 * load_lsm_index_internal(), sets valid=1 (or load_error=1 + valid=0 on
 * failure), then broadcasts the CV so waiting backends wake up.
 */

#include "postgres.h"
#include "access/xact.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "storage/condition_variable.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "storage/lwlock.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "tcop/tcopprot.h"
#include "utils/elog.h"
#include "utils/memutils.h"
#include "utils/wait_event.h"

#include "index_load_worker.h"
#include "lsmindex.h"

static volatile sig_atomic_t ilw_got_sigterm = false;
static volatile sig_atomic_t ilw_got_sighup  = false;

static void
ilw_sigterm(SIGNAL_ARGS)
{
    int save_errno = errno;
    ilw_got_sigterm = true;
    SetLatch(MyLatch);
    errno = save_errno;
}

static void
ilw_sighup(SIGNAL_ARGS)
{
    int save_errno = errno;
    ilw_got_sighup = true;
    SetLatch(MyLatch);
    errno = save_errno;
}

/*
 * index_load_worker_init
 *
 * Called from shmem_startup() — nothing to do here; the coordinator struct
 * is already initialized in lsm_index_buffer_shmem_initialize().
 */
void
index_load_worker_init(void)
{
}

/*
 * index_load_worker_main
 *
 * Entry point for the IndexLoadWorker background worker.
 */
void
index_load_worker_main(Datum main_arg)
{
    bool db_connected = false;
    Oid  connected_db_oid = InvalidOid;

    pqsignal(SIGTERM, ilw_sigterm);
    pqsignal(SIGHUP,  ilw_sighup);
    BackgroundWorkerUnblockSignals();

    /* Publish our pgprocno so backends can wake us */
    SharedIndexLoadCoordinator->worker_pgprocno = MyProc->pgprocno;
    elog(LOG, "[IndexLoadWorker] started, pgprocno=%d", MyProc->pgprocno);

    /*
     * Crash recovery: if the previous worker crashed while loading,
     * some slots may still be in valid==2 state with no one to signal us.
     * Set our own latch so the first iteration of the main loop will scan
     * for any such stuck slots immediately.
     */
    SetLatch(MyLatch);

    for (;;)
    {
        int rc;
        bool found_work;

        rc = WaitLatch(MyLatch,
                       WL_LATCH_SET | WL_POSTMASTER_DEATH,
                       0,   /* no timeout */
                       PG_WAIT_EXTENSION);
        ResetLatch(MyLatch);

        if (rc & WL_POSTMASTER_DEATH)
        {
            SharedIndexLoadCoordinator->worker_pgprocno = -1;
            proc_exit(0);
        }

        if (ilw_got_sigterm)
        {
            SharedIndexLoadCoordinator->worker_pgprocno = -1;
            proc_exit(0);
        }

        if (ilw_got_sighup)
        {
            ilw_got_sighup = false;
            ProcessConfigFile(PGC_SIGHUP);
        }

        /* Scan for slots that need loading (valid == 2) */
        do {
            found_work = false;

            for (int i = 0; i < INDEX_BUF_SIZE; i++)
            {
                LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
                uint32 v = pg_atomic_read_u32(&slot->valid);

                if (v != 2)
                    continue;

                found_work = true;

                /*
                 * Connect to the database for this request if not yet
                 * connected.  Background workers may only call
                 * BackgroundWorkerInitializeConnection* once, so we stay
                 * connected to the first DB we serve.  If a different DB is
                 * requested we error the slot rather than crash.
                 */
                if (!db_connected)
                {
                    Oid db_oid   = slot->request_db_oid;
                    Oid db_user  = slot->request_db_userid;

                    BackgroundWorkerInitializeConnectionByOid(db_oid, db_user, 0);
                    connected_db_oid = db_oid;
                    db_connected = true;
                    elog(LOG, "[IndexLoadWorker] connected to database %u", db_oid);
                }
                else if (slot->request_db_oid != connected_db_oid)
                {
                    elog(WARNING,
                         "[IndexLoadWorker] slot %d requests db %u but worker is connected to db %u — failing slot",
                         i, slot->request_db_oid, connected_db_oid);
                    pg_atomic_write_u32(&slot->load_error, 1);
                    pg_atomic_write_u32(&slot->valid, 0);
                    ConditionVariableBroadcast(&slot->load_cv);
                    continue;
                }

                /* Call the internal load function inside PG_TRY to handle elog(ERROR) */
                StartTransactionCommand();
                PG_TRY();
                {
                    load_lsm_index_internal(slot->lsmIndex.indexRelId, (uint32_t) i);
                    /* load_lsm_index_internal sets valid=1 itself */
                    CommitTransactionCommand();
                }
                PG_CATCH();
                {
                    /* Loading failed — abort transaction, mark slot as failed and reset to free */
                    AbortCurrentTransaction();
                    elog(WARNING,
                         "[IndexLoadWorker] load_lsm_index_internal failed for index %u slot %d",
                         slot->lsmIndex.indexRelId, i);
                    pg_atomic_write_u32(&slot->load_error, 1);
                    pg_atomic_write_u32(&slot->valid, 0);
                    FlushErrorState();
                }
                PG_END_TRY();

                /* Wake all backends waiting on this slot's CV */
                ConditionVariableBroadcast(&slot->load_cv);
            }
        } while (found_work);
    }
}
