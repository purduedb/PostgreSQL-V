/*
 * index_recovery_worker.c
 *
 * Background worker that processes LSM index recovery requests.
 *
 * Backends that need to load an index into SharedLSMIndexBuffer claim a slot
 * (CAS FREE→RECOVERING), fill in request_db_oid / request_db_userid, signal
 * this worker via SharedIndexRecoveryCoordinator->worker_pgprocno, then sleep on
 * the slot's state_cv.  This worker scans for LSM_SLOT_RECOVERING slots, calls
 * recover_lsm_index_internal(), sets valid=LSM_SLOT_WRITABLE (or
 * state_error=1 + valid=LSM_SLOT_FREE on failure), then broadcasts the CV so
 * waiting backends wake up.
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

#include "index_recovery_worker.h"
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
 * index_recovery_worker_init
 *
 * Called from shmem_startup() — nothing to do here; the coordinator struct
 * is already initialized in lsm_index_buffer_shmem_initialize().
 */
void
index_recovery_worker_init(void)
{
}

/*
 * index_recovery_worker_main
 *
 * Entry point for the IndexRecoveryWorker background worker.
 */
void
index_recovery_worker_main(Datum main_arg)
{
    bool db_connected = false;
    Oid  connected_db_oid = InvalidOid;

    pqsignal(SIGTERM, ilw_sigterm);
    pqsignal(SIGHUP,  ilw_sighup);
    BackgroundWorkerUnblockSignals();

    /* Publish our pgprocno so backends can wake us */
    SharedIndexRecoveryCoordinator->worker_pgprocno = MyProcNumber;
    elog(LOG, "[IndexRecoveryWorker] started, pgprocno=%d", MyProcNumber);

    /*
     * Crash recovery: if the previous worker crashed while loading,
     * some slots may still be in LSM_SLOT_RECOVERING state with no one to
     * signal us.  Set our own latch so the first iteration of the main loop
     * will scan for any such stuck slots immediately.
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
            SharedIndexRecoveryCoordinator->worker_pgprocno = -1;
            proc_exit(0);
        }

        if (ilw_got_sigterm)
        {
            SharedIndexRecoveryCoordinator->worker_pgprocno = -1;
            proc_exit(0);
        }

        if (ilw_got_sighup)
        {
            ilw_got_sighup = false;
            ProcessConfigFile(PGC_SIGHUP);
        }

        /* Scan for slots that need recovery (LSM_SLOT_RECOVERING) */
        do {
            found_work = false;

            for (int i = 0; i < INDEX_BUF_SIZE; i++)
            {
                LSMIndexBufferSlot *slot = &SharedLSMIndexBuffer->slots[i];
                uint32 v = pg_atomic_read_u32(&slot->valid);

                if (v != (uint32) LSM_SLOT_RECOVERING)
                    continue;

                found_work = true;

                /*
                 * Connect to the database for this request if not yet
                 * connected.  Background workers may only call
                 * BackgroundWorkerInitializeConnection* once, so we stay
                 * connected to the first DB we serve.  If a different DB is
                 * requested we error the slot rather than crash.
                 */
                /*
                 * TODO(multi-DB): this worker is a single bgworker that
                 * stickily binds to the first dbOid it serves. With one
                 * DB on the cluster this is fine. For multi-DB standby
                 * (or any deployment serving multiple DBs from one
                 * cluster), this needs to become a per-DB pool — register
                 * one bgworker per database at extension load time,
                 * mirroring segment_fetcher_main's per-DB registration in
                 * vector.c. Until then, requests for a non-matching dbOid
                 * fail the slot below.
                 */
                if (!db_connected)
                {
                    Oid db_oid   = slot->request_db_oid;
                    Oid db_user  = slot->request_db_userid;

                    BackgroundWorkerInitializeConnectionByOid(db_oid, db_user, 0);
                    connected_db_oid = db_oid;
                    db_connected = true;
                    elog(LOG, "[IndexRecoveryWorker] connected to database %u", db_oid);
                }
                else if (slot->request_db_oid != connected_db_oid)
                {
                    elog(WARNING,
                         "[IndexRecoveryWorker] slot %d requests db %u but worker is connected to db %u — failing slot",
                         i, slot->request_db_oid, connected_db_oid);
                    pg_atomic_write_u32(&slot->state_error, 1);
                    pg_atomic_write_u32(&slot->valid, (uint32) LSM_SLOT_FREE);
                    ConditionVariableBroadcast(&slot->state_cv);
                    continue;
                }

                /* Call the internal recovery function inside PG_TRY to handle elog(ERROR) */
                StartTransactionCommand();
                PG_TRY();
                {
                    recover_lsm_index_internal(slot->lsmIndex.indexRelId, (uint32_t) i);
                    /* recover_lsm_index_internal sets valid to LSM_SLOT_WRITABLE itself */
                    CommitTransactionCommand();
                }
                PG_CATCH();
                {
                    /* Recovery failed — abort transaction, mark slot as failed and reset to free */
                    AbortCurrentTransaction();
                    elog(WARNING,
                         "[IndexRecoveryWorker] recover_lsm_index_internal failed for index %u slot %d",
                         slot->lsmIndex.indexRelId, i);
                    pg_atomic_write_u32(&slot->state_error, 1);
                    pg_atomic_write_u32(&slot->valid, (uint32) LSM_SLOT_FREE);
                    FlushErrorState();
                }
                PG_END_TRY();

                /* Wake all backends waiting on this slot's CV */
                ConditionVariableBroadcast(&slot->state_cv);
            }
        } while (found_work);
    }
}
