#include "postgres.h"
#include "utils/guc.h"
#include "replication_gucs.h"

int   dpv_replication_role = DPV_ROLE_DISABLED;
char *dpv_replication_primary_host = NULL;
int   dpv_replication_primary_port = 0;
char *dpv_replication_shared_secret = NULL;
int   dpv_replication_fetch_parallelism = 2;
int   dpv_replication_fetch_wait_timeout_ms = 30000;  /* 30 s */

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

    DefineCustomIntVariable("pgvector.replication_fetch_wait_timeout",
        "Standby queryability barrier: max milliseconds a SELECT will wait "
        "for the initial-build segment to be fetched before failing.",
        NULL, &dpv_replication_fetch_wait_timeout_ms, 30000,
        0, 3600000, PGC_USERSET, GUC_UNIT_MS, NULL, NULL, NULL);
}
