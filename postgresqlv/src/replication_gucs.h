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
extern int   dpv_replication_fetch_wait_timeout_ms;  /* standby barrier (ms) */

extern void dpv_replication_gucs_register(void);  /* call from _PG_init */

#endif
