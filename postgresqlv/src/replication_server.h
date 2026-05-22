#ifndef DPV_REPLICATION_SERVER_H
#define DPV_REPLICATION_SERVER_H

#include "postgres.h"
#include "fmgr.h"

/*
 * Wire protocol (binary, network byte order; helpers convert):
 *
 *   client -> server:
 *     [4]  protocol version (currently 1)
 *     [4]  shared secret length N
 *     [N]  shared secret bytes
 *     [4]  request kind (1 = GET_FILE)
 *     [4]  index_rel_id
 *     [4]  start_sid
 *     [4]  end_sid
 *     [4]  version
 *     [1]  file_kind enum (1=index 2=mapping 3=offset 4=bitmap 5=metadata)
 *
 *   server -> client:
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
    DPV_FILE_INDEX          = 1,
    DPV_FILE_MAPPING        = 2,
    DPV_FILE_OFFSET         = 3,
    DPV_FILE_BITMAP         = 4,
    DPV_FILE_METADATA       = 5,    /* per-segment: metadata_<s>_<e>_v<v> */
    DPV_FILE_INDEX_METADATA = 6,    /* per-index overall: <oid>/metadata */
} DpvFileKind;

typedef enum {
    DPV_RESULT_OK         = 0,
    DPV_RESULT_AUTH_FAIL  = 1,
    DPV_RESULT_NOT_FOUND  = 2,
    DPV_RESULT_IO_ERROR   = 3,
} DpvResultCode;

/* bgworker entry point. */
extern PGDLLEXPORT void replication_server_main(Datum main_arg) pg_attribute_noreturn();

#endif
