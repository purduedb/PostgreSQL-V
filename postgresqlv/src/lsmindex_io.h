#ifndef DPV_LSMINDEX_IO_H
#define DPV_LSMINDEX_IO_H

#include "postgres.h"
#include <stddef.h>
#include <stdint.h>

/*
 * File-path helpers for LSM segment content files. These were previously
 * static inside lsmindex_io.c; they are exposed here so that other
 * translation units (the physical-replication side channel:
 * replication_server.c on the primary, segment_fetcher.c on the standby)
 * can compute the same on-disk paths without duplicating the formatting.
 *
 * NOTE: only the helpers needed by other TUs are declared here; the
 * remaining helpers inside lsmindex_io.c stay file-local.
 */

extern void GetLSMIndexFilePathWithVersion(char *buf, size_t buflen,
                                           Oid indexRelId,
                                           uint32_t segmentIdStart,
                                           uint32_t segmentIdEnd,
                                           uint32_t version);

extern void GetLSMMappingFilePathWithVersion(char *buf, size_t buflen,
                                             Oid indexRelId,
                                             uint32_t segmentIdStart,
                                             uint32_t segmentIdEnd,
                                             uint32_t version);

extern void GetLSMOffsetFilePathWithVersion(char *buf, size_t buflen,
                                            Oid indexRelId,
                                            uint32_t segmentIdStart,
                                            uint32_t segmentIdEnd,
                                            uint32_t version);

extern void GetLSMBitmapFilePathWithVersion(char *buf, size_t buflen,
                                            Oid indexRelId,
                                            uint32_t segmentIdStart,
                                            uint32_t segmentIdEnd,
                                            uint32_t version);

extern void GetLSMSegmentMetadataPathWithVersion(char *buf, size_t buflen,
                                                 Oid indexRelId,
                                                 uint32_t segmentIdStart,
                                                 uint32_t segmentIdEnd,
                                                 uint32_t version);

/*
 * Per-index overall metadata file: <storage_base>/<indexRelId>/metadata.
 * Holds (index_type, dim, elem_size). Written once at CREATE INDEX time;
 * read by recover_lsm_index_internal on first-touch. The side channel must
 * transfer this file too (in addition to per-segment files) for the
 * standby's index load to succeed.
 */
extern void get_lsm_metadata_path(char *buf, size_t buflen, Oid indexRelId);

/*
 * Directory path helper and directory-creation utility.
 * Exposed so that redo callbacks (replication_rmgr.c) can create the
 * per-index storage directory on the standby before writing index_metadata.
 */
extern void GetLsmDirPath(char *buf, size_t buflen, Oid indexRelId);
extern void ensure_dir_exists(const char *path);

#endif /* DPV_LSMINDEX_IO_H */
