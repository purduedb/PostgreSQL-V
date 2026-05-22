#ifndef INDEX_RECOVERY_WORKER_H
#define INDEX_RECOVERY_WORKER_H

#include "postgres.h"

void index_recovery_worker_init(void);
PGDLLEXPORT void index_recovery_worker_main(Datum main_arg);

#endif
