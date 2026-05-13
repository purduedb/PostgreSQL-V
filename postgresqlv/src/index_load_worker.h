#ifndef INDEX_LOAD_WORKER_H
#define INDEX_LOAD_WORKER_H

#include "postgres.h"

void index_load_worker_init(void);
void index_load_worker_main(Datum main_arg);

#endif
