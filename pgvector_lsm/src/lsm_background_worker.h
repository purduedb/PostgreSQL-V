#ifndef LSM_BACKGROUND_WORKER_H
#define LSM_BACKGROUND_WORKER_H

#include "postgres.h"


void lsm_index_bgworker_main(Datum main_arg);

#endif