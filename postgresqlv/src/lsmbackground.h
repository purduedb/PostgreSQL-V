#ifndef LSMBACKGROUND_H
#define LSMBACKGROUND_H

#include "postgres.h"

void lsm_index_bgworker_main(Datum main_arg);

#endif