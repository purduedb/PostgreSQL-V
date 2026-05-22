#ifndef DPV_SEGMENT_FETCHER_H
#define DPV_SEGMENT_FETCHER_H

#include "postgres.h"
#include "fmgr.h"

extern PGDLLEXPORT void segment_fetcher_main(Datum main_arg) pg_attribute_noreturn();

#endif
