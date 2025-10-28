#ifndef VECTOR_INDEX_WORKER_H
#define VECTOR_INDEX_WORKER_H

#include "postgres.h"
#include "storage/proc.h"   
#include "ringbuffer.h"

void vector_index_worker_init(void);
void vector_index_worker_main(Datum main_arg);


// typedef struct {
//     PGPROC *volatile client_proc;
//     PGPROC *volatile worker_proc;  /* new: set by worker at startup */
// } TaskDesc;

#endif