#ifndef NBODY_RUN_H_
#define NBODY_RUN_H_

#include "util.h"

typedef struct TaskParam_t
{
    int start, end; // [start, end)
    Body *bodies, *new_bodies;
} TaskParam;

extern "C" {

// pthread run
void run_pthread_version(int i, int num_threads, Body *bodies,
    Body *new_bodies, float *elapsed_time);

// cuda run
}

#endif // NBODY_RUN_H_
