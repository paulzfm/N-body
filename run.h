#ifndef NBODY_RUN_H_
#define NBODY_RUN_H_

#include "util.h"
#include "QuadTree.h"

struct TaskParam
{
    int start, end; // [start, end)
    Body *bodies;   // bodies
    QuadTree *tree; // data structure
};

// pthread run
void run_pthread_version(int i, int num_threads, Body *bodies,
    float *elapsed_time, QuadTree *tree);

// cuda run
void run_cuda_version(int i, Body *bodies,
    float *elapsed_time, QuadTree *tree);

#endif // NBODY_RUN_H_
