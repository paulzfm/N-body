#ifndef NBODY_RUN_H_
#define NBODY_RUN_H_

#include "util.h"
#include "QuadTree.h"

struct TaskParam
{
    int start, end; // [start, end)
    Body *bodies;   // bodies
    Node *tree;     // data structure
};

// pthread run
void run_pthread_version(int i, int num_threads, Body *bodies,
    float *elapsed_time, Node *tree);

// cuda run
void run_cuda_version(int i, Body *bodies,
    float *elapsed_time, Node *tree);

#endif // NBODY_RUN_H_
