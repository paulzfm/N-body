#ifndef NBODY_RUN_H_
#define NBODY_RUN_H_

#include "util.h"
#include "QuadTree.h"

#include <pthread.h>

struct TaskParam
{
    int start, end; // [start, end)
    Body *bodies;   // bodies
    Node *tree;     // data structure
};

// pthread run
void run_pthread_version(int i, int num_threads, Body *bodies,
    float *elapsed_time, Node *tree, pthread_t *threads, TaskParam *param);

// cuda run
void run_cuda_version(int i, Body *bodies,
    float *elapsed_time, Node *tree, Body *d_bodies, Node *d_tree);

#endif // NBODY_RUN_H_
