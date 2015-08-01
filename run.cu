#include "run.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

extern double dt;
extern int N;

// pthread worker
void *thread_worker(void *args)
{
    TaskParam *param = (TaskParam*)args;
    for (int i = param->start; i < param->end; i++) {
        param->tree->update(param->bodies + i);
    }

    pthread_exit(NULL);
}

// pthread version
void run_pthread_version(int i, int num_threads, Body *bodies,
    float *elapsed_time, QuadTree *tree)
{
    pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    TaskParam *param = (TaskParam*)malloc(sizeof(TaskParam) * num_threads);
    int width = ceil((float)N / num_threads);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // build tree
    tree->build(bodies);

    // create threads to compute
    for (int j = 0; j < num_threads; j++) {
        param[j].start = j * width;
        if (j == num_threads - 1) {
            param[j].end = N;
        } else {
            param[j].end = (j + 1) * width;
        }
        param[j].bodies = bodies;
        param[j].tree = tree;
        pthread_create(threads + j, NULL, thread_worker, param + j);
    }

    // wait
    for (int j = 0; j < num_threads; j++) {
        pthread_join(threads[j], NULL);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(threads);
    free(param);
}
