#include "run.h"

#include <pthread.h>
#include <stdlib.h>
#include <math.h>

extern "C" {
    void update_body(int i, Body *bodies, Body *new_body);
}

// pthread worker
void *thread_worker(void *args)
{
    TaskParam *param = (TaskParam*)args;
    int i;
    for (i = param->start; i < param->end; i++) {
        update_body(i, param->bodies, param->new_bodies + i);
    }

    pthread_exit(NULL);
}

// pthread version
void run_pthread_version(int i, int num_threads, Body *bodies,
    Body *new_bodies, float *elapsed_time)
{
    pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    TaskParam *param = (TaskParam*)malloc(sizeof(TaskParam) * num_threads);
    int width = ceil((float)global.N / num_threads);
    int j;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // create threads to compute
    for (j = 0; j < num_threads; j++) {
        param[j].start = j * width;
        if (j == num_threads - 1) {
            param[j].end = global.N;
        } else {
            param[j].end = (j + 1) * width;
        }
        param[j].bodies = bodies;
        param[j].new_bodies = new_bodies;
        pthread_create(threads + j, NULL, thread_worker, param + j);
    }

    // wait
    for (j = 0; j < num_threads; j++) {
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
