#include "run.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

extern int N;
extern int n;
extern double threshold;

// pthread worker
void *thread_worker(void *args)
{
/*
    TaskParam *param = (TaskParam*)args;
    for (int i = param->start; i < param->end; i++) {
        param->tree->update(param->bodies + i);
    }

    pthread_exit(NULL);
*/
    pthread_exit(NULL);
}

// pthread version
void run_pthread_version(int i, int num_threads, Body *bodies,
    float *elapsed_time, Node *tree)
{
/*
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
*/
}

// cuda worker
__global__ void cuda_worker(Node *tree, Body *bodies, double threshold, double size, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    tree_update(bodies + i, tree, size, threshold);
}

// cuda version
void run_cuda_version(int i, Body *bodies,
    float *elapsed_time, Node *tree)
{
    cudaEvent_t start, stop;
    Body *d_bodies;
    Node *d_tree;
    cudaMalloc((void**)&d_bodies, sizeof(Body) * N);
    cudaMalloc((void**)&d_tree, sizeof(Node) * n);
    double size;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // build tree
    tree_build(bodies, tree, N, &size);

    cudaMemcpy(d_bodies, bodies, sizeof(Body) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tree, tree, sizeof(Node) * n, cudaMemcpyHostToDevice);

    // compute
    int block = ceil(N / 512.0);
    cuda_worker<<<block, 512>>>(d_tree, d_bodies, threshold, size, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(bodies, d_bodies, sizeof(Body) * N, cudaMemcpyDeviceToHost);
}
