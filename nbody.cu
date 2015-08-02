#include "run.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define __WAIT_AVAILABLE_GPU(x) cudaSetDevice(1);\
int __waitGPU;\
while (cudaMalloc((void**)&__waitGPU, sizeof(int)) == 46)\
{printf("waiting...\n");sleep(1);}printf("running...\n")

extern int N; // num of samples
int n; // num of nodes
double threshold;
double dt;

int main(int argc, char **argv)
{
    if (argc == 8) {
        if (strcmp(argv[7], "enable") == 0) {
            fprintf(stderr, "Option error: expected xwindow size.\n");
            exit(1);
        }
    } else if (argc == 12) {
        ;
    } else {
        fprintf(stderr, "Usage: %s num_of_threads m T t θ enable/disable xmin ymin length Length\n", argv[0]);
        exit(1);
    }

    int num_threads = atoi(argv[1]);
    printf("[loader] num of threads: %d\n", num_threads);
    float m = atof(argv[2]);
    printf("[loader] mass: %f\n", m);
    int iter = atoi(argv[3]);
    printf("[loader] total iter: %d\n", iter);
    dt = atof(argv[4]);
    printf("[loader] time interval: %f\n", dt);
    char file[255];
    strcpy(file, argv[5]);
    threshold = atof(argv[6]);
    printf("[loader] threshold: %f\n", threshold);
    bool opt_xwindow = strcmp(argv[7], "enable") == 0;

    if (opt_xwindow) {
        float xmin = atof(argv[8]);
        float ymin = atof(argv[9]);
        float len_axis = atof(argv[10]);
        int len_window = atoi(argv[11]);
        printf("[loader] xwindow: enable (%f, %f), %d:%f\n",
            xmin, ymin, len_window, len_axis);
        xwindow_init(xmin, ymin, len_axis, len_window);
    } else {
        printf("[loader] xwindow: disable\n");
    }

    // load sample
    Body *samples = load_input(file, m);

    // allocate memory for QuadTree
    n = N * 4;
    Node *tree = (Node*)malloc(sizeof(Node) * n);

    // record time costs
    float *pthread_time = new float[iter];
    float *cuda_time = new float[iter];

    // 1 run pthread version
    // printf("running pthread version...\n");
    // Body *bodies = (Body*)malloc(sizeof(Body) * N); // working array
    // memcpy(bodies, samples, sizeof(Body) * N);
    //
    // for (int k = 0; k < iter; k++) {
    //     run_pthread_version(k, num_threads, bodies, pthread_time + k, tree);
    //     printf("[pthread] iter: %d, time elapsed: %.4f ms\n", k, pthread_time[k]);
    //     if (opt_xwindow) {
    //         xwindow_show(bodies, true);
    // //         xwindow_show(bodies, k % 100 == 0);
    //     }
    // }

    // 2 run cuda version
    printf("running cuda version...\n");
    Body *bodies = (Body*)malloc(sizeof(Body) * N); // working array on cpu
    memcpy(bodies, samples, sizeof(Body) * N);

    // allocate gpu memory
    Body *d_bodies;
    Node *d_tree;
    cudaError_t err;

    __WAIT_AVAILABLE_GPU(1);
    cudaSetDevice(1);
    printf("[cuda] set device: 1\n");
    cudaDeviceSetLimit(cudaLimitStackSize, 4096);
    printf("[cuda] set stack size: 4096\n");

    err = cudaMalloc((void**)&d_bodies, sizeof(Body) * N);
    printf("[cuda] malloc d_bodies: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void**)&d_tree, sizeof(Node) * n);
    printf("[cuda] malloc d_tree: %s\n", cudaGetErrorString(err));

    // let's run
    for (int k = 0; k < iter; k++) {
        run_cuda_version(k, bodies, pthread_time + k, tree, d_bodies, d_tree);
        printf("[cuda] iter: %d, time elapsed: %.4f ms\n", k, cuda_time[k]);
        if (opt_xwindow) {
            xwindow_show(bodies, true);
            // xwindow_show(bodies, k % 100 == 0);
        }
    }

    free(samples);
    free(bodies);
    cudaFree(d_bodies);
    cudaFree(d_tree);

    delete[] pthread_time;
    delete[] cuda_time;

    return 0;
}
