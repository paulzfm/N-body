#include "run.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define __WAIT_AVAILABLE_GPU(x) \
    cudaSetDevice(x);\
    int __waitGPU;\
    while (cudaMalloc((void**)&__waitGPU, sizeof(int)) == 46)\
    {printf("[gpu] waiting...\n");sleep(1);}

extern int N; // num of samples
int n; // num of nodes
double threshold;
double dt; // time interval

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
        fprintf(stderr, "Usage: %s num_of_threads m T t file θ enable/disable xmin ymin length Length\n", argv[0]);
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
    float *pthread_time = (float*)malloc(sizeof(float) * iter);
    float *cuda_time = (float*)malloc(sizeof(float) * iter);

    // 1 run pthread version
    printf("[pthread] launch\n");
    Body *bodies = (Body*)malloc(sizeof(Body) * N); // working array
    memcpy(bodies, samples, sizeof(Body) * N);

    // allocate memory for threads
    pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    TaskParam *param = (TaskParam*)malloc(sizeof(TaskParam) * num_threads);

    // run
    for (int k = 0; k < iter; k++) {
        run_pthread_version(k, num_threads, bodies, pthread_time + k, tree,
            threads, param);
        printf("[pthread] iter: %d, time elapsed: %.4f ms\n", k, pthread_time[k]);
        if (opt_xwindow) {
            xwindow_show(bodies, true);
        }
    }

    free(threads);
    free(param);

    // 2 run cuda version
    printf("[cuda] launch\n");
    memcpy(bodies, samples, sizeof(Body) * N);

    // allocate gpu memory
    Body *d_bodies;
    Node *d_tree;

    __WAIT_AVAILABLE_GPU(1);
    // cudaDeviceSetLimit(cudaLimitStackSize, 10240);

    cudaMalloc((void**)&d_bodies, sizeof(Body) * N);
    cudaMalloc((void**)&d_tree, sizeof(Node) * n);

    // run
    for (int k = 0; k < iter; k++) {
        run_cuda_version(k, bodies, cuda_time + k, tree, d_bodies, d_tree);
        printf("[cuda] iter: %d, time elapsed: %.4f ms\n", k, cuda_time[k]);
        if (opt_xwindow) {
            xwindow_show(bodies, true);
        }
    }

    cudaFree(d_bodies);
    cudaFree(d_tree);

// save to file
FILE *___file = fopen("all_times.txt", "w");
fprintf(___file, "pthread=[");
for (int i = 0; i < iter; i++) {
    fprintf(___file, "%f ", pthread_time[i]);
}
fprintf(___file, "]\ncuda=[");
for (int i = 0; i < iter; i++) {
    fprintf(___file, "%f ", cuda_time[i]);
}
fprintf(___file, "]\n");
fclose(___file);

    // summary
    printf("[summary] computation time (ms)\n");

    float min = 1e8;
    float max = -1e8;
    float total = 0.0;
    for (int i = 0; i < iter; i++) {
        if (pthread_time[i] < min) {
            min = pthread_time[i];
        } else if (pthread_time[i] > max) {
            max = pthread_time[i];
        }
        total += pthread_time[i];
    }
    printf("pthread: min %.4f, max %.4f, average %.4f, total %.4f\n",
        min, max, total / iter, total);

    min = 1e8;
    max = -1e8;
    total = 0.0;
    for (int i = 0; i < iter; i++) {
        if (cuda_time[i] < min) {
            min = cuda_time[i];
        } else if (cuda_time[i] > max) {
            max = cuda_time[i];
        }
        total += cuda_time[i];
    }
    printf("cuda   : min %.4f, max %.4f, average %.4f, total %.4f\n",
        min, max, total / iter, total);


    free(pthread_time);
    free(cuda_time);
    free(bodies);
    free(samples);

    printf("[loader] all done\n");

    return 0;
}
