#include "run.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double m;     // mass for each body
double dt;    // time interval
extern int N; // num of samples

int main(int argc, char **argv)
{
    if (argc == 8) {
        if (strcmp(argv[7], "enable") == 0) {
            fprintf(stderr, "Option error: expected xwindow size.\n");
            exit(1);
        }
    } else if (argc == 12) {
        (void*)0;
    } else {
        fprintf(stderr, "Usage: %s num_of_threads m T t θ enable/disable xmin ymin length Length\n", argv[0]);
        exit(1);
    }

    int num_threads = atoi(argv[1]);
    printf("[loader] num of threads: %d\n", num_threads);
    m = atof(argv[2]);
    printf("[loader] mass: %f\n", m);
    int iter = atoi(argv[3]);
    printf("[loader] total iter: %d\n", iter);
    dt = atof(argv[4]);
    printf("[loader] time interval: %f\n", dt);
    char file[255];
    strcpy(file, argv[5]);
    int opt_bha = argv[6][0] == 'y';
    int opt_xwindow = strcmp(argv[7], "enable") == 0;

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

    if (opt_bha) {
        printf("[loader] algorithm: bha\n");
    } else {
        printf("[loader] algorithm: brute-force\n");
    }

    // load sample
    Body *samples = load_input(file);

    // record time costs
    float *pthread_time = (float*)malloc(sizeof(float) * iter);
    float *cuda_time = (float*)malloc(sizeof(float) * iter);

    int i, k;

    // 1 run pthread version
    printf("running pthread version...\n");
    Body *bodies = (Body*)malloc(sizeof(Body) * N);
    Body *buffer = (Body*)malloc(sizeof(Body) * N);
    memcpy(bodies, samples, sizeof(Body) * N);

    for (k = 0; k < iter; k++) {
        run_pthread_version(k, num_threads, bodies, buffer, pthread_time + k);
        printf("[pthread] iter: %d, time elapsed: %.4f ms\n", k, pthread_time[k]);
        for (i = 0; i < N; i++) {
            bodies[i] = buffer[i];
        }
        if (opt_xwindow) {
            xwindow_show(bodies);
        }
    }

    free(bodies);
    free(buffer);

    // 2 run cuda version

    return 0;
}
