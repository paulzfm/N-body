#include "run.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    global.m = atof(argv[2]);
    printf("[loader] mass: %f\n", global.m);
    int iter = atoi(argv[3]);
    printf("[loader] total iter: %d\n", iter);
    global.dt = atof(argv[4]);
    printf("[loader] time interval: %f\n", global.dt);
    char file[255];
    strcpy(file, argv[5]);
    bool opt_bha = argv[6][0] == 'y';
    bool opt_xwindow = argv[7][0] == 'e';

    if (opt_xwindow) {
        float xmin = atof(argv[8]);
        float ymin = atof(argv[9]);
        float len_axis = atof(argv[10]);
        int len_window = atoi(argv[11]);
        printf("[loader] xwindow: enable (%f, %f), %d:%f\n", xmin, ymin, len_window, len_axis);
        init_xwindow(xmin, ymin, len_axis, len_window);
    } else {
        printf("[loader] xwindow: disable\n");
    }

    if (opt_bha) {
        printf("[loader] algorithm: bha\n");
    } else {
        printf("[loader] algorithm: brute-force\n");
    }

    // load sample
    Body *samples = malloc(sizeof(Body) * global.N);
    load_input(file, samples);

    // record time costs
    pthread_time = (float*)malloc(sizeof(float) * iter);
    cuda_time = (float*)malloc(sizeof(float) * iter);

    int i, k;

    // 1 run pthread version
    printf("running pthread version...\n");
    Body *bodies = malloc(sizeof(Body) * global.N);
    Body *buffer = malloc(sizeof(Body) * global.N);
    memcpy(bodies, samples, sizeof(Body) * global.N);

    for (k = 0; k < iter; k++) {
        run_pthread_version(k, num_threads, bodies, buffer, pthread_time + k);
        printf("[pthread] iter: %d, time elapsed: %.4f ms\n", i, pthread_time[i]);
        for (i = 0; i < global.N; i++) {
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
