#include "run.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    float threshold = atof(argv[6]);
    printf("[loader] threshold: %f\n", threshold);
    bool opt_xwindow = strcmp(argv[7], "enable") == 0;
    float xmin, ymin, len_axis;
    int len_window;

    if (opt_xwindow) {
        xmin = atof(argv[8]);
        ymin = atof(argv[9]);
        len_axis = atof(argv[10]);
        len_window = atoi(argv[11]);
        printf("[loader] xwindow: enable (%f, %f), %d:%f\n",
            xmin, ymin, len_window, len_axis);
        xwindow_init(xmin, ymin, len_axis, len_window);
    } else {
        printf("[loader] xwindow: disable\n");
    }

    // load sample
    Body *samples = load_input(file, m);

    // record time costs
    float *pthread_time = new float[iter];
    float *cuda_time = new float[iter];

    // quad tree
    QuadTree tree(threshold, xmin, ymin, len_axis, len_axis, N);

    // 1 run pthread version
    printf("running pthread version...\n");
    Body *bodies = (Body*)malloc(sizeof(Body) * N); // working array
    memcpy(bodies, samples, sizeof(Body) * N);

    for (int k = 0; k < iter; k++) {
        run_pthread_version(k, num_threads, bodies, pthread_time + k, &tree);
        printf("[pthread] iter: %d, time elapsed: %.4f ms\n", k, pthread_time[k]);
        if (opt_xwindow) {
            xwindow_show(bodies, k % 20 == 0);
        }
    }

    free(bodies);

    // 2 run cuda version


    delete[] pthread_time;
    delete[] cuda_time;

    return 0;
}
