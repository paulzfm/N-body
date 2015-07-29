#include "run.h"

#include <pthread.h>
#include <stdlib.h>
#include <math.h>

extern double m;
extern double dt;
extern int N;

const double k = 0.001;

void update_body(int i, Body *bodies, Body *new_body)
{
	double a_x = 0;
	double a_y = 0;
	int j;
    for (j = 0; j < N; j++) {
        if (i != j) {
            double r2 = (bodies[j].x - bodies[i].x) * (bodies[j].x - bodies[i].x)
                + (bodies[j].y - bodies[i].y) * (bodies[j].y - bodies[i].y);
            double a = k * m / (r2 * sqrt(r2));
            a_x += a * (bodies[j].x - bodies[i].x);
            a_y += a * (bodies[j].y - bodies[i].y);
        }
    }

    new_body->vx = bodies[i].vx + a_x * dt;
	new_body->vy = bodies[i].vy + a_y * dt;
    new_body->x = bodies[i].x + new_body->vx * dt;
    new_body->y = bodies[i].y + new_body->vy * dt;
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
    int width = ceil((float)N / num_threads);
    int j;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // create threads to compute
    for (j = 0; j < num_threads; j++) {
        param[j].start = j * width;
        if (j == num_threads - 1) {
            param[j].end = N;
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
