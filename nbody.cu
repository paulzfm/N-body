#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <X11/Xlib.h>
#include <pthread.h>

// global data
int NTHREADS; // number of threads
int N;        // number of bodies
float *vx;    // current velocity in x-axis
float *vy;    // current velocity in y-axis
float *x;     // current x-position
float *y;     // current y-position
float m;      // mass for all bodies

// options
bool opt_xwindow; // enable xwindow?
bool opt_bha;     // use Barnes-Hut algorithm?

// parameters
float dt = 1.0; // time inteval
float k = 1.0;  // gravitational constant

// timer
cudaEvent_t start, stop;

// time elapsed (in ms)
float *pthread_time, *cuda_time;

// xwindow
Display *display;
Window window;
GC gc;
int screen;
int xmin, ymin, len_axis, len_window;

// init xwindow
void init_xwindow()
{
    /* open connection with the server */
	display = XOpenDisplay(NULL);
	if (display == NULL) {
		fprintf(stderr, "Error: cannot open display\n");
        exit(1);
	}

	screen = DefaultScreen(display);

	/* set window position */
	int x = xmin;
	int y = ymin;

	/* border width in pixels */
	int border_width = 0;

	/* create window */
	window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y,
        len_window, len_window, border_width,
        BlackPixel(display, screen), WhitePixel(display, screen));

	/* create graph */
	XGCValues values;
	long valuemask = 0;

	gc = XCreateGC(display, window, valuemask, &values);
	XSetForeground(display, gc, BlackPixel(display, screen));
	XSetBackground(display, gc, 0X0000FF00);
	XSetLineAttributes(display, gc, 1, LineSolid, CapRound, JoinRound);

	/* map(show) the window */
	XMapWindow(display, window);
	XSync(display, 0);
}

// render window
void render(float *xs, float *ys, int n)
{
    XSetForeground(display, gc, BlackPixel(display, screen));
    int i, x, y;
    for (i = 0; i < n; i++) {
        x = (xs[i] + 0.5 * len_axis) / len_axis * len_window;
        y = (ys[i] + 0.5 * len_axis) / len_axis * len_window;
        XDrawPoint(display, window, gc, x, y);
    }
    XFlush(display);
}

// force routine
void force_routine(int i, float *f_x, float *f_y)
{
    int j;
    *f_x = 0;
    *f_y = 0;
    for (j = 0; j < N; j++) {
        if (i != j) {
            float r2 = (x[j] - x[i]) * (x[j] - x[i]) +
                (y[j] - y[i]) * (y[j] - y[i]);
            float f = k * m * m / (r2 * sqrt(r2));
            *f_x += f * (x[j] - x[i]);
            *f_y += f * (y[j] - y[i]);
        }
    }
}

// cuda task routine: update status of i-th body
void update(int i, float *vx_new, float *vy_new, float *x_new, float *y_new)
{
    float f_x, f_y;
    force_routine(i, &f_x, &f_y);
    vx_new[i] = vx[i] + f_x * dt / m;
    vy_new[i] = vy[i] + f_y * dt / m;
    x_new[i] = x[i] + vx_new[i] * dt;
    y_new[i] = y[i] + vy_new[i] * dt;
}

// pthread task routine
struct TaskParam
{
    // task range: [start, end)
    int start;
    int end;

    // buffers
    float *vx_new;
    float *vy_new;
    float *x_new;
    float *y_new;
};

void *task(void *args)
{
    TaskParam *param = (TaskParam*)args;
    int i;
    for (i = param->start; i < param->end; i++) {
        float f_x, f_y;
        force_routine(i, &f_x, &f_y);
        param->vx_new[i] = vx[i] + f_x * dt / m;
        param->vy_new[i] = vy[i] + f_y * dt / m;
        param->x_new[i] = x[i] + param->vx_new[i] * dt;
        param->y_new[i] = y[i] + param->vy_new[i] * dt;
    }

    pthread_exit(NULL);
}

// pthread version main
void pthread_control(int iter)
{
    int i, j, k;
    float *vx_new = (float*)malloc(sizeof(float) * N);
    float *vy_new = (float*)malloc(sizeof(float) * N);
    float *x_new = (float*)malloc(sizeof(float) * N);
    float *y_new = (float*)malloc(sizeof(float) * N);
    pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * NTHREADS);
    TaskParam *param = (TaskParam*)malloc(sizeof(TaskParam) * NTHREADS);
    int width = ceil(N / NTHREADS); // width for each task package
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (i = 0; i < iter; i++) {
        cudaEventRecord(start);

        // create threads to compute
        for (j = 0; j < NTHREADS; j++) {
            param[j].start = j * width;
            if (j == NTHREADS - 1) {
                param[j].end = N;
            } else {
                param[j].end = (j + 1) * width;
            }
            param[j].vx_new = vx_new;
            param[j].vy_new = vy_new;
            param[j].x_new = x_new;
            param[j].y_new = y_new;
            pthread_create(threads + j, NULL, task, param + j);
        }

        // wait
        for (j = 0; j < NTHREADS; j++) {
            pthread_join(threads[j], NULL);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(pthread_time + i, start, stop);

        printf("[pthread] iter: %d, time elapsed: %.4f ms\n", i, pthread_time[i]);

        // commit changes to global data
        for (k = 0; k < N; k++) {
            vx[k] = vx_new[k];
            vy[k] = vy_new[k];
            x[k] = x_new[k];
            y[k] = y_new[k];
        }

        // display
        if (opt_xwindow) {
            render(x, y, N);
        }
    }

    free(vx_new);
    free(vy_new);
    free(x_new);
    free(y_new);
    free(threads);
    free(param);
}

// cuda version main
void cuda_control(int iter)
{
    // float *vx_last, *vy_last, *x_last, *y_last;
    // float *vx_new, *vy_new, *x_new, *y_new;
    // cudaMalloc(vx_last, );
}

// load sample from file
void load_input(const char *sample)
{
    FILE *fin = fopen(sample, "r");
    fscanf(fin, "%d", &N);
    vx = (float*)malloc(sizeof(float) * N);
    vy = (float*)malloc(sizeof(float) * N);
    x = (float*)malloc(sizeof(float) * N);
    y = (float*)malloc(sizeof(float) * N);

    int i;
    for (i = 0; i < N; i++) {
        fscanf(fin, "%f%f%f%f", x + i, y + i, vx + i, vy + i);
    }

    fclose(fin);
    printf("[loader] %d samples loaded.\n", N);
}

int main(int argc, char **argv)
{
    if (argc == 8) {
        if (strcmp(argv[7], "enable") == 0) {
            fprintf(stderr, "Option error: expected xwindow size.\n");
            exit(1);
        }
    } else if (argc == 12) {
        xmin = atoi(argv[8]);
        ymin = atoi(argv[9]);
        len_axis = atoi(argv[10]);
        len_window = atoi(argv[11]);
    } else {
        fprintf(stderr, "Usage: %s num_of_threads m T t FILE θ enable/disable xmin ymin length Length\n", argv[0]);
        exit(1);
    }

    NTHREADS = atoi(argv[1]);
    printf("[loader] num of threads: %d\n", NTHREADS);
    m = atof(argv[2]);
    printf("[loader] mass: %f\n", m);
    int iter = atoi(argv[3]);
    printf("[loader] total iter: %d\n", iter);
    dt = atof(argv[4]);
    printf("[loader] time interval: %f\n", dt);
    char sample[255];
    strcpy(sample, argv[5]);
    opt_bha = argv[6][0] == 'y';
    opt_xwindow = argv[7][0] == 'e';

    if (opt_xwindow) {
        printf("[loader] xwindow: enable\n");
        init_xwindow();
    } else {
        printf("[loader] xwindow: disable\n");
    }

    if (opt_bha) {
        printf("[loader] algorithm: bha\n");
    } else {
        printf("[loader] algorithm: brute-force\n");
    }

    // load sample
    load_input(sample);

    // allocate memory
    pthread_time = (float*)malloc(sizeof(float) * iter);
    cuda_time = (float*)malloc(sizeof(float) * iter);

    // 1 run pthread version
    pthread_control(iter);

    // 2 run cuda version

    return 0;
}
