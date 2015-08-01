#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

// xwindow parameters
Display *display;
Window window;
GC gc;
int screen;
float xmin, ymin, len_axis;
int len_window;

extern double dt;
int N;

Body* load_input(const char *file, double mass)
{
	FILE *fin = fopen(file, "r");
	if (!fin) {
		fprintf(stderr, "[loader] no such file: \"%s\"\n", file);
		exit(1);
	}

	fscanf(fin, "%d", &N);
	Body* samples = (Body*)malloc(sizeof(Body) * N);

	for (int i = 0; i < N; i++) {
		samples[i].idx = i;
		fscanf(fin, "%lf%lf%lf%lf", &(samples[i].x), &(samples[i].y),
			&(samples[i].vx), &(samples[i].vy));
		samples[i].m = mass;
	}

	fclose(fin);
	printf("[loader] load from \"%s\": %d samples.\n", file, N);
	return samples;
}

void xwindow_init(float _xmin, float _ymin, float _len_axis, int _len_window)
{
    xmin = _xmin;
    ymin = _ymin;
    len_axis = _len_axis;
    len_window = _len_window;

    /* open connection with the server */
    display = XOpenDisplay(NULL);
    if (display == NULL) {
        fprintf(stderr, "Error: cannot open display\n");
        exit(1);
    }

    screen = DefaultScreen(display);

    /* create window */
    window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0,
        len_window, len_window, 0,
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

void xwindow_show(Body *bodies, bool sync)
{
    XClearArea(display, window, 0, 0, len_window, len_window, 0);
    XSetForeground(display, gc, BlackPixel(display, screen));
    int i, x, y;
    for (i = 0; i < N; i++) {
        x = (int)(((float)bodies[i].x - xmin) / len_axis * (float)len_window);
        y = (int)(((float)bodies[i].y - ymin) / len_axis * (float)len_window);
        XDrawPoint(display, window, gc, x, y);
    }

	if (sync) {
		XSync(display, 0);
		// XFlush(display);
	}
}
