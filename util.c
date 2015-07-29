#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void update_body(int i, Body *bodies, Body *new_body)
{
	double a_x = 0;
	double a_y = 0;
	int j;
    for (j = 0; j < global.N; j++) {
        if (i != j) {
            double r2 = (bodies[j].x - bodies[i].x) * (bodies[j].x - bodies[i].x)
                + (bodies[j].y - bodies[i].y) * (bodies[j].y - bodies[i].y);
            double a = global.k * global.m / (r2 * sqrt(r2));
            a_x += a * (x[j] - x[i]);
            a_y += a * (y[j] - y[i]);
        }
    }

    new_body.vx = bodies[i].vx + a_x * global.dt;
	new_body.vy = bodies[i].vy + a_y * global.dt;
    new_body.x = bodies[i].x + new_body.vx * global.dt;
    new_body.y = bodies[i].y + new_body.vy * global.dt;
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

void xwindow_show(Body *bodies)
{
    XClearArea(display, window, 0, 0, len_window, len_window, 0);
    XSetForeground(display, gc, BlackPixel(display, screen));
    int i, x, y;
	// printf("drawing (%lf, %lf) -> (%d, %d)\n", xs[0], ys[0],
		// (int)(((float)xs[0] - xmin) / len_axis * (float)len_window),
		// (int)(((float)ys[0] - ymin) / len_axis * (float)len_window)
	// );
    for (i = 0; i < global.N; i++) {
        x = (int)(((float)bodies[i].x - xmin) / len_axis * (float)len_window);
        y = (int)(((float)bodies[i].y - ymin) / len_axis * (float)len_window);
        XDrawPoint(display, window, gc, x, y);
    }
    XFlush(display);
}
