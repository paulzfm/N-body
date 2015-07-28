#include <X11/Xlib.h>
#include <stdio.h>

/* window size */
#define width 400
#define height 400

Display *display;
Window window;
GC gc;

void init_xwindow()
{
    /* open connection with the server */
	display = XOpenDisplay(NULL);
	if (display == NULL) {
		fprintf(stderr, "Error: cannot open display\n");
        exit(1);
	}

	int screen = DefaultScreen(display);

	/* set window position */
	int x = 0;
	int y = 0;

	/* border width in pixels */
	int border_width = 0;

	/* create window */
	window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y,
        width, height, border_width,
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

void render(float *xs, float *ys, int n)
{
    XSetForeground(display, gc, BlackPixel(display, screen));
    int i;
    for (i = 0; i < n; i++) {
        XDrawPoint(display, window, gc, xs[i], ys[i]);
    }
    XFlush(display);
}
