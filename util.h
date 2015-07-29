#ifndef NBODY_UTIL_H_
#define NBODY_UTIL_H_

#include <X11/Xlib.h>

typedef struct
{
    double x; // x-coordinate
    double y; // y-coordinate
    double vx; // x-velocity
    double vy; // y-velocity
} Body;

// global parameters
struct
{
    double m; // mass for each object
    double k; //
    double dt; // time interval
    int N;  // num of bodies
} global;

// xwindow parameters
Display *display;
Window window;
GC gc;
int screen;
float xmin, ymin, len_axis;
int len_window;

Body* load_input(const char *file);
void update_body(int i, Body *bodies, Body *new_body);
void xwindow_init(float _xmin, float _ymin, float _len_axis, int _len_window);
void xwindow_show(Body *bodies);

#endif // NBODY_UTIL_H_
