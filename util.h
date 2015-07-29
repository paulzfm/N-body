#ifndef NBODY_UTIL_H_
#define NBODY_UTIL_H_

#include <X11/Xlib.h>

struct Body
{
    double x; // x-coordinate
    double y; // y-coordinate
    double vx; // x-velocity
    double vy; // y-velocity
};

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

#endif // NBODY_UTIL_H_
