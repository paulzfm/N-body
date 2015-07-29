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
double m; // mass for each object
double k; //
double dt; // time interval
int N;  // num of bodies

Body* load_input(const char *file);
void xwindow_init(float _xmin, float _ymin, float _len_axis, int _len_window);
void xwindow_show(Body *bodies);

#endif // NBODY_UTIL_H_
