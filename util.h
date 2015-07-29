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
extern double m; // mass for each object
extern double k; //
extern double dt; // time interval
extern int N;  // num of bodies

extern float xmin, ymin, len_axis;
extern int len_window;

Body* load_input(const char *file);
void xwindow_init(float _xmin, float _ymin, float _len_axis, int _len_window);
void xwindow_show(Body *bodies);

#endif // NBODY_UTIL_H_
