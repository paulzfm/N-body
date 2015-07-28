#ifndef NBODY_UTIL_H_
#define NBODY_UTIL_H_

int xmin, ymin, len_axis, len_window;

extern void init_xwindow();

extern void render(float *xs, float *ys, int n);

#endif // NBODY_UTIL_H_
