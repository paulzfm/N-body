#ifndef NBODY_UTIL_H_
#define NBODY_UTIL_H_

int xmin, ymin, len_axis, len_window;

void init_xwindow();

void render(float *xs, float *ys, int n);

#endif // NBODY_UTIL_H_
