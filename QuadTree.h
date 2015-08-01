#ifndef NBODY_QUAD_TREE_H_
#define NBODY_QUAD_TREE_H_

#include "util.h"

struct Node
{
    int status;
    // one of the following:
    const static int EMPTY = 0;
    const static int INTERNAL = 1;
    const static int EXTERNAL = 2;

    int children[4]; // index of children

    double x, y, w, h; // border rectangle

    Body body; // body

    Node() { status = EMPTY; }
};

#endif // NBODY_QUAD_TREE_H_
