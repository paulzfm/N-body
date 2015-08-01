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

    // body locales inside this node?
    bool inside(const Body& body)
    {
        return (x <= body.x && body.x <= x + w) &&
            (y <= body.y && body.y <= y + h);
    }
};

// "class" QuadTree: both __host__ and __device__
__host__ __device__ void tree_build(Body *bodies, Node *nodes, int N, double *size);

__host__ __device__ void tree_update(Body *body, Node *nodes, double size,
    double threshold, double dt);

__host__ __device__ void tree_print(Node *nodes, int node, int indent);

// help functions
__host__ __device__ void tree_insert(Body *body, int node, Node *nodes, int *next);

__host__ __device__ void tree_search(int node, Body *body, double *a_x,
    double *a_y, Node *nodes, double size, double threshold);

#endif // NBODY_QUAD_TREE_H_
