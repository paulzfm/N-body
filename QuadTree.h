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

class QuadTree
{
public:
    QuadTree(float threshold, int N, double dt)
        : _threshold(threshold), _N(N), _dt(dt)
    {
        _nodes = new Node[100 * N];
    }

    void build(Body *bodies);

    void update(Body *body);

    void print(int node = 0, int indent = 0);

    const static double k;

private:
    Node *_nodes;

    int _next;
    float _threshold;
    int _N;
    double _size;
    double _dt;

    void insert(const Body& body, int node);

    void search(int node, Body *body, double& a_x, double& a_y);
};

#endif // NBODY_QUAD_TREE_H_
