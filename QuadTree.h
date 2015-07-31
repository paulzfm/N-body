#ifndef NBODY_QUAD_TREE_H_
#define NBODY_QUAD_TREE_H_

// #include "util.h"

struct Body
{
    int idx;  // global unique index
    double x; // x-coordinate
    double y; // y-coordinate
    double vx; // x-velocity
    double vy; // y-velocity
    double m;  // mass
};

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
    QuadTree(float threshold, double xmin, double ymin,
        double width, double height, int N)
        : _threshold(threshold), _N(N), _size(width)
    {
        _nodes = new Node[100 * N];
        _nodes[0].x = xmin;
        _nodes[0].y = ymin;
        _nodes[0].w = width;
        _nodes[0].h = height;
        _next = 1;
    }

    void build(Body *bodies);

    void search(int node, const Body& body, double& a_x, double& a_y, int& cnt);

    void print(int node = 0, int indent = 0);

    const static double k;

private:
    Node *_nodes;

    int _next;
    float _threshold;
    int _N;
    double _size;

    void insert(const Body& body, int node);
};

#endif // NBODY_QUAD_TREE_H_
