#ifndef NBODY_QUAD_TREE_H_
#define NBODY_QUAD_TREE_H_

#include "util.h"

#define EMPTY -1

struct Point
{
    double x;
    double y;
};

struct Node
{
    Point center;
    int size;
    int children[4]; // index of children, -1 if null

    Node()
    {
        children[0] = children[1] = children[2] = children[3] = EMPTY;
    }
};

class QuadTree
{
public:
    void build(const std::vector<Point>& points);
    void setThreshold(double threshold);
    double force(int body);

private:
    std::vector<Node> _nodes;
};

#endif // NBODY_QUAD_TREE_H_
