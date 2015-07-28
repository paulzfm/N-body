#ifndef NBODY_QUAD_TREE_H_
#define NBODY_QUAD_TREE_H_

#define EMPTY -1

struct Body
{
    int x;
    int y;
    int mass;
};

struct Rect
{
    double x;
    double y;
    double width;
    double height;
};

struct Node
{
    double mass;
    Point center;
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
