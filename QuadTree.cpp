#include "QuadTree.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DISTANCE(x1, y1, x2, y2) \
    (sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)))

const double QuadTree::k = 6.67384e-11;

void QuadTree::print(int node, int indent)
{
    if (_nodes[node].status == Node::EMPTY) {
        return;
    }

    for (int i = 0; i < indent; i++) {
        printf("    ");
    }

    printf("(%.2lf, %.2lf), m = %.2lf\n", _nodes[node].body.x,
        _nodes[node].body.y, _nodes[node].body.m);

    if (_nodes[node].status == Node::INTERNAL) {
        for (int i = 0; i < 4; i++) {
            print(_nodes[node].children[i], indent + 1);
        }
    }
}

void QuadTree::build(Body *bodies)
{
    for (int i = 0; i < _N; i++) {
        insert(bodies[i], 0);
    }
}

void QuadTree::insert(const Body& body, int node)
{
    if (_nodes[node].status == Node::EMPTY) { // is empty node
// printf("node %d is empty\n", node);
        _nodes[node].body = body;
        _nodes[node].status = Node::EXTERNAL;
        return;
    }

    // is internal or external node
    Body tmp = _nodes[node].body;

    // update total mass
    _nodes[node].body.m += body.m;

    // update center of mass
    _nodes[node].body.x = (tmp.x * tmp.m + body.x * body.m) / _nodes[node].body.m;
    _nodes[node].body.y = (tmp.y * tmp.m + body.y * body.m) / _nodes[node].body.m;

    if (_nodes[node].status == Node::EXTERNAL) {
// printf("node %d is external\n", node);
        // expand this node
        for (int i = 0; i < 4; i++) {
            _nodes[node].children[i] = _next++;
            int child = _nodes[node].children[i];
            _nodes[child].status = Node::EMPTY;
            _nodes[child].x = _nodes[node].x + (i % 2) * 0.5 * _nodes[node].w;
            _nodes[child].y = _nodes[node].y + (i / 2) * 0.5 * _nodes[node].h;
            _nodes[child].w = 0.5 * _nodes[node].w;
            _nodes[child].h = 0.5 * _nodes[node].h;
// printf("node %d: [%.2lf, %.2lf, %.2lf, %.2lf]\n", node, _nodes[node].x, _nodes[node].y, _nodes[node].w, _nodes[node].h);
// printf("add child %d: [%.2lf, %.2lf, %.2lf, %.2lf]\n", child, _nodes[child].x, _nodes[child].y, _nodes[child].w, _nodes[child].h);
        }
        _nodes[node].status = Node::INTERNAL;

        // insert body in the appropriate child
        for (int i = 0; i < 4; i++) {
            int child = _nodes[node].children[i];
            if (_nodes[child].inside(tmp)) {
                insert(tmp, child);
                break;
            }
        }
    } else {
// printf("node %d is internal\n", node);
    }

    // insert body in the appropriate child
    for (int i = 0; i < 4; i++) {
        int child = _nodes[node].children[i];
        if (_nodes[child].inside(body)) {
            insert(body, child);
            break;
        }
    }
}

void QuadTree::search(int node, const Body& body, double& a_x, double& a_y, int& cnt)
{
    if (_nodes[node].status == Node::EXTERNAL) {
        if (_nodes[node].body.idx != body.idx) {
            cnt++;
            double dis = DISTANCE(body.x, body.y, _nodes[node].body.x, _nodes[node].body.y);
            double a = k * _nodes[node].body.m / (dis * dis * dis);
            a_x += a * (_nodes[node].body.x - body.x);
            a_y += a * (_nodes[node].body.y - body.y);
        }
        return;
    }

    double dis = DISTANCE(body.x, body.y, _nodes[node].body.x, _nodes[node].body.y);
    if ((double)_size / dis < _threshold) { // treat as single body
        // cnt++;
        fprintf(stderr, "Should not be here!\n");exit(1);
        double a = k * _nodes[node].body.m / (dis * dis * dis);
        a_x += a * (_nodes[node].body.x - body.x);
        a_y += a * (_nodes[node].body.y - body.y);
        return;
    }

    for (int i = 0; i < 4; i++) {
        int child = _nodes[node].children[i];
        if (_nodes[child].status != Node::EMPTY) {
            search(child, body, a_x, a_y);
        }
    }
}
