#include "QuadTree.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <assert.h>

#define DISTANCE(x1, y1, x2, y2) \
    (sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)))

#define MAX(x, y) (x > y ? (x) : (y))

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
    // find the min and max
    double xmin = std::numeric_limits<double>::max();
    double ymin = std::numeric_limits<double>::max();
    double xmax = -std::numeric_limits<double>::max();
    double ymax = -std::numeric_limits<double>::max();
    for (int i = 0; i < _N; i++) {
        if (bodies[i].x < xmin) {
            xmin = bodies[i].x;
        } else if (bodies[i].x > xmax) {
            xmax = bodies[i].x;
        }
        if (bodies[i].y < ymin) {
            ymin = bodies[i].y;
        } else if (bodies[i].y > ymax) {
            ymax = bodies[i].y;
        }
    }

    // printf("[(%.4lf, %.4lf), (%.4lf, %.4lf)]\n", xmin, ymin, xmax, ymax);

    // root node
    _nodes[0].x = xmin - 1.0;
    _nodes[0].y = ymin - 1.0;
    _nodes[0].w = xmax - xmin + 2.0;
    _nodes[0].h = ymax - ymin + 2.0;
    _nodes[0].status = Node::EMPTY;
    _size = MAX(_nodes[0].w, _nodes[0].h);

    // next empty node
    _next = 1;

    // insert nodes one by one
    for (int i = 0; i < _N; i++) {
// printf("now insert body %d...\n", bodies[i].idx);
        insert(bodies[i], 0);
    }

    int cnt = 0;
    for (int i = 0; i < _next; i++) {
        if (_nodes[i].status == Node::EXTERNAL) {
            cnt++;
        }
    }
    assert(cnt == _N);
}


void QuadTree::update(Body *body)
{
    // acceleration routine
    double a_x = 0;
    double a_y = 0;
    search(0, body, a_x, a_y);

    // update positions
    body->vx += a_x * _dt;
    body->vy += a_y * _dt;
    body->x += body->vx * _dt;
    body->y += body->vy * _dt;

    // reverse velocity if out of bound
    if (body->x < _nodes[0].x || body->x > _nodes[0].x + _nodes[0].w ||
        body->y < _nodes[0].y || body->y > _nodes[0].y + _nodes[0].h) {
        body->vx = -body->vx;
        body->vy = -body->vy;
    }
}


void QuadTree::insert(const Body& body, int node)
{
    if (_nodes[node].status == Node::EMPTY) { // is empty node
// printf("node %d is empty\n", node);
// printf("insert body %d at %d\n", body.idx, node);
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
            return;
        }
    }

    printf("[error] body %d (%.4lf, %.4lf) isn't in any subtree of #%d [%.4lf, %.4lf, %.4lf, %.4lf]\n",
        body.idx, body.x, body.y, node, _nodes[node].x, _nodes[node].y, _nodes[node].w, _nodes[node].h);
    exit(1);
}


void QuadTree::search(int node, Body *body, double& a_x, double& a_y)
{
    if (_nodes[node].status == Node::EXTERNAL) {
        if (_nodes[node].body.idx != body->idx) {
            double dis = DISTANCE(body->x, body->y, _nodes[node].body.x, _nodes[node].body.y);
            double a = k * _nodes[node].body.m / (dis * dis * dis);
            a_x += a * (_nodes[node].body.x - body->x);
            a_y += a * (_nodes[node].body.y - body->y);
        }
        return;
    }

    double dis = DISTANCE(body->x, body->y, _nodes[node].body.x, _nodes[node].body.y);
    if (_size / dis < _threshold) { // treat as single body
        double a = k * _nodes[node].body.m / (dis * dis * dis);
        a_x += a * (_nodes[node].body.x - body->x);
        a_y += a * (_nodes[node].body.y - body->y);
        return;
    }

    for (int i = 0; i < 4; i++) {
        int child = _nodes[node].children[i];
        if (_nodes[child].status != Node::EMPTY) {
            search(child, body, a_x, a_y);
        }
    }
}
