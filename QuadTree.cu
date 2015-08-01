#include "QuadTree.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <assert.h>

#define DISTANCE(x1, y1, x2, y2) \
    (sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)))

#define MAX(x, y) (x > y ? (x) : (y))

const double k = 6.67384e-11;


__host__ __device__ void tree_print(Node *nodes, int node, int indent)
{
    if (nodes[node].status == Node::EMPTY) {
        return;
    }

    for (int i = 0; i < indent; i++) {
        printf("    ");
    }

    printf("(%.2lf, %.2lf), m = %.2lf\n", nodes[node].body.x,
        nodes[node].body.y, nodes[node].body.m);

    if (nodes[node].status == Node::INTERNAL) {
        for (int i = 0; i < 4; i++) {
            tree_print(nodes, nodes[node].children[i], indent + 1);
        }
    }
}


__host__ __device__ void tree_build(Body *bodies, Node *nodes, int N, double *size)
{
    // find the min and max
    double xmin = std::numeric_limits<double>::max();
    double ymin = std::numeric_limits<double>::max();
    double xmax = -std::numeric_limits<double>::max();
    double ymax = -std::numeric_limits<double>::max();
    for (int i = 0; i < N; i++) {
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

    printf("[(%.4lf, %.4lf), (%.4lf, %.4lf)]\n", xmin, ymin, xmax, ymax);

    // root node
    nodes[0].x = xmin - 1.0;
    nodes[0].y = ymin - 1.0;
    nodes[0].w = xmax - xmin + 2.0;
    nodes[0].h = ymax - ymin + 2.0;
    nodes[0].status = Node::EMPTY;
    *size = MAX(nodes[0].w, nodes[0].h);

    // next empty node
    next = 1;

    // insert nodes one by one
    for (int i = 0; i < N; i++) {
// printf("now insert body %d...\n", bodies[i].idx);
        tree_insert(bodies + i, 0, nodes, &next);
    }

    int cnt = 0;
    for (int i = 0; i < next; i++) {
        if (nodes[i].status == Node::EXTERNAL) {
            cnt++;
        }
    }
    assert(cnt == N);

    printf("build done\n");

    return size;
}


__host__ __device__ void tree_update(Body *body, Node *nodes, double size, double threshold)
{
    // acceleration routine
    double a_x = 0;
    double a_y = 0;
    tree_search(0, body, &a_x, &a_y, nodes, size, threshold);

    // update positions
    body->vx += a_x * _dt;
    body->vy += a_y * _dt;
    body->x += body->vx * _dt;
    body->y += body->vy * _dt;

    // reverse velocity if out of bound
    if (body->x < nodes[0].x || body->x > nodes[0].x + nodes[0].w ||
        body->y < nodes[0].y || body->y > nodes[0].y + nodes[0].h) {
        body->vx = -body->vx;
        body->vy = -body->vy;
    }
}


__host__ __device__ void tree_insert(Body *body, int node, Node *nodes, int *next)
{
    if (nodes[node].status == Node::EMPTY) { // is empty node
// printf("node %d is empty\n", node);
// printf("insert body %d at %d\n", body.idx, node);
        nodes[node].body = body;
        nodes[node].status = Node::EXTERNAL;
        return;
    }

    // is internal or external node
    Body tmp = nodes[node].body;

    // update total mass
    nodes[node].body.m += body.m;

    // update center of mass
    nodes[node].body.x = (tmp.x * tmp.m + body->x * body->m) / nodes[node].body.m;
    nodes[node].body.y = (tmp.y * tmp.m + body->y * body->m) / nodes[node].body.m;

    if (nodes[node].status == Node::EXTERNAL) {
// printf("node %d is external\n", node);
        // expand this node
        for (int i = 0; i < 4; i++) {
            nodes[node].children[i] = (*next)++;
            int child = nodes[node].children[i];
            nodes[child].status = Node::EMPTY;
            nodes[child].x = nodes[node].x + (i % 2) * 0.5 * nodes[node].w;
            nodes[child].y = nodes[node].y + (i / 2) * 0.5 * nodes[node].h;
            nodes[child].w = 0.5 * nodes[node].w;
            nodes[child].h = 0.5 * nodes[node].h;
// printf("node %d: [%.2lf, %.2lf, %.2lf, %.2lf]\n", node, nodes[node].x, nodes[node].y, nodes[node].w, nodes[node].h);
// printf("add child %d: [%.2lf, %.2lf, %.2lf, %.2lf]\n", child, nodes[child].x, nodes[child].y, nodes[child].w, nodes[child].h);
        }
        nodes[node].status = Node::INTERNAL;

        // insert body in the appropriate child
        for (int i = 0; i < 4; i++) {
            int child = nodes[node].children[i];
            if (nodes[child].inside(tmp)) {
                tree_insert(tmp, child, nodes, next);
                break;
            }
        }
    } else {
// printf("node %d is internal\n", node);
    }

    // insert body in the appropriate child
    for (int i = 0; i < 4; i++) {
        int child = nodes[node].children[i];
        if (nodes[child].inside(body)) {
            tree_insert(body, child, nodes, next);
            return;
        }
    }

    printf("[error] body %d (%.4lf, %.4lf) isn't in any subtree of #%d [%.4lf, %.4lf, %.4lf, %.4lf]\n",
        body->idx, body->x, body->y, node, nodes[node].x, nodes[node].y, nodes[node].w, nodes[node].h);
    exit(1);
}


__host__ __device__ void tree_search(int node, Body *body, double *a_x,
    double *a_y, Node *nodes, double size, double threshold)
{
    if (nodes[node].status == Node::EXTERNAL) {
        if (nodes[node].body.idx != body->idx) {
            double dis = DISTANCE(body->x, body->y, nodes[node].body.x, nodes[node].body.y);
            double a = k * nodes[node].body.m / (dis * dis * dis);
            *a_x += a * (nodes[node].body.x - body->x);
            *a_y += a * (nodes[node].body.y - body->y);
        }
        return;
    }

    double dis = DISTANCE(body->x, body->y, nodes[node].body.x, nodes[node].body.y);
    if (size / dis < threshold) { // treat as single body
        double a = k * nodes[node].body.m / (dis * dis * dis);
        *a_x += a * (nodes[node].body.x - body->x);
        *a_y += a * (nodes[node].body.y - body->y);
        return;
    }

    for (int i = 0; i < 4; i++) {
        int child = nodes[node].children[i];
        if (nodes[child].status != Node::EMPTY) {
            tree_search(child, body, a_x, a_y, nodes, size, threshold);
        }
    }
}
