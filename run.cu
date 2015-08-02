#include "run.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

extern int N;
extern int n;
extern double threshold;
extern double dt;

#define k 6.67384e-11
#define BLOCK_SIZE 512


// body locales inside this node?
__host__ __device__ bool inside(Node *node, Body* body)
{
    return (node->x <= body->x && body->x <= node->x + node->w) &&
        (node->y <= body->y && body->y <= node->y + node->h);
}


// "class" QuadTree
void tree_build(Body *bodies, Node *nodes, int N);

__host__ __device__ void tree_update(Body *body, Node *nodes,
    double threshold, double dt);

__host__ __device__ void tree_print(Node *nodes, int node, int indent);

// help functions
void tree_insert(Body *body, int node, Node *nodes, int *next);

// do NOT use this
__host__ __device__ void tree_search_recursive(int node, Body *body, double *a_x,
    double *a_y, Node *nodes, double threshold);

// use this
__host__ __device__ void tree_search(int node, Body *body, double *a_x,
    double *a_y, Node *nodes, double threshold);


// pthread worker
void *thread_worker(void *args)
{
    TaskParam *param = (TaskParam*)args;
    for (int i = param->start; i < param->end; i++) {
        tree_update(param->bodies + i, param->tree, threshold, dt);
    }

    pthread_exit(NULL);
}


// pthread version
void run_pthread_version(int i, int num_threads, Body *bodies,
    float *elapsed_time, Node *tree, pthread_t *threads, TaskParam *param)
{
    cudaEvent_t start, stop;
    float build_time;
    int width = ceil((double)N / num_threads);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // build tree
    tree_build(bodies, tree, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&build_time, start, stop);

    printf("[pthread] build tree: %.4f ms\n", build_time);

    cudaEventRecord(start);

    // create threads to compute
    for (int j = 0; j < num_threads; j++) {
        param[j].start = j * width;
        if (j == num_threads - 1) {
            param[j].end = N;
        } else {
            param[j].end = (j + 1) * width;
        }
        param[j].bodies = bodies;
        param[j].tree = tree;
        pthread_create(threads + j, NULL, thread_worker, param + j);
    }

    // wait
    for (int j = 0; j < num_threads; j++) {
        pthread_join(threads[j], NULL);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// cuda worker
__global__ void cuda_worker(Node *tree, Body *bodies, double threshold,
    int N, double dt)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    tree_update(bodies + i, tree, threshold, dt);
}


// cuda version
void run_cuda_version(int i, Body *bodies,
    float *elapsed_time, Node *tree, Body *d_bodies, Node *d_tree)
{
    cudaEvent_t start, stop;
    cudaError_t err;
    float build_time;
    int block = ceil(N / (double)BLOCK_SIZE);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // build tree
    tree_build(bodies, tree, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&build_time, start, stop);

    printf("[cuda] build tree: %.4f ms\n", build_time);

    err = cudaMemcpy(d_bodies, bodies, sizeof(Body) * N, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[error] when copy bodies: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMemcpy(d_tree, tree, sizeof(Node) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[error] when copy tree: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaEventRecord(start);

    cuda_worker<<<block, BLOCK_SIZE>>>(d_tree, d_bodies, threshold, N, dt);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed_time, start, stop);

    err = cudaMemcpy(bodies, d_bodies, sizeof(Body) * N, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[error] when copy back: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// QuadTree functions
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


void tree_build(Body *bodies, Node *nodes, int N)
{
    // find the min and max
    double xmin = 1.0e10;
    double ymin = 1.0e10;
    double xmax = -1.0e10;
    double ymax = -1.0e10;
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

    // root node
    double eps = 0.001;
    nodes[0].x = xmin - eps;
    nodes[0].y = ymin - eps;
    nodes[0].w = xmax - xmin + 2 * eps;
    nodes[0].h = ymax - ymin + 2 * eps;
    nodes[0].status = Node::EMPTY;

    // next empty node
    int next = 1;

    // insert nodes one by one
    for (int i = 0; i < N; i++) {
        tree_insert(bodies + i, 0, nodes, &next);
    }
}


__host__ __device__ void tree_update(Body *body, Node *nodes,
    double threshold, double dt)
{
    // acceleration routine
    double a_x = 0;
    double a_y = 0;
    tree_search(0, body, &a_x, &a_y, nodes, threshold);

    // update positions
    body->vx += a_x * dt;
    body->vy += a_y * dt;
    body->x += body->vx * dt;
    body->y += body->vy * dt;

    // reverse velocity if out of bound
    if (body->x < nodes[0].x || body->x > nodes[0].x + nodes[0].w ||
        body->y < nodes[0].y || body->y > nodes[0].y + nodes[0].h) {
        body->vx = -body->vx;
        body->vy = -body->vy;
    }
}


void tree_insert(Body *body, int node, Node *nodes, int *next)
{
    if (nodes[node].status == Node::EMPTY) { // is empty node
        nodes[node].body = *body;
        nodes[node].status = Node::EXTERNAL;
        return;
    }

    // is internal or external node
    Body tmp = nodes[node].body;

    // update total mass
    nodes[node].body.m += body->m;

    // update center of mass
    nodes[node].body.x = (tmp.x * tmp.m + body->x * body->m) / nodes[node].body.m;
    nodes[node].body.y = (tmp.y * tmp.m + body->y * body->m) / nodes[node].body.m;

    if (nodes[node].status == Node::EXTERNAL) {
        // expand this node
        for (int i = 0; i < 4; i++) {
            nodes[node].children[i] = (*next)++;
            int child = nodes[node].children[i];
            nodes[child].status = Node::EMPTY;
            nodes[child].x = nodes[node].x + (i % 2) * 0.5 * nodes[node].w;
            nodes[child].y = nodes[node].y + (i / 2) * 0.5 * nodes[node].h;
            nodes[child].w = 0.5 * nodes[node].w;
            nodes[child].h = 0.5 * nodes[node].h;
        }
        nodes[node].status = Node::INTERNAL;

        // insert body in the appropriate child
        for (int i = 0; i < 4; i++) {
            int child = nodes[node].children[i];
            if (inside(nodes + child, &tmp)) {
                tree_insert(&tmp, child, nodes, next);
                break;
            }
        }
    }

    // insert body in the appropriate child
    for (int i = 0; i < 4; i++) {
        int child = nodes[node].children[i];
        if (inside(nodes + child, body)) {
            tree_insert(body, child, nodes, next);
            return;
        }
    }

    // fprintf(stderr, "[error] body %d (%.4lf, %.4lf) isn't in any subtree of #%d [%.4lf, %.4lf, %.4lf, %.4lf]\n",
        // body->idx, body->x, body->y, node, nodes[node].x, nodes[node].y, nodes[node].w, nodes[node].h);
    // exit(1);
}


// @recursive
__host__ __device__ void tree_search_recursive(int node, Body *body, double *a_x,
    double *a_y, Node *nodes, double threshold)
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
    if (nodes[node].w / dis < threshold) { // treat as single body
        double a = k * nodes[node].body.m / (dis * dis * dis);
        *a_x += a * (nodes[node].body.x - body->x);
        *a_y += a * (nodes[node].body.y - body->y);
        return;
    }

    for (int i = 0; i < 4; i++) {
        int child = nodes[node].children[i];
        if (nodes[child].status != Node::EMPTY) {
            tree_search_recursive(child, body, a_x, a_y, nodes, threshold);
        }
    }
}


// @loop
__host__ __device__ void tree_search(int node, Body *body, double *a_x,
    double *a_y, Node *nodes, double threshold)
{
    int stack[64];
    stack[0] = node;
    int stack_ptr = 0;

    while (stack_ptr >= 0) {
        node = stack[stack_ptr];

        if (nodes[node].status == Node::EXTERNAL) {
            if (nodes[node].body.idx != body->idx) {
                double dis = DISTANCE(body->x, body->y, nodes[node].body.x, nodes[node].body.y);
                double a = k * nodes[node].body.m / (dis * dis * dis);
                *a_x += a * (nodes[node].body.x - body->x);
                *a_y += a * (nodes[node].body.y - body->y);
            }
            // pop
            --stack_ptr;
            continue;
        }

        double dis = DISTANCE(body->x, body->y, nodes[node].body.x, nodes[node].body.y);
        if (nodes[node].w / dis < threshold) { // treat as single body
            double a = k * nodes[node].body.m / (dis * dis * dis);
            *a_x += a * (nodes[node].body.x - body->x);
            *a_y += a * (nodes[node].body.y - body->y);
            // pop
            --stack_ptr;
            continue;
        }

        // push children
        for (int i = 3; i >= 0; i--) {
            int child = nodes[node].children[i];
            if (nodes[child].status != Node::EMPTY) {
                stack[stack_ptr++] = child;
            }
        }

        --stack_ptr;
    }
}
