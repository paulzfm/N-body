#include "run.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

extern int N;
extern int n;
extern double threshold;
extern double dt;

const double k = 6.67384e-11;

// body locales inside this node?
__host__ __device__ bool inside(Node *node, Body* body)
{
    return (node->x <= body->x && body->x <= node->x + node->w) &&
        (node->y <= body->y && body->y <= node->y + node->h);
}

// "class" QuadTree: both __host__ and __device__
__host__ __device__ void tree_build(Body *bodies, Node *nodes, int N, double *size);

__host__ __device__ void tree_update(Body *body, Node *nodes, double size,
    double threshold, double dt);

__host__ __device__ void tree_print(Node *nodes, int node, int indent);

// help functions
__host__ __device__ void tree_insert(Body *body, int node, Node *nodes, int *next);

__host__ __device__ void tree_search(int node, Body *body, double *a_x,
    double *a_y, Node *nodes, double size, double threshold);

__host__ __device__ void tree_search_loop(int node, Body *body, double *a_x,
        double *a_y, Node *nodes, double size, double threshold);


// pthread worker
void *thread_worker(void *args)
{
    TaskParam *param = (TaskParam*)args;
    for (int i = param->start; i < param->end; i++) {
        tree_update(param->bodies + i, param->tree, param->size, threshold, dt);
    }

    pthread_exit(NULL);
}

// pthread version
void run_pthread_version(int i, int num_threads, Body *bodies,
    float *elapsed_time, Node *tree)
{
    pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    TaskParam *param = (TaskParam*)malloc(sizeof(TaskParam) * num_threads);
    int width = ceil((float)N / num_threads);
    double size = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // build tree
    tree_build(bodies, tree, N, &size);

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
        param[j].size = size;
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

    free(threads);
    free(param);
}

// cuda worker
__global__ void cuda_worker(Node *tree, Body *bodies, double threshold,
    double size, int N, double dt)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    tree_update(bodies + i, tree, size, threshold, dt);
}

__global__ void test(Node *tree, Body *bodies, int N)
{
    // for (int i = 0; i < 945; i++) {
    //     printf("%d %.4lf %.4lf %.4lf %.4lf %4.lf\n",
    //         bodies[i].idx, bodies[i].x, bodies[i].y, bodies[i].vx, bodies[i].vy, bodies[i].m);
    // }

    // for (int i = 0; i < 1000; i++) {
    //     printf("%d# (%d) {%d,%d,%d,%d} [%.4lf, %.4lf, %.4lf, %.4lf], %d# (%.4lf, %.4lf) (%.4lf, %.4lf) %.4lf\n", i,
    //         tree[i].status, tree[i].children[0], tree[i].children[1], tree[i].children[2], tree[i].children[3],
    //         tree[i].x, tree[i].y, tree[i].w, tree[i].h, tree[i].body.idx, tree[i].body.x, tree[i].body.y,
    //         tree[i].body.vx, tree[i].body.vy, tree[i].body.m);
    // }

    // for (int i = 1000; i < 2000; i++) {
    //     printf("%d# (%d) {%d,%d,%d,%d} [%.4lf, %.4lf, %.4lf, %.4lf], %d# (%.4lf, %.4lf) (%.4lf, %.4lf) %.4lf\n", i,
    //         tree[i].status, tree[i].children[0], tree[i].children[1], tree[i].children[2], tree[i].children[3],
    //         tree[i].x, tree[i].y, tree[i].w, tree[i].h, tree[i].body.idx, tree[i].body.x, tree[i].body.y,
    //         tree[i].body.vx, tree[i].body.vy, tree[i].body.m);
    // }

    for (int i = 2000; i < 2933; i++) {
        printf("%d# (%d) {%d,%d,%d,%d} [%.4lf, %.4lf, %.4lf, %.4lf], %d# (%.4lf, %.4lf) (%.4lf, %.4lf) %.4lf\n", i,
            tree[i].status, tree[i].children[0], tree[i].children[1], tree[i].children[2], tree[i].children[3],
            tree[i].x, tree[i].y, tree[i].w, tree[i].h, tree[i].body.idx, tree[i].body.x, tree[i].body.y,
            tree[i].body.vx, tree[i].body.vy, tree[i].body.m);
    }
}

// cuda version
void run_cuda_version(int i, Body *bodies,
    float *elapsed_time, Node *tree, Body *d_bodies, Node *d_tree)
{
    // cudaEvent_t start, stop;

    cudaError_t err;
    double size;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);

    // build tree
    tree_build(bodies, tree, N, &size);

    err = cudaMemcpy(d_bodies, bodies, sizeof(Body) * N, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cpy bodies: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMemcpy(d_tree, tree, sizeof(Node) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cpy tree: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // compute
    int block = ceil(N / 512.0);
    cuda_worker<<<block, 512>>>(d_tree, d_bodies, threshold, size, N, dt);
    // test<<<1, 1>>>(d_tree, d_bodies, N);
    cudaStreamSynchronize(0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "after calling: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(elapsed_time, start, stop);

    err = cudaMemcpy(bodies, d_bodies, sizeof(Body) * N, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cpy back: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
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


__host__ __device__ void tree_build(Body *bodies, Node *nodes, int N, double *size)
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
    nodes[0].x = xmin - 1.0;
    nodes[0].y = ymin - 1.0;
    nodes[0].w = xmax - xmin + 2.0;
    nodes[0].h = ymax - ymin + 2.0;
    nodes[0].status = Node::EMPTY;
    *size = MAX(nodes[0].w, nodes[0].h);

    // next empty node
    int next = 1;

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
}


__host__ __device__ void tree_update(Body *body, Node *nodes, double size,
    double threshold, double dt)
{
    // printf("before: (%.4lf, %.4lf)\n", body->x, body->y);

    // acceleration routine
    double a_x = 0;
    double a_y = 0;
    tree_search(0, body, &a_x, &a_y, nodes, size, threshold);

    // update positions
    body->vx += a_x * dt;
    body->vy += a_y * dt;
    body->x += body->vx * dt;
    body->y += body->vy * dt;
    // printf("after: (%.4lf, %.4lf)\n", body->x, body->y);

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
            if (inside(nodes + child, &tmp)) {
                tree_insert(&tmp, child, nodes, next);
                break;
            }
        }
    } else {
// printf("node %d is internal\n", node);
    }

    // insert body in the appropriate child
    for (int i = 0; i < 4; i++) {
        int child = nodes[node].children[i];
        if (inside(nodes + child, body)) {
            tree_insert(body, child, nodes, next);
            return;
        }
    }

    printf("[error] body %d (%.4lf, %.4lf) isn't in any subtree of #%d [%.4lf, %.4lf, %.4lf, %.4lf]\n",
        body->idx, body->x, body->y, node, nodes[node].x, nodes[node].y, nodes[node].w, nodes[node].h);
    // exit(1);
    assert(1 == 2);
}


// @recursive
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


// @loop
__host__ __device__ void tree_search_loop(int node, Body *body, double *a_x,
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
