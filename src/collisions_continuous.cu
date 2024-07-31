#pragma once

#include "aabb.cu"
#include "utils.cu"
#include "bvh.cu"
#include "build_bvh.cu"
#include "figures.cu"
#include "collisions_static.cu"
//#include <gmp.h>
//#include "../include/cgbn/cgbn.h"
#include "../include/campari/multiplication.h"
#include <cmath>

template<typename T>
__global__
void make_moving_triangles(Triangle<T> *triangles, Triangle<T> *triangles_next, MovingTriangle<T> *moving_triangles,
                           int num_triangles) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_triangles) return;

    AABB<T> b_0 = triangles[idx].ComputeBBox();
    AABB<T> b_1 = triangles_next[idx].ComputeBBox();

    moving_triangles[idx] = MovingTriangle<T>(triangles[idx], triangles_next[idx]);
    AABB<T> b_new = moving_triangles[idx].ComputeBBox();
}

__device__ void
put_minimal_collision_tid(long2 *triangle_collisions, long3 *minimal_collisions_tids, int from_idx, int to_idx,
                          int collision_id) {
    minimal_collisions_tids[to_idx].x = triangle_collisions[from_idx].x;
    minimal_collisions_tids[to_idx].y = triangle_collisions[from_idx].y;
    minimal_collisions_tids[to_idx].z = collision_id;
}

template<typename T>
__device__ void
make_tetrahedron(long3 *minimal_collisions_tids, MovingTriangle<T> *moving_triangles,
                 MovingTetrahedron<T> *tetrahedrons,
                 int idx) {

    long3 *coll = minimal_collisions_tids + idx;
    MovingTriangle<T> *triangle0 = moving_triangles + coll->x;
    MovingTriangle<T> *triangle1 = moving_triangles + coll->y;

    switch (coll->z) {
        case 0 :
//            minimal_collisions_vids[to_idx] = make_long4(0, 1, 0, 1);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v0, triangle0->v1, triangle1->v0, triangle1->v1,
                                                     triangle0->dv0, triangle0->dv1, triangle1->dv0, triangle1->dv1);
            break;
        case 1:
//            minimal_collisions_vids[to_idx] = make_long4(0, 1, 0, 2);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v0, triangle0->v1, triangle1->v0, triangle1->v2,
                                                     triangle0->dv0, triangle0->dv1, triangle1->dv0, triangle1->dv2);
            break;
        case 2:
//            minimal_collisions_vids[to_idx] = make_long4(0, 1, 1, 2);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v0, triangle0->v1, triangle1->v1, triangle1->v2,
                                                     triangle0->dv0, triangle0->dv1, triangle1->dv1, triangle1->dv2);
            break;
        case 3 :
//            minimal_collisions_vids[to_idx] = make_long4(0, 2, 0, 1);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v0, triangle0->v2, triangle1->v0, triangle1->v1,
                                                     triangle0->dv0, triangle0->dv2, triangle1->dv0, triangle1->dv1);
            break;
        case 4:
//            minimal_collisions_vids[to_idx] = make_long4(0, 2, 0, 2);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v0, triangle0->v2, triangle1->v0, triangle1->v2,
                                                     triangle0->dv0, triangle0->dv2, triangle1->dv0, triangle1->dv2);
            break;
        case 5:
//            minimal_collisions_vids[to_idx] = make_long4(0, 2, 1, 2);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v0, triangle0->v2, triangle1->v1, triangle1->v2,
                                                     triangle0->dv0, triangle0->dv2, triangle1->dv1, triangle1->dv2);
            break;
        case 6 :
//            minimal_collisions_vids[to_idx] = make_long4(1, 2, 0, 1);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v1, triangle0->v2, triangle1->v0, triangle1->v1,
                                                     triangle0->dv1, triangle0->dv2, triangle1->dv0, triangle1->dv1);
            break;
        case 7:
//            minimal_collisions_vids[to_idx] = make_long4(1, 2, 0, 2);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v1, triangle0->v2, triangle1->v0, triangle1->v2,
                                                     triangle0->dv1, triangle0->dv2, triangle1->dv0, triangle1->dv2);
            break;
        case 8:
//            minimal_collisions_vids[to_idx] = make_long4(1, 2, 1, 2);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v1, triangle0->v2, triangle1->v1, triangle1->v2,
                                                     triangle0->dv1, triangle0->dv2, triangle1->dv1, triangle1->dv2);
            break;
        case 9 :
//            minimal_collisions_vids[to_idx] = make_long4(0, 1, 2, 0);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v0, triangle0->v1, triangle0->v2, triangle1->v0,
                                                     triangle0->dv0, triangle0->dv1, triangle0->dv2, triangle1->dv0);
            break;
        case 10:
//            minimal_collisions_vids[to_idx] = make_long4(0, 1, 2, 1);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v0, triangle0->v1, triangle0->v2, triangle1->v1,
                                                     triangle0->dv0, triangle0->dv1, triangle0->dv2, triangle1->dv1);
            break;
        case 11:
//            minimal_collisions_vids[to_idx] = make_long4(0, 1, 2, 2);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v0, triangle0->v1, triangle0->v2, triangle1->v2,
                                                     triangle0->dv0, triangle0->dv1, triangle0->dv2, triangle1->dv2);
            break;
        case 12 :
//            minimal_collisions_vids[to_idx] = make_long4(0, 0, 1, 2);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v0, triangle1->v0, triangle1->v1, triangle1->v2,
                                                     triangle0->dv0, triangle1->dv0, triangle1->dv1, triangle1->dv2);
            break;
        case 13:
//            minimal_collisions_vids[to_idx] = make_long4(1, 0, 1, 2);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v1, triangle1->v0, triangle1->v1, triangle1->v2,
                                                     triangle0->dv1, triangle1->dv0, triangle1->dv1, triangle1->dv2);
            break;
        case 14:
//            minimal_collisions_vids[to_idx] = make_long4(2, 0, 1, 2);
            tetrahedrons[idx] = MovingTetrahedron<T>(triangle0->v2, triangle1->v0, triangle1->v1, triangle1->v2,
                                                     triangle0->dv2, triangle1->dv0, triangle1->dv1, triangle1->dv2);
            break;
    }

}


template<typename T>
__device__ int traverseBVHPartial(long2 *collisionIndices, BVHNodePtr<T> root,
                           const AABB<T> &queryAABB, int queryObjectIdx,
                           BVHNodePtr<T> leaf, bool* triangles_to_check, int max_total_collisions,
                           int *counter, T threshold = 0, bool collision_ordering = true) {
    int num_collisions = 0;
    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    BVHNodePtr<T> stack[STACK_SIZE];
    BVHNodePtr<T> *stackPtr = stack;
    *stackPtr++ = nullptr; // push

    // Traverse nodes starting from the root.
    BVHNodePtr<T> node = root;
    do {
        // Check each child node for overlap.
        BVHNodePtr<T> childL = node->left;
        BVHNodePtr<T> childR = node->right;
        bool overlapL = checkOverlap<T>(queryAABB, childL->bbox, threshold);
        bool overlapR = checkOverlap<T>(queryAABB, childR->bbox, threshold);


        if (collision_ordering) {
            /*
               If we do not impose any order, then all potential collisions will be
               reported twice (i.e. the query object with the i-th colliding object
               and the i-th colliding object with the query). In order to avoid
               this, we impose an ordering, saying that an object can collide with
               another only if it comes before it in the tree. For example, if we
               are checking for the object 10, there is no need to check the subtree
               that has the objects that are before it, since they will already have
               been checked.
            */
            if (leaf >= childL->rightmost) {
                overlapL = false;
            }
            if (leaf >= childR->rightmost) {
                overlapR = false;
            }
        }

        // Query overlaps a leaf node => report collision.
        if (overlapL && childL->isLeaf()) {
            // Append the collision to the main array
            // Increase the number of detection collisions
            // num_collisions++;

            bool is_child_to_check = triangles_to_check[childL->idx];
            if (not is_child_to_check or queryObjectIdx < childL->idx) {

                int coll_idx = atomicAdd(counter, 1);
                coll_idx = coll_idx % max_total_collisions;
                long2 col = make_long2(min(queryObjectIdx, childL->idx),
                                       max(queryObjectIdx, childL->idx));

//            if (col.x == 1300 or col.y == 1300) {
//                printf("%d, %d\n", (int)col.x, (int)col.y);
//            }
//            printf("%d, %d\n", (int)col.x, (int)col.y);
                collisionIndices[coll_idx] = col;
                // collisionIndices[num_collisions % max_collisions] =
                // *collisionIndices++ =

                num_collisions++;
            }
        }

        if (overlapR && childR->isLeaf()) {

            bool is_child_to_check = triangles_to_check[childR->idx];
            if (not is_child_to_check or queryObjectIdx < childR->idx) {

                int coll_idx = atomicAdd(counter, 1);
                coll_idx = coll_idx % max_total_collisions;
                long2 col = make_long2(min(queryObjectIdx, childR->idx), max(queryObjectIdx, childR->idx));

//            if (col.x == 1300 or col.y == 1300) {
//                printf("%d, %d\n", (int)col.x, (int)col.y);
//            }
//            printf("%d, %d\n", (int)col.x, (int)col.y);
                collisionIndices[coll_idx] = col;
                num_collisions++;
            }

        }

        // Query overlaps an internal node => traverse.
        bool traverseL = (overlapL && !childL->isLeaf());
        bool traverseR = (overlapR && !childR->isLeaf());

        if (!traverseL && !traverseR) {
            node = *--stackPtr; // pop
        } else {
            node = (traverseL) ? childL : childR;
            if (traverseL && traverseR) {
                *stackPtr++ = childR; // push
            }
        }
    } while (node != nullptr);



    return num_collisions;
}


//
template<typename T>
__device__ bool
check_minimal_bbox_overlap(long2 *triangle_collisions, MovingTriangle<T> *moving_triangles,
                           int from_idx, int collision_id) {


    long2 *coll = triangle_collisions + from_idx;
    MovingTriangle<T> *triangle0 = moving_triangles + coll->x;
    MovingTriangle<T> *triangle1 = moving_triangles + coll->y;


    if (shareVertex(*triangle0, *triangle1))
        return false;

    AABB<T> bbox0;
    AABB<T> bbox1;
    switch (collision_id) {
        case 0 :
            bbox0 = triangle0->ComputeBBoxEdge(0, 1);
            bbox1 = triangle1->ComputeBBoxEdge(0, 1);
            break;
        case 1:
            bbox0 = triangle0->ComputeBBoxEdge(0, 1);
            bbox1 = triangle1->ComputeBBoxEdge(0, 2);
            break;
        case 2:
            bbox0 = triangle0->ComputeBBoxEdge(0, 1);
            bbox1 = triangle1->ComputeBBoxEdge(1, 2);
            break;
        case 3 :
            bbox0 = triangle0->ComputeBBoxEdge(0, 2);
            bbox1 = triangle1->ComputeBBoxEdge(0, 1);
            break;
        case 4:
            bbox0 = triangle0->ComputeBBoxEdge(0, 2);
            bbox1 = triangle1->ComputeBBoxEdge(0, 2);
            break;
        case 5:
            bbox0 = triangle0->ComputeBBoxEdge(0, 2);
            bbox1 = triangle1->ComputeBBoxEdge(1, 2);
            break;
        case 6 :
            bbox0 = triangle0->ComputeBBoxEdge(1, 2);
            bbox1 = triangle1->ComputeBBoxEdge(0, 1);
            break;
        case 7:
            bbox0 = triangle0->ComputeBBoxEdge(1, 2);
            bbox1 = triangle1->ComputeBBoxEdge(0, 2);
            break;
        case 8:
            bbox0 = triangle0->ComputeBBoxEdge(1, 2);
            bbox1 = triangle1->ComputeBBoxEdge(1, 2);
            break;
        case 9 :
            bbox0 = triangle0->ComputeBBox();
            bbox1 = triangle1->ComputeBBoxVertex(0);
            break;
        case 10:
            bbox0 = triangle0->ComputeBBox();
            bbox1 = triangle1->ComputeBBoxVertex(1);
            break;
        case 11:
            bbox0 = triangle0->ComputeBBox();
            bbox1 = triangle1->ComputeBBoxVertex(2);
            break;
        case 12 :
            bbox0 = triangle0->ComputeBBoxVertex(0);
            bbox1 = triangle1->ComputeBBox();
            break;
        case 13:
            bbox0 = triangle0->ComputeBBoxVertex(1);
            bbox1 = triangle1->ComputeBBox();
            break;
        case 14:
            bbox0 = triangle0->ComputeBBoxVertex(2);
            bbox1 = triangle1->ComputeBBox();
            break;
    }

    bool overlap = checkOverlap<T>(bbox0, bbox1);
//    printf("collx: %d, coll.y: %d, type: %d, overlap: %d\n", (int)coll->x, (int)coll->y, (int)collision_id, (int)overlap);


    return overlap;
}
//

template<typename T>
__global__
void
findMinimalCollisions(long2 *triangle_collisions, MovingTriangle<T> *moving_triangles, long3 *minimal_collisions_tids,
                      int num_collisions, int *counter) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;

    int collision_id = blockIdx.y;


    bool overlap = check_minimal_bbox_overlap<T>(triangle_collisions, moving_triangles, idx, collision_id);
    if (not overlap)
        return;

    int to_idx = atomicAdd(counter, 1);
    put_minimal_collision_tid(triangle_collisions, minimal_collisions_tids, idx, to_idx, collision_id);
}

template<typename T>
__global__
void
checkMinimalCollisions(long2 *triangle_collisions, MovingTriangle<T> *moving_triangles, bool *minimal_collisions_flags,
                       int num_collisions) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;

    int collision_id = blockIdx.y;



//    if (triangle_collisions[idx].x != 0 or triangle_collisions[idx].y != 1 or collision_id != 8) return;

    bool overlap = check_minimal_bbox_overlap<T>(triangle_collisions, moving_triangles, idx, collision_id);
    if (not overlap)
        return;

    int to_idx = idx * 15 + collision_id;
    minimal_collisions_flags[to_idx] = true;

}

template<typename T>
__global__
void
copyMinimalCollisions(long2 *triangle_collisions, long3 *minimal_collisions_tids,
                      bool *minimal_collision_flags,
                      int num_collisions, int *counter) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;
    int collision_id = blockIdx.y;

    int flag_idx = idx * 15 + collision_id;
    if (not minimal_collision_flags[flag_idx]) return;

////    // TODO: REMOVE
//    if (triangle_collisions[idx].x != 0 or triangle_collisions[idx].y != 1 or collision_id != 0) {
//        return;
//    }

//    double m1 =  -6.1585009098052978515625000000000000000000e-04;
//    double m2 =  -6.6852569580078125000000000000000000000000e-04;
//    double mult1 = m1 * m2;
//    printf("DOUBLE: mult: %.40e\n", mult1);
//
//    double mult2[2];
//    certifiedMul<1,1,2>(&m1, &m2, &mult2[0]);
//    printf("certifiedMul: mult[0]: %.40e\n", mult2[0]);
//    printf("certifiedMul: mult[1]: %.40e\n", mult2[1]);


    int to_idx = atomicAdd(counter, 1);


    put_minimal_collision_tid(triangle_collisions, minimal_collisions_tids, idx, to_idx, collision_id);
}


template<typename T>
__global__
void make_tetrahedrons(long3 *minimal_collisions_tids, MovingTriangle<T> *moving_triangles,
                       MovingTetrahedron<T> *tetrahedrons, int num_collisions) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;

    make_tetrahedron<T>(minimal_collisions_tids, moving_triangles, tetrahedrons, idx);
}

template<typename T>
__global__
void
getCubicCoeffs(MovingTetrahedron<T> *__restrict__ tetrahedrons, vec4<T> *__restrict__ coefs, long3 *minimal_collisions,
               int num_collisions) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;

    long3 *mcoll = minimal_collisions + idx;

    MovingTetrahedron<T> tth = tetrahedrons[idx];
//    printf("v0: {%g, %g, %g}\n"
//           "v1: {%g, %g, %g}\n"
//           "v2: {%g, %g, %g}\n"
//           "v3: {%g, %g, %g}\n",
//           tth.v0.x, tth.v0.y, tth.v0.z,
//           tth.v1.x, tth.v1.y, tth.v1.z,
//           tth.v2.x, tth.v2.y, tth.v2.z,
//           tth.v3.x, tth.v3.y, tth.v3.z
//    );

    MovingTetraMatrix<T> matrix = MovingTetraMatrix<T>(tetrahedrons + idx);

//    if (mcoll->z != 5) {
//        return;
//    }



    coefs[idx] = matrix.getCubicCoeffs();

//    printf("f1: %d, f2: %d, f3: %d\n"
//           "c3: %.60e, c2: %.60e, c1: %.60e, c0: %.60e\n", (int) mcoll->x, (int) mcoll->y, (int) mcoll->z, coefs[idx].x,
//           coefs[idx].y, coefs[idx].z, coefs[idx].w);


}


template<typename T>
__device__ void
add_root(vec3<T> *__restrict__ roots_row, T root, int &n_roots) {
    if (root < -EPSILON or root > 1 + EPSILON) return;

//    printf("ADD ROOT: %g\n", root);

//    printf("ADD ROOT: %.60e\n", root);
    if (n_roots == 0) {
        roots_row->x = root;
    } else if (n_roots == 1) {
        roots_row->y = root;
    } else if (n_roots == 2) {
        roots_row->z = root;
    }
    n_roots++;
}

//template<typename T>
//__device__ double
//mypow3(double& a, double* temp1) {
//
//}

__device__ double
computeG(double &a, double &b, double &c, double &d, double *temp1,  double *temp2) {

//    double g = (((2.0 * pow(b, 3)) / pow(a, 3)) - ((9.0 * b * c) / pow(a, 2)) + (27.0 * d / a)) / 27.0;

    certifiedMul<1, 1, 2>(&b, &b, temp1);
    certifiedMul<2, 1, 2>(temp1, &b, temp2);
    double b3 = 2*temp2[0];

    certifiedMul<1, 1, 2>(&a, &a, temp1);
    double a2 = temp1[0];
    certifiedMul<2, 1, 2>(temp1, &a, temp2);
    double a3 = temp2[0];
    double p1 = b3 / a3;


//    certifiedMul<1,1,2>(&b, &c, &temp1[0]);
    double p2 = (b * c * 9) / a2;
    double p3 = 27.0 * d / a;
    double gtemp = (p1 - p2 + p3) / 27.0;
    return gtemp;

}


__device__ double
computeH(double &g, double &f, double *temp1,  double *temp2) {
//    double h = (pow(g, 2) / 4.0 + (pow(f, 3)) / 27.0);

    certifiedMul<1, 1, 2>(&f, &f,  temp1);
    certifiedMul<2, 1, 2>(temp1, &f, temp2);
    double f3 = temp2[0];
    double h = pow(g, 2) / 4.0 + f3 / 27.0;
    return h;

}



template<typename T>
__global__
void solveCubic(vec4<T> *__restrict__ coefs, vec3<T> *__restrict__ roots, int num_collisions) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;
    double root = -1;

    double a = coefs[idx].x;
    double b = coefs[idx].y;
    double c = coefs[idx].z;
    double d = coefs[idx].w;

    double temp1[2];
    double temp2[2];

    vec3<T> *roots_row = roots + idx;
    int n_roots = 0;

//    printf("a: %.20g\tb: %.20g\tc: %.20g\td: %.20g\n", a, b, c, d);


    if (a == 0 and b == 0) {
//        printf("Linear\n");
        double x1 = -d / c;
        if (x1 >= -EPSILON and x1 <= 1 + EPSILON)
            root = x1;
    } else if (abs(a) < EPSILON) {
//        printf("Quadratic\n");
        double D = c * c - 4.0 * b * d;
        if (D >= 0) {
            D = sqrt(D);
            double x1 = (-c + D) / (2.0 * b);
            double x2 = (-c - D) / (2.0 * b);

            add_root<T>(roots_row, x1, n_roots);
            add_root<T>(roots_row, x2, n_roots);
        }
    } else {
//        printf("Cubic\n");

//        double f = ((3.0 * c / a) - ((b * b) / (a * a))) / 3.0;
//        double g = (((2.0 * (b * b * b)) / (a * a * a)) - ((9.0 * b * c) / (a * a)) + (27.0 * d / a)) / 27.0;
//        double h = ((g * g) / 4.0 + (f * f * f) / 27.0);



        double f = ((3.0 * c / a) - ((b * b) / (a * a))) / 3.0;
//        double g = (((2.0 * pow(b, 3)) / pow(a, 3)) - ((9.0 * b * c) / pow(a, 2)) + (27.0 * d / a)) / 27.0;
        double g =computeG(a, b, c, d, &temp1[0], &temp2[0]);
//        double h = (pow(g, 2) / 4.0 + (pow(f, 3)) / 27.0);
        double h =  computeH(g, f, &temp1[0], &temp2[0]);


        if (f == 0 and g == 0 and h == 0) {
            float k = (d / a);
            double x1;

            if (k >= 0) {
                x1 = cbrt(k) * -1;
            } else {
                x1 = cbrt(-k);
            }


            add_root<T>(roots_row, x1, n_roots);
        } else if (h <= 0) {
            double i = sqrt(((g * g) / 4.0) - h);
            double j = cbrt(i);
            double k = acos(-(g / (2 * i)));
            double L = j * -1;
            double M = cos(k / 3.0);
            double N = sqrt(3) * sin(k / 3.0);
            double P = (b / (3.0 * a)) * -1;

            double x1 = 2 * j * cos(k / 3.0) - (b / (3.0 * a));
            double x2 = L * (M + N) + P;
            double x3 = L * (M - N) + P;

            add_root<T>(roots_row, x1, n_roots);
            add_root<T>(roots_row, x2, n_roots);
            add_root<T>(roots_row, x3, n_roots);
        } else {
            double R = -(g / 2.0) + sqrt(h);
            double S;
            double U;

            if (R >= 0) {
                S = cbrt(R);
            } else {
                S = -cbrt(-R);
            }
            double V = -(g / 2.0) - sqrt(h);
            if (V >= 0) {
                U = cbrt(V);
            } else {
                U = -cbrt(-V);
            }

            double x1 = (S + U) - (b / (3.0 * a));
            add_root<T>(roots_row, x1, n_roots);
        }
    }
}

template<typename T>
__device__ bool
is_inside(vec3<T> &v0, vec3<T> &v1, vec3<T> &v2, vec3<T> &vt) {
    vec3<T> e1 = v0 - v1;
    vec3<T> e2 = v2 - v1;
    vec3<T> ep = vt - v1;
    T d00 = dot(e1, e1);
    T d01 = dot(e1, e2);
    T d11 = dot(e2, e2);
    T d20 = dot(ep, e1);
    T d21 = dot(ep, e2);

    T denom = d00 * d11 - d01 * d01;
    T v = (d11 * d20 - d01 * d21) / denom;
    T w = (d00 * d21 - d01 * d20) / denom;
    T u = 1 - v - w;

    if (v < 0 or v > 1) return false;
    if (w < 0 or w > 1) return false;
    if (u < 0 or u > 1) return false;

    return true;
}


template<typename T>
__global__
void
filterCollisions(MovingTetrahedron<T> *tetrahedrons, long3 *minimal_collisions_tids,
                 T *cubic_roots, bool *is_valid_collision, int num_collisions) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;


    long3 *tcoll = minimal_collisions_tids + idx;
    MovingTetrahedron<T> *tetrahedron = tetrahedrons + idx;


    T root = cubic_roots[idx];

//    printf("tcoll: {%d, %d, %d}\nroot: %g", (int) tcoll->x, (int) tcoll->y, (int) tcoll->z);
    if (root < -EPSILON) return;
    Tetrahedron<T> thz = Tetrahedron<T>(tetrahedron, root);


//    if (tcoll->x != 0 or tcoll->y != 1 or tcoll->z != 0) return;


    bool intersects = true;
    if (tcoll->z < 9) {
        vec3<T> da = thz.v1 - thz.v0;
        vec3<T> db = thz.v3 - thz.v2;
        vec3<T> dc = thz.v2 - thz.v0;

        T s = dot(cross(dc, db), cross(da, db)) / norm2(cross(da, db));
        T p = dot(cross(dc, da), cross(da, db)) / norm2(cross(da, db));

        if (s < 0 or s > 1 or p < 0 or p > 1)
            intersects = false;
    } else if (tcoll->z < 12) {
        intersects = is_inside<T>(thz.v0, thz.v1, thz.v2, thz.v3);
    } else {
        intersects = is_inside<T>(thz.v1, thz.v2, thz.v3, thz.v0);
    }

    if (intersects) {
//        printf("tcoll: {%d, %d, %d}\n", (int) tcoll->x, (int) tcoll->y, (int) tcoll->z);
        is_valid_collision[idx] = 1;
    }
}

template<typename T>
__global__
void
copy_filtered_collisions(long3 *minimal_collisions_tids, T *cubic_roots, bool *is_valid_collision,
                         long3 *filtered_collisions, T *filtered_roots, int num_collisions, int *counter) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;

    if (not is_valid_collision[idx]) return;

    int to_idx = atomicAdd(counter, 1);
    filtered_collisions[to_idx] = minimal_collisions_tids[idx];
    filtered_roots[to_idx] = cubic_roots[idx];

}

template<typename T>
void
compute_cubic_roots(MovingTetrahedron<T> *tetrahedrons, vec3<T> *roots, long3 *minimal_collisions, int num_collisions) {
    int blockSize = NUM_THREADS;
    int grid_size = (num_collisions + blockSize - 1) / blockSize;

    thrust::device_vector <vec4<T>> cubic_coefs(num_collisions);
    getCubicCoeffs<T><<<grid_size, blockSize>>>(tetrahedrons,
                                                cubic_coefs.data().get(),
                                                minimal_collisions,
                                                num_collisions);
    cudaCheckError();

    solveCubic<T><<<grid_size, blockSize>>>(cubic_coefs.data().get(),
                                            roots,
                                            num_collisions);

    cudaCheckError();
}

template<typename T>
__device__
void flatten_move(MovingTetrahedron<T> *tetrahedrons_from, long3 *collisions_from,
                  MovingTetrahedron<T> *tetrahedrons_to, long3 *collisions_to, T *roots_flat_to, int from_idx,
                  int to_idx, T root) {
    tetrahedrons_to[to_idx] = tetrahedrons_from[from_idx];
    collisions_to[to_idx] = collisions_from[from_idx];
    roots_flat_to[to_idx] = root;
}

template<typename T>
__global__
void
flatten_cubic_roots(MovingTetrahedron<T> *tetrahedrons, long3 *collisions, vec3<T> *root_triplets,
                    MovingTetrahedron<T> *tetrahedrons_valid, long3 *collisions_valid, T *roots_flat, int num_roots,
                    int *counter) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_roots) return;

    vec3<T> *root_triplet = root_triplets + idx;

    if (root_triplet->x >= 0 and root_triplet->x <= 1) {
        int to_idx = atomicAdd(counter, 1);
        flatten_move(tetrahedrons, collisions, tetrahedrons_valid, collisions_valid, roots_flat, idx, to_idx,
                     root_triplet->x);
//        roots_flat[to_idx] = root_triplet->x;
    }

    if (root_triplet->y >= 0 and root_triplet->y <= 1) {
        int to_idx = atomicAdd(counter, 1);
        flatten_move(tetrahedrons, collisions, tetrahedrons_valid, collisions_valid, roots_flat, idx, to_idx,
                     root_triplet->y);
//        roots_flat[to_idx] = root_triplet->y;
    }

    if (root_triplet->z >= 0 and root_triplet->z <= 1) {
        int to_idx = atomicAdd(counter, 1);

        flatten_move(tetrahedrons, collisions, tetrahedrons_valid, collisions_valid, roots_flat, idx, to_idx,
                     root_triplet->z);
//        roots_flat[to_idx] = root_triplet->z;

    }
}

template<typename T>
struct is_valid_root : public thrust::unary_function<T, int> {
public:
    __host__ __device__ int operator()(T root) const {
        return root >= 0 && root <= 1;
    }
};

template<typename T>
struct num_roots : public thrust::unary_function<T, int> {
public:
    __host__ __device__ int operator()(vec3<T> root) const {
        int c = 0;
        if (root.x >= 0 and root.x <= 1) {
            c += 1;
//            c = 1;
        }
        if (root.y >= 0 and root.y <= 1) {
            c += 1;
//            c = 1;
        }
        if (root.z >= 0 and root.z <= 1) {
            c += 1;
//            c = 1;
        }

        return c;
    }
};


struct sum_impulses : public thrust::unary_function<int3, int> {
public:
    __host__ __device__ int operator()(int3 impulse) const {
        int c = impulse.x + impulse.y + impulse.z;
        return c;
    }
};


struct is_true : public thrust::unary_function<bool, int> {
public:
    __host__ __device__ int operator()(bool b) const {
        return b;
    }
};

void find_collisions_from_bbox_tree(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                    at::Tensor triangles_next_tensor,
                                    at::Tensor *collision_tensor_ptr, at::Tensor *time_tensor_ptr,
                                    int max_candidates_per_triangle,
                                    int max_collisions_per_triangle) {
    const auto batch_size = bboxes_tensor.size(0);
    const auto num_triangles = (bboxes_tensor.size(1) + 1) / 2;
    const auto num_nodes = bboxes_tensor.size(1);
    const auto max_total_candidates = max_candidates_per_triangle * num_triangles;

    // list of pairs of triangle indices
    // each pair represents a triangle-triangle penetration
    // max number of collisions is `num_triangles * max_candidates_per_triangle`; if trying to add more collisions -> error
    thrust::device_vector <long2> collisionIndices(max_total_candidates);

    thrust::device_vector<int> triangle_ids(num_triangles);
    thrust::sequence(triangle_ids.begin(), triangle_ids.end());


    // int *counter;
    // number of collisions in each sample of the batch
    thrust::device_vector<int> collision_idx_cnt(batch_size);
    thrust::fill(collision_idx_cnt.begin(), collision_idx_cnt.end(), 0);


    thrust::device_vector<int> minimal_collision_idx_cnt(batch_size);
    thrust::fill(minimal_collision_idx_cnt.begin(), minimal_collision_idx_cnt.end(), 0);

    thrust::device_vector<int> valid_roots_idx_cnt(batch_size);
    thrust::fill(valid_roots_idx_cnt.begin(), valid_roots_idx_cnt.end(), 0);

    thrust::device_vector<int> filtered_minimal_collision_idx_cnt(batch_size);
    thrust::fill(filtered_minimal_collision_idx_cnt.begin(), filtered_minimal_collision_idx_cnt.end(), 0);
    int blockSize = NUM_THREADS;


    AT_DISPATCH_FLOATING_TYPES(
            bboxes_tensor.type(), "bvh_tree_building", ([&] {
                thrust::device_vector <BVHNode<scalar_t>> leaf_nodes(num_triangles);
                thrust::device_vector <BVHNode<scalar_t>> internal_nodes(num_triangles - 1);

                auto bboxes_float_ptr = bboxes_tensor.data<scalar_t>();
                auto tree_long_ptr = tree_tensor.data<long>();
                auto triangle_float_ptr = triangles_tensor.data<scalar_t>();
                auto triangle_next_float_ptr = triangles_next_tensor.data<scalar_t>();

                for (int bidx = 0; bidx < batch_size; ++bidx) {
                    AABB<scalar_t> *bboxes_ptr =
                            (AABB<scalar_t> *) bboxes_float_ptr +
                            num_nodes * bidx;
                    long4 *tree_ptr = (long4 *) tree_long_ptr +
                                      num_nodes * bidx;

                    reconstruct_bvh<scalar_t>(bboxes_ptr, tree_ptr, internal_nodes.data().get(),
                                              leaf_nodes.data().get(), triangle_ids.data().get(), num_nodes,
                                              num_triangles);
                    cudaCheckError();


                    int gridSize = (num_triangles + blockSize - 1) / blockSize;

//                     ================================ findPotentialCollisions
                    thrust::fill(collisionIndices.begin(), collisionIndices.end(),
                                 make_long2(-1, -1));
                    findPotentialCollisions<scalar_t><<<gridSize, blockSize>>>(
                            collisionIndices.data().get(),
                            internal_nodes.data().get(),
                            leaf_nodes.data().get(), triangle_ids.data().get(), num_triangles,
                            max_total_candidates, &collision_idx_cnt.data().get()[bidx]);
                    cudaDeviceSynchronize();
                    if ((int) collision_idx_cnt[bidx] > max_total_candidates) {
                        printf("Number of candidate collisions exceeds maximal allowed number. Some of the candidate collisions are omitted.\n");
                    }
                    int num_cand_collisions =
                            thrust::reduce(thrust::make_transform_iterator(
                                                   collisionIndices.begin(), is_valid_cnt()),
                                           thrust::make_transform_iterator(
                                                   collisionIndices.end(), is_valid_cnt()));

//                    printf("num_cand_collisions: %d\n", num_cand_collisions);
                    if (num_cand_collisions > 0) {

////                         cpt0
//
                        Triangle<scalar_t> *triangles_ptr =
                                (TrianglePtr<scalar_t>) triangle_float_ptr +
                                num_triangles * bidx;
                        Triangle<scalar_t> *triangles_next_ptr =
                                (TrianglePtr<scalar_t>) triangle_next_float_ptr +
                                num_triangles * bidx;


                        // Keep only the pairs of ids where a bounding box to bounding box
                        // collision was detected.
                        thrust::device_vector <long2> triangle_collisions(num_cand_collisions,
                                                                          make_long2(-1, -1));

                        thrust::copy_if(collisionIndices.begin(), collisionIndices.end(),
                                        triangle_collisions.begin(), is_valid_cnt());

                        cudaCheckError();
////                        // cpt1
////
                        int tri_grid_size =
                                (num_triangles + blockSize - 1) / blockSize;

                        thrust::device_vector <MovingTriangle<scalar_t>> moving_triangles(num_triangles);
                        make_moving_triangles<scalar_t><<<tri_grid_size, blockSize>>>(triangles_ptr, triangles_next_ptr,
                                                                                      moving_triangles.data().get(),
                                                                                      num_triangles);
                        cudaCheckError();
//
////                        // cpt2
//                        thrust::device_vector <long3> minimal_collisions(num_cand_collisions * 15,
//                                                 make_long3(-1, -1, -1));
                        thrust::device_vector<bool> minimal_collision_flags(num_cand_collisions * 15, 0);

                        int coll_grid_size = (triangle_collisions.size() + blockSize - 1) / blockSize;
                        dim3 blockGrid(coll_grid_size, 15);
                        checkMinimalCollisions<scalar_t><<<blockGrid, blockSize>>>(
                                triangle_collisions.data().get(),
                                moving_triangles.data().get(), minimal_collision_flags.data().get(),
                                num_cand_collisions);
                        cudaCheckError();
////
                        int num_cminimal_collisions_flags =
                                thrust::reduce(thrust::make_transform_iterator(
                                                       minimal_collision_flags.begin(), is_true()),
                                               thrust::make_transform_iterator(
                                                       minimal_collision_flags.end(), is_true()));
                        thrust::device_vector <long3> minimal_collisions(num_cminimal_collisions_flags,
                                                                         make_long3(-1, -1, -1));

//                        printf("num_cminimal_collisions_flags: %d\n", num_cminimal_collisions_flags);
                        if (num_cminimal_collisions_flags > 0) {


                            copyMinimalCollisions<scalar_t><<<blockGrid, blockSize>>>(
                                    triangle_collisions.data().get(), minimal_collisions.data().get(),
                                    minimal_collision_flags.data().get(),
                                    num_cand_collisions, &minimal_collision_idx_cnt.data().get()[bidx]);
                            cudaCheckError();


//                        // cpt3
                            int num_cminimal_collisions = minimal_collision_idx_cnt[bidx];

                            int min_coll_grid_size = (num_cminimal_collisions + blockSize - 1) / blockSize;
                            thrust::device_vector <MovingTetrahedron<scalar_t>> tetrahedrons(num_cminimal_collisions);
                            make_tetrahedrons<scalar_t><<<min_coll_grid_size, blockSize>>>(
                                    minimal_collisions.data().get(),
                                    moving_triangles.data().get(), tetrahedrons.data().get(), num_cminimal_collisions);
                            cudaCheckError();
                            thrust::device_vector <vec3<scalar_t>> cubic_roots_triplets(num_cminimal_collisions,
                                                                                        {-1, -1, -1});

//                            printf("\nnum_cminimal_collisions: %d\n", num_cminimal_collisions);
                            compute_cubic_roots<scalar_t>(tetrahedrons.data().get(), cubic_roots_triplets.data().get(),
                                                          minimal_collisions.data().get(),
                                                          num_cminimal_collisions);
                            int num_valid_roots =
                                    thrust::reduce(thrust::make_transform_iterator(
                                                           cubic_roots_triplets.begin(), num_roots<scalar_t>()),
                                                   thrust::make_transform_iterator(
                                                           cubic_roots_triplets.end(), num_roots<scalar_t>()));


                            if (num_valid_roots > 0) {
//
//
                                //                            printf("num_valid_roots: %d\n", num_valid_roots);
                                //                            printf("num_cminimal_collisions: %d\n", num_cminimal_collisions);

                                thrust::device_vector <MovingTetrahedron<scalar_t>> tetrahedrons_valid(num_valid_roots);
                                thrust::device_vector <long3> minimal_collisions_valid(num_valid_roots,
                                                                                       make_long3(-1, -1, -1));

                                int valroots_grid_size = (num_valid_roots + blockSize - 1) / blockSize;
                                thrust::device_vector <scalar_t> cubic_roots(num_valid_roots, -1);
                                flatten_cubic_roots<scalar_t><<<min_coll_grid_size, blockSize>>>(
                                        tetrahedrons.data().get(),
                                        minimal_collisions.data().get(),
                                        cubic_roots_triplets.data().get(),
                                        tetrahedrons_valid.data().get(),
                                        minimal_collisions_valid.data().get(),
                                        cubic_roots.data().get(),
                                        num_cminimal_collisions,
                                        &valid_roots_idx_cnt.data().get()[bidx]);

                                int num_valid_roots_counter = valid_roots_idx_cnt[bidx];
//                                printf("num_valid_roots_counter: %d\n", num_valid_roots_counter);

                                thrust::device_vector<bool> is_valid_collision(num_valid_roots, 0);
                                filterCollisions<scalar_t><<<valroots_grid_size, blockSize>>>(
                                        tetrahedrons_valid.data().get(),
                                        minimal_collisions_valid.data().get(),
                                        cubic_roots.data().get(),
                                        is_valid_collision.data().get(),
                                        num_valid_roots);
                                cudaCheckError();

                                int num_valid_collisions =
                                        thrust::reduce(thrust::make_transform_iterator(
                                                               is_valid_collision.begin(), is_true()),
                                                       thrust::make_transform_iterator(
                                                               is_valid_collision.end(), is_true()));

//                                printf("CONT: num_valid_collisions=%d\n", num_valid_collisions);
                                //                            printf("num_valid_collisions: %d\n", num_valid_collisions);
                                if (num_valid_collisions > 0) {
                                    thrust::device_vector <long3> filtered_collisions(num_valid_collisions,
                                                                                      make_long3(-1, -1, -1));
                                    thrust::device_vector <scalar_t> filtered_cubic_roots(num_valid_collisions);
                                    //
                                    copy_filtered_collisions<<<valroots_grid_size, blockSize>>>(
                                            minimal_collisions_valid.data().get(),
                                            cubic_roots.data().get(),
                                            is_valid_collision.data().get(),
                                            filtered_collisions.data().get(),
                                            filtered_cubic_roots.data().get(),
                                            num_valid_roots,
                                            &filtered_minimal_collision_idx_cnt.data().get()[bidx]);
                                    ////                        int num_valid_collisions_counter = minimal_collision_idx_cnt2[bidx];
                                    ////                        int num_filtered_collisions = filtered_minimal_collision_idx_cnt[bidx];
                                    ////                        printf("num_cand_collisions: %d\n", num_cand_collisions);
                                    ////                        printf("num_cminimal_collisions_flags: %d\n", num_cminimal_collisions_flags);
                                    ////                        printf("num_cminimal_collisions: %d\n", num_cminimal_collisions);
                                    ////                        printf("num_valid_roots: %d\n", num_valid_roots);
                                    ////                        printf("num_valid_collisions: %d\n", num_valid_collisions);
                                    //////                        printf("num_valid_collisions_counter: %d\n", num_valid_collisions_counter);
                                    ////                        printf("num_filtered_collisions: %d\n", num_filtered_collisions);
                                    //

                                    long size_to_copy = filtered_collisions.size();
                                    if (size_to_copy > num_triangles * max_collisions_per_triangle) {
                                        printf("Number of actual collisions exceeds maximal allowed number. Some of the collisions are omitted.\n");
                                        size_to_copy = num_triangles * max_collisions_per_triangle;
                                    }

                                    long *dev_ptr = collision_tensor_ptr->data<long>();
                                    cudaMemcpy(dev_ptr + bidx * num_triangles * max_collisions_per_triangle * 3,
                                               (long *) filtered_collisions.data().get(),
                                               3 * size_to_copy * sizeof(long),
                                               cudaMemcpyDeviceToDevice);

                                    scalar_t *time_ptr = time_tensor_ptr->data<scalar_t>();
                                    cudaMemcpy(time_ptr + bidx * num_triangles * max_collisions_per_triangle,
                                               (scalar_t *) filtered_cubic_roots.data().get(),
                                               size_to_copy * sizeof(scalar_t),
                                               cudaMemcpyDeviceToDevice);
                                    cudaCheckError();
                                }
                            }

                        }
                    }


                }

            }));

}


template<typename T>
__global__ void findPotentialCollisionsPartial(long2 *collisionIndices,
                                               BVHNodePtr<T> root,
                                               BVHNodePtr<T> leaves, int *triangle_ids,
                                               bool* triangles_to_check,
                                               int num_primitives,
                                               int max_total_collisions, int *counter, T threshold = 0) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_primitives) {
        BVHNodePtr<T> leaf = leaves + idx;
        int triangle_id = triangle_ids[idx];
        bool to_check = triangles_to_check[idx];

        if (to_check) {
            int num_collisions =
                    traverseBVHPartial<T>(collisionIndices, root, leaf->bbox, triangle_id,
                                   leaf, triangles_to_check, max_total_collisions, counter, threshold, false);
        }
    }
    return;
}



void find_collisions_from_bbox_tree_partial(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                    at::Tensor triangles_next_tensor,
                                    at::Tensor triangles_to_check_tensor,
                                    at::Tensor *collision_tensor_ptr, at::Tensor *time_tensor_ptr,
                                    int max_candidates_per_triangle,
                                    int max_collisions_per_triangle) {
    const auto batch_size = bboxes_tensor.size(0);
    const auto num_triangles = (bboxes_tensor.size(1) + 1) / 2;
    const auto num_nodes = bboxes_tensor.size(1);
    const auto max_total_candidates = max_candidates_per_triangle * num_triangles;

    // list of pairs of triangle indices
    // each pair represents a triangle-triangle penetration
    // max number of collisions is `num_triangles * max_candidates_per_triangle`; if trying to add more collisions -> error
    thrust::device_vector <long2> collisionIndices(max_total_candidates);

    thrust::device_vector<int> triangle_ids(num_triangles);
    thrust::sequence(triangle_ids.begin(), triangle_ids.end());


    // int *counter;
    // number of collisions in each sample of the batch
    thrust::device_vector<int> collision_idx_cnt(batch_size);
    thrust::fill(collision_idx_cnt.begin(), collision_idx_cnt.end(), 0);


    thrust::device_vector<int> minimal_collision_idx_cnt(batch_size);
    thrust::fill(minimal_collision_idx_cnt.begin(), minimal_collision_idx_cnt.end(), 0);

    thrust::device_vector<int> valid_roots_idx_cnt(batch_size);
    thrust::fill(valid_roots_idx_cnt.begin(), valid_roots_idx_cnt.end(), 0);

    thrust::device_vector<int> filtered_minimal_collision_idx_cnt(batch_size);
    thrust::fill(filtered_minimal_collision_idx_cnt.begin(), filtered_minimal_collision_idx_cnt.end(), 0);
    int blockSize = NUM_THREADS;


    AT_DISPATCH_FLOATING_TYPES(
            bboxes_tensor.type(), "bvh_tree_building", ([&] {
                thrust::device_vector <BVHNode<scalar_t>> leaf_nodes(num_triangles);
                thrust::device_vector <BVHNode<scalar_t>> internal_nodes(num_triangles - 1);

                auto bboxes_float_ptr = bboxes_tensor.data<scalar_t>();
                auto tree_long_ptr = tree_tensor.data<long>();
                auto triangle_float_ptr = triangles_tensor.data<scalar_t>();
                auto triangle_next_float_ptr = triangles_next_tensor.data<scalar_t>();
                auto triangles_to_check_bool_ptr =   triangles_to_check_tensor.data<bool>();

                for (int bidx = 0; bidx < batch_size; ++bidx) {
                    AABB<scalar_t> *bboxes_ptr =
                            (AABB<scalar_t> *) bboxes_float_ptr +
                            num_nodes * bidx;
                    long4 *tree_ptr = (long4 *) tree_long_ptr +
                                      num_nodes * bidx;
                    bool* triangles_to_check_ptr = (bool*) triangles_to_check_bool_ptr + num_triangles * bidx;

                    reconstruct_bvh<scalar_t>(bboxes_ptr, tree_ptr, internal_nodes.data().get(),
                                              leaf_nodes.data().get(), triangle_ids.data().get(), num_nodes,
                                              num_triangles);
                    cudaCheckError();


                    int gridSize = (num_triangles + blockSize - 1) / blockSize;

//                     ================================ findPotentialCollisions
                    thrust::fill(collisionIndices.begin(), collisionIndices.end(),
                                 make_long2(-1, -1));
                    findPotentialCollisionsPartial<scalar_t><<<gridSize, blockSize>>>(
                            collisionIndices.data().get(),
                            internal_nodes.data().get(),
                            leaf_nodes.data().get(), triangle_ids.data().get(), triangles_to_check_ptr, num_triangles,
                            max_total_candidates, &collision_idx_cnt.data().get()[bidx]);


//                    findPotentialCollisions<scalar_t><<<gridSize, blockSize>>>(
//                            collisionIndices.data().get(),
//                            internal_nodes.data().get(),
//                            leaf_nodes.data().get(), triangle_ids.data().get(), num_triangles,
//                            max_total_candidates, &collision_idx_cnt.data().get()[bidx]);
                    cudaDeviceSynchronize();

                    if ((int) collision_idx_cnt[bidx] > max_total_candidates) {
                        printf("Number of candidate collisions exceeds maximal allowed number. Some of the candidate collisions are omitted.\n");
                    }
                    int num_cand_collisions =
                            thrust::reduce(thrust::make_transform_iterator(
                                                   collisionIndices.begin(), is_valid_cnt()),
                                           thrust::make_transform_iterator(
                                                   collisionIndices.end(), is_valid_cnt()));

//                    printf("num_cand_collisions: %d\n", num_cand_collisions);
                    if (num_cand_collisions > 0) {

////                         cpt0
//
                        Triangle<scalar_t> *triangles_ptr =
                                (TrianglePtr<scalar_t>) triangle_float_ptr +
                                num_triangles * bidx;
                        Triangle<scalar_t> *triangles_next_ptr =
                                (TrianglePtr<scalar_t>) triangle_next_float_ptr +
                                num_triangles * bidx;


                        // Keep only the pairs of ids where a bounding box to bounding box
                        // collision was detected.
                        thrust::device_vector <long2> triangle_collisions(num_cand_collisions,
                                                                          make_long2(-1, -1));

                        thrust::copy_if(collisionIndices.begin(), collisionIndices.end(),
                                        triangle_collisions.begin(), is_valid_cnt());

                        cudaCheckError();
////                        // cpt1
////
                        int tri_grid_size =
                                (num_triangles + blockSize - 1) / blockSize;

                        thrust::device_vector <MovingTriangle<scalar_t>> moving_triangles(num_triangles);
                        make_moving_triangles<scalar_t><<<tri_grid_size, blockSize>>>(triangles_ptr, triangles_next_ptr,
                                                                                      moving_triangles.data().get(),
                                                                                      num_triangles);
                        cudaCheckError();
//
////                        // cpt2
//                        thrust::device_vector <long3> minimal_collisions(num_cand_collisions * 15,
//                                                 make_long3(-1, -1, -1));
                        thrust::device_vector<bool> minimal_collision_flags(num_cand_collisions * 15, 0);

                        int coll_grid_size = (triangle_collisions.size() + blockSize - 1) / blockSize;
                        dim3 blockGrid(coll_grid_size, 15);
                        checkMinimalCollisions<scalar_t><<<blockGrid, blockSize>>>(
                                triangle_collisions.data().get(),
                                moving_triangles.data().get(), minimal_collision_flags.data().get(),
                                num_cand_collisions);
                        cudaCheckError();
////
                        int num_cminimal_collisions_flags =
                                thrust::reduce(thrust::make_transform_iterator(
                                                       minimal_collision_flags.begin(), is_true()),
                                               thrust::make_transform_iterator(
                                                       minimal_collision_flags.end(), is_true()));
                        thrust::device_vector <long3> minimal_collisions(num_cminimal_collisions_flags,
                                                                         make_long3(-1, -1, -1));

//                        printf("num_cminimal_collisions_flags: %d\n", num_cminimal_collisions_flags);
                        if (num_cminimal_collisions_flags > 0) {


                            copyMinimalCollisions<scalar_t><<<blockGrid, blockSize>>>(
                                    triangle_collisions.data().get(), minimal_collisions.data().get(),
                                    minimal_collision_flags.data().get(),
                                    num_cand_collisions, &minimal_collision_idx_cnt.data().get()[bidx]);
                            cudaCheckError();


//                        // cpt3
                            int num_cminimal_collisions = minimal_collision_idx_cnt[bidx];

                            int min_coll_grid_size = (num_cminimal_collisions + blockSize - 1) / blockSize;
                            thrust::device_vector <MovingTetrahedron<scalar_t>> tetrahedrons(num_cminimal_collisions);
                            make_tetrahedrons<scalar_t><<<min_coll_grid_size, blockSize>>>(
                                    minimal_collisions.data().get(),
                                    moving_triangles.data().get(), tetrahedrons.data().get(), num_cminimal_collisions);
                            cudaCheckError();
                            thrust::device_vector <vec3<scalar_t>> cubic_roots_triplets(num_cminimal_collisions,
                                                                                        {-1, -1, -1});

//                            printf("\nnum_cminimal_collisions: %d\n", num_cminimal_collisions);
                            compute_cubic_roots<scalar_t>(tetrahedrons.data().get(), cubic_roots_triplets.data().get(),
                                                          minimal_collisions.data().get(),
                                                          num_cminimal_collisions);
                            int num_valid_roots =
                                    thrust::reduce(thrust::make_transform_iterator(
                                                           cubic_roots_triplets.begin(), num_roots<scalar_t>()),
                                                   thrust::make_transform_iterator(
                                                           cubic_roots_triplets.end(), num_roots<scalar_t>()));


                            if (num_valid_roots > 0) {
//
//
                                //                            printf("num_valid_roots: %d\n", num_valid_roots);
                                //                            printf("num_cminimal_collisions: %d\n", num_cminimal_collisions);

                                thrust::device_vector <MovingTetrahedron<scalar_t>> tetrahedrons_valid(num_valid_roots);
                                thrust::device_vector <long3> minimal_collisions_valid(num_valid_roots,
                                                                                       make_long3(-1, -1, -1));

                                int valroots_grid_size = (num_valid_roots + blockSize - 1) / blockSize;
                                thrust::device_vector <scalar_t> cubic_roots(num_valid_roots, -1);
                                flatten_cubic_roots<scalar_t><<<min_coll_grid_size, blockSize>>>(
                                        tetrahedrons.data().get(),
                                        minimal_collisions.data().get(),
                                        cubic_roots_triplets.data().get(),
                                        tetrahedrons_valid.data().get(),
                                        minimal_collisions_valid.data().get(),
                                        cubic_roots.data().get(),
                                        num_cminimal_collisions,
                                        &valid_roots_idx_cnt.data().get()[bidx]);

                                int num_valid_roots_counter = valid_roots_idx_cnt[bidx];
//                                printf("num_valid_roots_counter: %d\n", num_valid_roots_counter);

                                thrust::device_vector<bool> is_valid_collision(num_valid_roots, 0);
                                filterCollisions<scalar_t><<<valroots_grid_size, blockSize>>>(
                                        tetrahedrons_valid.data().get(),
                                        minimal_collisions_valid.data().get(),
                                        cubic_roots.data().get(),
                                        is_valid_collision.data().get(),
                                        num_valid_roots);
                                cudaCheckError();

                                int num_valid_collisions =
                                        thrust::reduce(thrust::make_transform_iterator(
                                                               is_valid_collision.begin(), is_true()),
                                                       thrust::make_transform_iterator(
                                                               is_valid_collision.end(), is_true()));

//                                printf("CONT: num_valid_collisions=%d\n", num_valid_collisions);
                                //                            printf("num_valid_collisions: %d\n", num_valid_collisions);
                                if (num_valid_collisions > 0) {
                                    thrust::device_vector <long3> filtered_collisions(num_valid_collisions,
                                                                                      make_long3(-1, -1, -1));
                                    thrust::device_vector <scalar_t> filtered_cubic_roots(num_valid_collisions);
                                    //
                                    copy_filtered_collisions<<<valroots_grid_size, blockSize>>>(
                                            minimal_collisions_valid.data().get(),
                                            cubic_roots.data().get(),
                                            is_valid_collision.data().get(),
                                            filtered_collisions.data().get(),
                                            filtered_cubic_roots.data().get(),
                                            num_valid_roots,
                                            &filtered_minimal_collision_idx_cnt.data().get()[bidx]);
                                    ////                        int num_valid_collisions_counter = minimal_collision_idx_cnt2[bidx];
                                    ////                        int num_filtered_collisions = filtered_minimal_collision_idx_cnt[bidx];
                                    ////                        printf("num_cand_collisions: %d\n", num_cand_collisions);
                                    ////                        printf("num_cminimal_collisions_flags: %d\n", num_cminimal_collisions_flags);
                                    ////                        printf("num_cminimal_collisions: %d\n", num_cminimal_collisions);
                                    ////                        printf("num_valid_roots: %d\n", num_valid_roots);
                                    ////                        printf("num_valid_collisions: %d\n", num_valid_collisions);
                                    //////                        printf("num_valid_collisions_counter: %d\n", num_valid_collisions_counter);
                                    ////                        printf("num_filtered_collisions: %d\n", num_filtered_collisions);
                                    //

                                    long size_to_copy = filtered_collisions.size();
                                    if (size_to_copy > num_triangles * max_collisions_per_triangle) {
                                        printf("Number of actual collisions exceeds maximal allowed number. Some of the collisions are omitted.\n");
                                        size_to_copy = num_triangles * max_collisions_per_triangle;
                                    }

                                    long *dev_ptr = collision_tensor_ptr->data<long>();
                                    cudaMemcpy(dev_ptr + bidx * num_triangles * max_collisions_per_triangle * 3,
                                               (long *) filtered_collisions.data().get(),
                                               3 * size_to_copy * sizeof(long),
                                               cudaMemcpyDeviceToDevice);

                                    scalar_t *time_ptr = time_tensor_ptr->data<scalar_t>();
                                    cudaMemcpy(time_ptr + bidx * num_triangles * max_collisions_per_triangle,
                                               (scalar_t *) filtered_cubic_roots.data().get(),
                                               size_to_copy * sizeof(scalar_t),
                                               cudaMemcpyDeviceToDevice);
                                    cudaCheckError();
                                }
                            }

                        }
                    }


                }

            }));

}


void
find_minimal_candidates_from_bbox_tree(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                       at::Tensor triangles_next_tensor,
                                       at::Tensor *collision_tensor_ptr,
                                       int max_collisions) {

    const auto batch_size = bboxes_tensor.size(0);
    const auto num_triangles = (bboxes_tensor.size(1) + 1) / 2;
    const auto num_nodes = bboxes_tensor.size(1);
    const auto max_total_candidates = max_collisions * num_triangles;

    // list of pairs of triangle indices
    // each pair represents a triangle-triangle penetration
    // max number of collisions is `num_triangles * max_collisions`; if trying to add more collisions -> error
    thrust::device_vector <long2> collisionIndices(num_triangles * max_collisions);

    thrust::device_vector<int> triangle_ids(num_triangles);
    thrust::sequence(triangle_ids.begin(), triangle_ids.end());


    // int *counter;
    // number of collisions in each sample of the batch
    thrust::device_vector<int> collision_idx_cnt(batch_size);
    thrust::fill(collision_idx_cnt.begin(), collision_idx_cnt.end(), 0);


    thrust::device_vector<int> minimal_collision_idx_cnt(batch_size);
    thrust::fill(minimal_collision_idx_cnt.begin(), minimal_collision_idx_cnt.end(), 0);

    thrust::device_vector<int> filtered_minimal_collision_idx_cnt(batch_size);
    thrust::fill(filtered_minimal_collision_idx_cnt.begin(), filtered_minimal_collision_idx_cnt.end(), 0);

    AT_DISPATCH_FLOATING_TYPES(
            bboxes_tensor.type(), "bvh_tree_building", ([&] {
                thrust::device_vector <BVHNode<scalar_t>> leaf_nodes(num_triangles);
                thrust::device_vector <BVHNode<scalar_t>> internal_nodes(num_triangles - 1);

                auto bboxes_float_ptr = bboxes_tensor.data<scalar_t>();
                auto tree_long_ptr = tree_tensor.data<long>();
                auto triangle_float_ptr = triangles_tensor.data<scalar_t>();
                auto triangle_next_float_ptr = triangles_next_tensor.data<scalar_t>();

                for (int bidx = 0; bidx < batch_size; ++bidx) {
                    AABB<scalar_t> *bboxes_ptr =
                            (AABB<scalar_t> *) bboxes_float_ptr +
                            num_nodes * bidx;

                    long4 *tree_ptr = (long4 *) tree_long_ptr +
                                      num_nodes * bidx;

                    thrust::fill(collisionIndices.begin(), collisionIndices.end(),
                                 make_long2(-1, -1));

                    reconstruct_bvh<scalar_t>(bboxes_ptr, tree_ptr, internal_nodes.data().get(),
                                              leaf_nodes.data().get(), triangle_ids.data().get(), num_nodes,
                                              num_triangles);
                    cudaCheckError();

                    int blockSize = NUM_THREADS;
                    int gridSize = (num_triangles + blockSize - 1) / blockSize;

//                     ================================ findPotentialCollisions
                    findPotentialCollisions<scalar_t><<<gridSize, blockSize>>>(
                            collisionIndices.data().get(),
                            internal_nodes.data().get(),
                            leaf_nodes.data().get(), triangle_ids.data().get(), num_triangles,
                            max_total_candidates, &collision_idx_cnt.data().get()[bidx]);
                    if ((int) collision_idx_cnt[bidx] > max_total_candidates) {
                        printf("Number of candidate collisions exceeds maximal allowed number. Some of the candidate collisions are omitted.\n");
                    }
                    cudaDeviceSynchronize();

                    int num_cand_collisions =
                            thrust::reduce(thrust::make_transform_iterator(
                                                   collisionIndices.begin(), is_valid_cnt()),
                                           thrust::make_transform_iterator(
                                                   collisionIndices.end(), is_valid_cnt()));


                    if (num_cand_collisions > 0) {

                        Triangle<scalar_t> *triangles_ptr =
                                (TrianglePtr<scalar_t>) triangle_float_ptr +
                                num_triangles * bidx;

                        Triangle<scalar_t> *triangles_next_ptr =
                                (TrianglePtr<scalar_t>) triangle_next_float_ptr +
                                num_triangles * bidx;


                        // Keep only the pairs of ids where a bounding box to bounding box
                        // collision was detected.
                        thrust::device_vector <long2> triangle_collisions(num_cand_collisions,
                                                                          make_long2(-1, -1));
                        thrust::device_vector <long3> minimal_collisions(num_cand_collisions * 15,
                                                                         make_long3(-1, -1, -1));


                        thrust::copy_if(collisionIndices.begin(), collisionIndices.end(),
                                        triangle_collisions.begin(), is_valid_cnt());

                        cudaCheckError();

                        int tri_grid_size =
                                (num_triangles + blockSize - 1) / blockSize;


                        thrust::device_vector <MovingTriangle<scalar_t>> moving_triangles(num_triangles);
                        make_moving_triangles<scalar_t><<<tri_grid_size, blockSize>>>(triangles_ptr, triangles_next_ptr,
                                                                                      moving_triangles.data().get(),
                                                                                      num_triangles);
                        cudaCheckError();


                        int coll_grid_size = (triangle_collisions.size() + blockSize - 1) / blockSize;
                        dim3 blockGrid(coll_grid_size, 15);
                        findMinimalCollisions<scalar_t><<<blockGrid, blockSize>>>(
                                triangle_collisions.data().get(),
                                moving_triangles.data().get(), minimal_collisions.data().get(),
                                num_cand_collisions, &minimal_collision_idx_cnt.data().get()[bidx]);
                        cudaCheckError();


//                        printf("num_cand_collisions:\t%d\n", num_cand_collisions);
//                        printf("num_minimal_collisions:\t%d\n", num_cminimal_collisions);

                        long *dev_ptr = collision_tensor_ptr->data<long>();
                        cudaMemcpy(dev_ptr + bidx * num_triangles * max_collisions * 3 * 15,
                                   (long *) minimal_collisions.data().get(),
                                   3 * minimal_collisions.size() * sizeof(long),
                                   cudaMemcpyDeviceToDevice);


                    }


                }

            }));

}


