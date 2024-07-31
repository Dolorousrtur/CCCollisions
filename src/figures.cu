/*
   Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
   holder of all proprietary rights on this computer program.
   You can only use this computer program if you have closed
   a license agreement with MPG or you get the right to use the computer
   program from someone who is authorized to grant you that right.
   Any use of the computer program without a valid license is prohibited and
   liable to prosecution.

   Copyright©2019 Max-Planck-Gesellschaft zur Förderung
   der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
   for Intelligent Systems and the Max Planck Institute for Biological
   Cybernetics. All rights reserved.

   Contact: ps-license@tuebingen.mpg.de
*/


#pragma once

#include "flags.h"
#include "aabb.cu"

template<typename T>
struct Triangle {
public:
    vec3<T> v0;
    vec3<T> v1;
    vec3<T> v2;

    __host__ __device__ Triangle() {};

    __host__ __device__ Triangle(const vec3<T> &vertex0, const vec3<T> &vertex1,
                                 const vec3<T> &vertex2)
            : v0(vertex0), v1(vertex1), v2(vertex2) {};

    __host__ __device__ AABB<T> ComputeBBox() {
        return AABB<T>(min(v0.x, min(v1.x, v2.x)), min(v0.y, min(v1.y, v2.y)),
                       min(v0.z, min(v1.z, v2.z)), max(v0.x, max(v1.x, v2.x)),
                       max(v0.y, max(v1.y, v2.y)), max(v0.z, max(v1.z, v2.z)));
    }

    __host__ __device__ vec3<T> &getVertexByInd(int i) {
        switch (i) {
            case 0:
                return v0;
                break;
            case 1:
                return v1;
                break;
            case 2:
                return v2;
                break;
        }

    }

    __host__ __device__ AABB<T> ComputeBBoxVertex(int i) {
        vec3<T> v = getVertexByInd(i);
        return AABB<T>(v.x, v.y, v.z, v.x, v.y, v.z);
    }

    __host__ __device__ AABB<T> ComputeBBoxEdge(int i, int j) {
        AABB<T> b_i = ComputeBBoxVertex(i);
        AABB<T> b_j = ComputeBBoxVertex(j);
        return b_i + b_j;
    }


};

template<typename T> using TrianglePtr = Triangle<T> *;

template<typename T>
struct MovingTetrahedron {
public:
    vec3<T> v0;
    vec3<T> v1;
    vec3<T> v2;
    vec3<T> v3;
    vec3<T> dv0;
    vec3<T> dv1;
    vec3<T> dv2;
    vec3<T> dv3;

    __host__ __device__ MovingTetrahedron() {};

    __host__ __device__ MovingTetrahedron(const vec3<T> &vertex0, const vec3<T> &vertex1,
                                          const vec3<T> &vertex2, const vec3<T> &vertex3,
                                          const vec3<T> &delta_vertex0, const vec3<T> &delta_vertex1,
                                          const vec3<T> &delta_vertex2, const vec3<T> &delta_vertex3)
            : v0(vertex0), v1(vertex1), v2(vertex2), v3(vertex3), dv0(delta_vertex0), dv1(delta_vertex1),
              dv2(delta_vertex2), dv3(delta_vertex3) {};


};

template<typename T>
struct Tetrahedron {
public:
    vec3<T> v0;
    vec3<T> v1;
    vec3<T> v2;
    vec3<T> v3;

    __host__ __device__ Tetrahedron() {};

    __host__ __device__ Tetrahedron(const vec3<T> &vertex0, const vec3<T> &vertex1,
                                    const vec3<T> &vertex2, const vec3<T> &vertex3)
            : v0(vertex0), v1(vertex1), v2(vertex2), v3(vertex3) {};

    __host__ __device__ Tetrahedron(const MovingTetrahedron<T> *moving_tetrahedron, T time) {
        v0 = moving_tetrahedron->v0 + moving_tetrahedron->dv0 * time;
        v1 = moving_tetrahedron->v1 + moving_tetrahedron->dv1 * time;
        v2 = moving_tetrahedron->v2 + moving_tetrahedron->dv2 * time;
        v3 = moving_tetrahedron->v3 + moving_tetrahedron->dv3 * time;
    };


};

template<typename T>
struct MovingTetraMatrix {
public:
    vec3<T> v0;
    vec3<T> v1;
    vec3<T> v2;
    vec3<T> dv0;
    vec3<T> dv1;
    vec3<T> dv2;

    __host__ __device__ MovingTetraMatrix() {};

    __host__ __device__ MovingTetraMatrix(const MovingTetrahedron<T> *tertahedron) {
        v0 = tertahedron->v0 - tertahedron->v3;
        v1 = tertahedron->v1 - tertahedron->v3;
        v2 = tertahedron->v2 - tertahedron->v3;
        dv0 = tertahedron->dv0 - tertahedron->dv3;
        dv1 = tertahedron->dv1 - tertahedron->dv3;
        dv2 = tertahedron->dv2 - tertahedron->dv3;

        v0 = v0 * SCALE;
        v1 = v1 * SCALE;
        v2 = v2 * SCALE;
        dv0 = dv0 * SCALE;
        dv1 = dv1 * SCALE;
        dv2 = dv2 * SCALE;
    };

    __host__ __device__ T coef3() {
        T coef = 0;
        coef += dv2.z * dv1.y * dv0.x;
        coef += -dv1.z * dv2.y * dv0.x;

        coef += -dv2.z * dv0.y * dv1.x;
        coef += dv0.z * dv2.y * dv1.x;
        coef += dv1.z * dv0.y * dv2.x;
        coef += -dv0.z * dv1.y * dv2.x;
        return coef;
    };
//
//    __host__ __device__ T coef3() {
//        T coef = 0;
//
//        T a1 = dv2.z * dv1.y * dv0.x;
//        coef += a1;
//
//        printf("dv2.z: %.20g\tdv1.y: %.20g\tdv0.x: %.20g\n", dv2.z, dv1.y, dv0.x);
//        printf("a1: %.20g\n", a1);
//        printf("coef: %.20g\n", coef);
//
//        T a2 = -dv1.z * dv2.y * dv0.x;
//        coef += a2;
////        printf("a2: %.20g\n", a2);
////        printf("coef: %.20g\n", coef);
//
//        T a3 = -dv2.z * dv0.y * dv1.x;
//        coef += a3;
////        printf("a3: %.20g\n", a3);
////        printf("coef: %.20g\n", coef);
//
//        T a4 = dv0.z * dv2.y * dv1.x;
//        coef += a4;
////        printf("a4: %.20g\n", a4);
////        printf("coef: %.20g\n", coef);
//
//        T a5 = dv1.z * dv0.y * dv2.x;
//        coef += a5;
////        printf("a5: %.20g\n", a5);
////        printf("coef: %.20g\n", coef);
//
//        T a6 = -dv0.z * dv1.y * dv2.x;
//        coef += a6;
////        printf("a6: %.20g\n", a6);
////        printf("coef: %.20g\n", coef);
//
//
////        coef += dv2.z * dv1.y * dv0.x;
////        coef += -dv1.z * dv2.y * dv0.x;
//
////        coef += -dv2.z * dv0.y * dv1.x;
////        coef += dv0.z * dv2.y * dv1.x;
////        coef += dv1.z * dv0.y * dv2.x;
////        coef += -dv0.z * dv1.y * dv2.x;
//        return coef;
//    };

//    __host__ __device__ T coef3() {
//        double coef = 0;
//        coef += dv2.z * dv1.y * dv0.x;
//        printf("coef0: %g\n", coef);
//        coef += -dv1.z * dv2.y * dv0.x;
//        printf("coef1: %g\n", coef);
//        coef += -dv2.z * dv0.y * dv1.x;
//        printf("coef2: %g\n", coef);
//        coef += dv0.z * dv2.y * dv1.x;
//        printf("coef3: %g\n", coef);
//        coef += dv1.z * dv0.y * dv2.x;
//        printf("coef4: %g\n", coef);
//        coef += -(double)dv0.z * (double)dv1.y * (double)dv2.x;
//        printf("dv0.z: %g dv1.y: %g dv2.x: %g\n", dv0.z, dv1.y, dv2.x);
//        printf("coef5: %g\n", coef);
//        return coef;
//    };

    __host__ __device__ T coef2() {
        T coef = 0;
        coef += v2.z * dv1.y * dv0.x;
        coef += dv2.z * v1.y * dv0.x;
        coef += dv2.z * dv1.y * v0.x;
        coef += -v1.z * dv2.y * dv0.x;
        coef += -dv1.z * v2.y * dv0.x;
        coef += -dv1.z * dv2.y * v0.x;
        coef += -v2.z * dv0.y * dv1.x;
        coef += -dv2.z * v0.y * dv1.x;
        coef += -dv2.z * dv0.y * v1.x;
        coef += v0.z * dv2.y * dv1.x;
        coef += dv0.z * v2.y * dv1.x;
        coef += dv0.z * dv2.y * v1.x;
        coef += v1.z * dv0.y * dv2.x;
        coef += dv1.z * v0.y * dv2.x;
        coef += dv1.z * dv0.y * v2.x;
        coef += -v0.z * dv1.y * dv2.x;
        coef += -dv0.z * v1.y * dv2.x;
        coef += -dv0.z * dv1.y * v2.x;
        return coef;
    };

    __host__ __device__ T coef1() {
        T coef = 0;
        coef += v2.z * v1.y * dv0.x;
        coef += v2.z * dv1.y * v0.x;
        coef += dv2.z * v1.y * v0.x;
        coef += -v1.z * v2.y * dv0.x;
        coef += -v1.z * dv2.y * v0.x;
        coef += -dv1.z * v2.y * v0.x;
        coef += -v2.z * v0.y * dv1.x;
        coef += -v2.z * dv0.y * v1.x;
        coef += -dv2.z * v0.y * v1.x;
        coef += v0.z * v2.y * dv1.x;
        coef += v0.z * dv2.y * v1.x;
        coef += dv0.z * v2.y * v1.x;
        coef += v1.z * v0.y * dv2.x;
        coef += v1.z * dv0.y * v2.x;
        coef += dv1.z * v0.y * v2.x;
        coef += -v0.z * v1.y * dv2.x;
        coef += -v0.z * dv1.y * v2.x;
        coef += -dv0.z * v1.y * v2.x;
        return coef;
    };

    __host__ __device__ T coef0() {
        T coef = 0;
        coef += v2.z * v1.y * v0.x;
        coef += -v1.z * v2.y * v0.x;
        coef += -v2.z * v0.y * v1.x;
        coef += v0.z * v2.y * v1.x;
        coef += v1.z * v0.y * v2.x;
        coef += -v0.z * v1.y * v2.x;
        return coef;
    };

    __host__ __device__ vec4<T> getCubicCoeffs() {
        T c3 = coef3();
        T c2 = coef2();
        T c1 = coef1();
        T c0 = coef0();


        return {c3, c2, c1, c0};
    };

};

template<typename T>
struct MovingTriangle {
public:
    vec3<T> v0;
    vec3<T> v1;
    vec3<T> v2;
    vec3<T> dv0;
    vec3<T> dv1;
    vec3<T> dv2;

    __host__ __device__ MovingTriangle() {};

    __host__ __device__ MovingTriangle(const Triangle<T> &triangle0, const Triangle<T> &triangle1) {
        v0 = triangle0.v0;
        v1 = triangle0.v1;
        v2 = triangle0.v2;
        dv0 = triangle1.v0 - triangle0.v0;
        dv1 = triangle1.v1 - triangle0.v1;
        dv2 = triangle1.v2 - triangle0.v2;
    };

    __host__ __device__ MovingTriangle(const vec3<T> &vertex0, const vec3<T> &vertex1,
                                       const vec3<T> &vertex2, const vec3<T> &delta_vertex0,
                                       const vec3<T> &delta_vertex1, const vec3<T> &delta_vertex2)
            : v0(vertex0), v1(vertex1), v2(vertex2), dv0(delta_vertex0), dv1(delta_vertex1), dv2(delta_vertex2) {};


    __host__ __device__ vec3<T> &getVertexByInd(int i, bool is_delta) {
        switch (i) {
            case 0:
                if (is_delta)
                    return dv0;
                else
                    return v0;
                break;
            case 1:
                if (is_delta)
                    return dv1;
                else
                    return v1;
                break;
            case 2:
                if (is_delta)
                    return dv2;
                else
                    return v2;
                break;
        }

    }

    __host__ __device__ AABB<T> ComputeBBoxVertex(int i) {
        vec3<T> v = getVertexByInd(i, false);
        vec3<T> dv = getVertexByInd(i, true);
        vec3<T> v2 = v + dv;
        return AABB<T>(min(v.x, v2.x), min(v.y, v2.y), min(v.z, v2.z), max(v.x, v2.x), max(v.y, v2.y), max(v.z, v2.z));
    }

    __host__ __device__ AABB<T> ComputeBBoxEdge(int i, int j) {
        AABB<T> b_i = ComputeBBoxVertex(i);
        AABB<T> b_j = ComputeBBoxVertex(j);
        return b_i + b_j;
    }

    __host__ __device__ AABB<T> ComputeBBox() {
        AABB<T> b_0 = ComputeBBoxVertex(0);
        AABB<T> b_1 = ComputeBBoxVertex(1);
        AABB<T> b_2 = ComputeBBoxVertex(2);
        return b_0 + b_1 + b_2;
    }

};


template<typename T>
std::ostream &operator<<(std::ostream &os, const Triangle<T> &x) {
    os << x.v0 << std::endl;
    os << x.v1 << std::endl;
    os << x.v2 << std::endl;
    return os;
}


template<typename T>
__device__ inline vec2<T> isect_interval(const vec3<T> &sep_axis,
                                         const Triangle<T> &tri) {
    // Check the separating sep_axis versus the first point of the triangle
    T proj_distance = dot(sep_axis, tri.v0);

    vec2<T> interval;
    interval.x = proj_distance;
    interval.y = proj_distance;

    proj_distance = dot(sep_axis, tri.v1);
    interval.x = min(interval.x, proj_distance);
    interval.y = max(interval.y, proj_distance);

    proj_distance = dot(sep_axis, tri.v2);
    interval.x = min(interval.x, proj_distance);
    interval.y = max(interval.y, proj_distance);

    return interval;
}


template<typename T>
__device__ inline vec2<T> point_interval(const vec3<T> &sep_axis,
                                         const T &point) {
    // Check the separating sep_axis versus the first point of the triangle
    T proj_distance = dot(sep_axis, point);

    vec2<T> interval;
    interval.x = proj_distance;
    interval.y = proj_distance;

    return interval;
}

template<typename T>
__device__ inline bool TriangleTriangleOverlap(const Triangle<T> &tri1,
                                               const Triangle<T> &tri2,
                                               const vec3<T> &sep_axis,
                                               T threshold) {
    // Calculate the projected segment of each triangle on the separating
    // axis.
    vec2<T> tri1_interval = isect_interval(sep_axis, tri1);
    vec2<T> tri2_interval = isect_interval(sep_axis, tri2);

    // In order for the triangles to overlap then there must exist an
    // intersection of the two intervals
    return (tri1_interval.x <= tri2_interval.y + threshold) &&
           (tri1_interval.y >= tri2_interval.x - threshold);
}

template<typename T>
__device__ inline bool TrianglePointOverlap(const Triangle<T> &tri1,
                                            const T &point,
                                            const vec3<T> &sep_axis,
                                            T threshold) {
    // Calculate the projected segment of each triangle on the separating
    // axis.
    vec2<T> tri1_interval = isect_interval(sep_axis, tri1);
    vec2<T> tri2_interval = point_interval(sep_axis, point);

    // In order for the triangles to overlap then there must exist an
    // intersection of the two intervals
    return (tri1_interval.x <= tri2_interval.y + threshold) &&
           (tri1_interval.y >= tri2_interval.x - threshold);
}


template<typename T>
__device__ bool TriangleTriangleIsectSepAxis(const Triangle<T> &tri1,
                                             const Triangle<T> &tri2,
                                             T threshold) {
    // Calculate the edges and the normal for the first triangle
    vec3<T> tri1_edge0 = tri1.v1 - tri1.v0;
    vec3<T> tri1_edge1 = tri1.v2 - tri1.v0;
    vec3<T> tri1_edge2 = tri1.v2 - tri1.v1;
    vec3<T> tri1_normal = cross(tri1_edge1, tri1_edge2);

    // Calculate the edges and the normal for the second triangle
    vec3<T> tri2_edge0 = tri2.v1 - tri2.v0;
    vec3<T> tri2_edge1 = tri2.v2 - tri2.v0;
    vec3<T> tri2_edge2 = tri2.v2 - tri2.v1;
    vec3<T> tri2_normal = cross(tri2_edge1, tri2_edge2);

//    printf("tri1_edge0 : (%g, %g, %g)\n", tri1_edge0.x, tri1_edge0.y, tri1_edge0.z);
//    printf("tri1_edge1 : (%g, %g, %g)\n", tri1_edge1.x, tri1_edge1.y, tri1_edge1.z);
//    printf("tri1_edge2 : (%g, %g, %g)\n", tri1_edge2.x, tri1_edge2.y, tri1_edge2.z);
//    printf("tri1_normal : (%g, %g, %g)\n", tri1_normal.x, tri1_normal.y, tri1_normal.z);
//
//    printf("\n\n");
//    printf("tri2_edge0 : (%g, %g, %g)\n", tri2_edge0.x, tri2_edge0.y, tri2_edge0.z);
//    printf("tri2_edge1 : (%g, %g, %g)\n", tri2_edge1.x, tri2_edge1.y, tri2_edge1.z);
//    printf("tri2_edge2 : (%g, %g, %g)\n", tri2_edge2.x, tri2_edge2.y, tri2_edge2.z);
//    printf("tri2_normal : (%g, %g, %g)\n", tri2_normal.x, tri2_normal.y, tri2_normal.z);

//    printf("\n\n");

    // If the triangles are coplanar then the first 11 cases are all the same,
    // since the cross product will just give us the normal vector
    vec3<T> axes[17] = {
            tri1_normal,
            tri2_normal,
            cross(tri1_edge0, tri2_edge0),
            cross(tri1_edge0, tri2_edge1),
            cross(tri1_edge0, tri2_edge2),
            cross(tri1_edge1, tri2_edge0),
            cross(tri1_edge1, tri2_edge1),
            cross(tri1_edge1, tri2_edge2),
            cross(tri1_edge2, tri2_edge0),
            cross(tri1_edge2, tri2_edge1),
            cross(tri1_edge2, tri2_edge2),
            // Triangles are coplanar
            // Check the axis created by the normal of the triangle and the edges of
            // both triangles.
            cross(tri1_normal, tri1_edge0),
            cross(tri1_normal, tri1_edge1),
            cross(tri1_normal, tri1_edge2),
            cross(tri1_normal, tri2_edge0),
            cross(tri1_normal, tri2_edge1),
            cross(tri1_normal, tri2_edge2),
    };

    bool isect_flag = true;
#pragma unroll
    for (int i = 0; i < 17; ++i) {
        isect_flag = isect_flag && (TriangleTriangleOverlap(tri1, tri2, axes[i], threshold));
//        printf("axes[%d] : (%g, %g, %g)\n"
//               "isect_flag: %d\n", i, axes[i].x, axes[i].y, axes[i].z, isect_flag);
    }

    return isect_flag;
}

template<typename T>
__device__ bool TrianglePointIsectSepAxis(const Triangle<T> &tri1,
                                          const T &point,
                                          T threshold) {
    // Calculate the edges and the normal for the first triangle
    vec3<T> tri1_edge0 = tri1.v1 - tri1.v0;
    vec3<T> tri1_edge1 = tri1.v2 - tri1.v0;
    vec3<T> tri1_edge2 = tri1.v2 - tri1.v1;
    vec3<T> tri1_normal = cross(tri1_edge1, tri1_edge2);

    bool isect_flag = TrianglePointOverlap(tri1, point, tri1_normal, threshold);

    return isect_flag;
}

// Returns true if the triangles share one or multiple vertices
template<typename T>
__device__
#if FORCE_INLINE == 1
__forceinline__
#endif
bool
shareVertex(const MovingTriangle<T> &tri1, const MovingTriangle<T> &tri2) {

    return (tri1.v0.x == tri2.v0.x && tri1.v0.y == tri2.v0.y && tri1.v0.z == tri2.v0.z) ||
           (tri1.v0.x == tri2.v1.x && tri1.v0.y == tri2.v1.y && tri1.v0.z == tri2.v1.z) ||
           (tri1.v0.x == tri2.v2.x && tri1.v0.y == tri2.v2.y && tri1.v0.z == tri2.v2.z) ||
           (tri1.v1.x == tri2.v0.x && tri1.v1.y == tri2.v0.y && tri1.v1.z == tri2.v0.z) ||
           (tri1.v1.x == tri2.v1.x && tri1.v1.y == tri2.v1.y && tri1.v1.z == tri2.v1.z) ||
           (tri1.v1.x == tri2.v2.x && tri1.v1.y == tri2.v2.y && tri1.v1.z == tri2.v2.z) ||
           (tri1.v2.x == tri2.v0.x && tri1.v2.y == tri2.v0.y && tri1.v2.z == tri2.v0.z) ||
           (tri1.v2.x == tri2.v1.x && tri1.v2.y == tri2.v1.y && tri1.v2.z == tri2.v1.z) ||
           (tri1.v2.x == tri2.v2.x && tri1.v2.y == tri2.v2.y && tri1.v2.z == tri2.v2.z);
}


// Returns true if the triangles share one or multiple vertices
template<typename T>
__device__
#if FORCE_INLINE == 1
__forceinline__
#endif
int
countSharedVerts(const Triangle<T> &tri1, const Triangle<T> &tri2) {
    int count = 0;

    if ((tri1.v0.x == tri2.v0.x && tri1.v0.y == tri2.v0.y && tri1.v0.z == tri2.v0.z) ||
           (tri1.v0.x == tri2.v1.x && tri1.v0.y == tri2.v1.y && tri1.v0.z == tri2.v1.z) ||
           (tri1.v0.x == tri2.v2.x && tri1.v0.y == tri2.v2.y && tri1.v0.z == tri2.v2.z)) {
        count++;
    }

    if ((tri1.v1.x == tri2.v0.x && tri1.v1.y == tri2.v0.y && tri1.v1.z == tri2.v0.z) ||
           (tri1.v1.x == tri2.v1.x && tri1.v1.y == tri2.v1.y && tri1.v1.z == tri2.v1.z) ||
           (tri1.v1.x == tri2.v2.x && tri1.v1.y == tri2.v2.y && tri1.v1.z == tri2.v2.z)) {
        count++;
    }

    if ((tri1.v2.x == tri2.v0.x && tri1.v2.y == tri2.v0.y && tri1.v2.z == tri2.v0.z) ||
           (tri1.v2.x == tri2.v1.x && tri1.v2.y == tri2.v1.y && tri1.v2.z == tri2.v1.z) ||
           (tri1.v2.x == tri2.v2.x && tri1.v2.y == tri2.v2.y && tri1.v2.z == tri2.v2.z)) {
        count++;
    }

    return count;
}

// Returns true if the triangles share one or multiple vertices
template<typename T>
__device__
#if FORCE_INLINE == 1
__forceinline__
#endif
bool
shareVertex(const Triangle<T> &tri1, const Triangle<T> &tri2) {

    return (tri1.v0.x == tri2.v0.x && tri1.v0.y == tri2.v0.y && tri1.v0.z == tri2.v0.z) ||
           (tri1.v0.x == tri2.v1.x && tri1.v0.y == tri2.v1.y && tri1.v0.z == tri2.v1.z) ||
           (tri1.v0.x == tri2.v2.x && tri1.v0.y == tri2.v2.y && tri1.v0.z == tri2.v2.z) ||
           (tri1.v1.x == tri2.v0.x && tri1.v1.y == tri2.v0.y && tri1.v1.z == tri2.v0.z) ||
           (tri1.v1.x == tri2.v1.x && tri1.v1.y == tri2.v1.y && tri1.v1.z == tri2.v1.z) ||
           (tri1.v1.x == tri2.v2.x && tri1.v1.y == tri2.v2.y && tri1.v1.z == tri2.v2.z) ||
           (tri1.v2.x == tri2.v0.x && tri1.v2.y == tri2.v0.y && tri1.v2.z == tri2.v0.z) ||
           (tri1.v2.x == tri2.v1.x && tri1.v2.y == tri2.v1.y && tri1.v2.z == tri2.v1.z) ||
           (tri1.v2.x == tri2.v2.x && tri1.v2.y == tri2.v2.y && tri1.v2.z == tri2.v2.z);
}


template<typename T>
__device__
#if FORCE_INLINE == 1
__forceinline__
#endif
bool
hasVertex(const Triangle<T> &tri1, const vec3<T> &vt) {

    return (tri1.v0.x == vt.x && tri1.v0.y == vt.y && tri1.v0.z == vt.z) ||
           (tri1.v1.x == vt.x && tri1.v1.y == vt.y && tri1.v1.z == vt.z) ||
           (tri1.v2.x == vt.x && tri1.v2.y == vt.y && tri1.v2.z == vt.z);
}