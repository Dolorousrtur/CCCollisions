#pragma once

#include "aabb.cu"
#include "utils.cu"
#include "bvh.cu"
#include "build_bvh.cu"
#include "figures.cu"
#include "collisions_static.cu"
#include "collisions_continuous.cu"


#ifndef THICKNESS
#define THICKNESS 2e-3
#endif


template<typename T>
__device__ bool
checkNanVec(vec3<T> &v0) {
    return isnan(v0.x) or isnan(v0.y) or isnan(v0.z);
}


template<typename T>
__device__ void
masses2quad(long3 *tcoll, vec3<T> *triangle_masses, vec4<T> *mass_quads) {
    vec3<T> *mass_tri0 = triangle_masses + tcoll->x;
    vec3<T> *mass_tri1 = triangle_masses + tcoll->y;

    switch (tcoll->z) {
        case 0:
            mass_quads->x = mass_tri0->x;
            mass_quads->y = mass_tri0->y;
            mass_quads->z = mass_tri1->x;
            mass_quads->w = mass_tri1->y;
            break;
        case 1:
            mass_quads->x = mass_tri0->x;
            mass_quads->y = mass_tri0->y;
            mass_quads->z = mass_tri1->x;
            mass_quads->w = mass_tri1->z;
            break;
        case 2:
            mass_quads->x = mass_tri0->x;
            mass_quads->y = mass_tri0->y;
            mass_quads->z = mass_tri1->y;
            mass_quads->w = mass_tri1->z;
            break;
        case 3:
            mass_quads->x = mass_tri0->x;
            mass_quads->y = mass_tri0->z;
            mass_quads->z = mass_tri1->x;
            mass_quads->w = mass_tri1->y;
            break;
        case 4:
            mass_quads->x = mass_tri0->x;
            mass_quads->y = mass_tri0->z;
            mass_quads->z = mass_tri1->x;
            mass_quads->w = mass_tri1->z;
            break;
        case 5:
            mass_quads->x = mass_tri0->x;
            mass_quads->y = mass_tri0->z;
            mass_quads->z = mass_tri1->y;
            mass_quads->w = mass_tri1->z;
            break;
        case 6:
            mass_quads->x = mass_tri0->y;
            mass_quads->y = mass_tri0->z;
            mass_quads->z = mass_tri1->x;
            mass_quads->w = mass_tri1->y;
            break;
        case 7:
            mass_quads->x = mass_tri0->y;
            mass_quads->y = mass_tri0->z;
            mass_quads->z = mass_tri1->x;
            mass_quads->w = mass_tri1->z;
            break;
        case 8:
            mass_quads->x = mass_tri0->y;
            mass_quads->y = mass_tri0->z;
            mass_quads->z = mass_tri1->y;
            mass_quads->w = mass_tri1->z;
            break;
        case 9:
            mass_quads->x = mass_tri0->x;
            mass_quads->y = mass_tri0->y;
            mass_quads->z = mass_tri0->z;
            mass_quads->w = mass_tri1->x;
            break;
        case 10:
            mass_quads->x = mass_tri0->x;
            mass_quads->y = mass_tri0->y;
            mass_quads->z = mass_tri0->z;
            mass_quads->w = mass_tri1->y;
            break;
        case 11:
            mass_quads->x = mass_tri0->x;
            mass_quads->y = mass_tri0->y;
            mass_quads->z = mass_tri0->z;
            mass_quads->w = mass_tri1->z;
            break;
        case 12:
            mass_quads->x = mass_tri0->x;
            mass_quads->y = mass_tri1->x;
            mass_quads->z = mass_tri1->y;
            mass_quads->w = mass_tri1->z;
            break;
        case 13:
            mass_quads->x = mass_tri0->y;
            mass_quads->y = mass_tri1->x;
            mass_quads->z = mass_tri1->y;
            mass_quads->w = mass_tri1->z;
            break;
        case 14:
            mass_quads->x = mass_tri0->z;
            mass_quads->y = mass_tri1->x;
            mass_quads->z = mass_tri1->y;
            mass_quads->w = mass_tri1->z;
            break;
    }

}

template<typename T>
__global__
void
copy_filtered_collisions(MovingTetrahedron<T> *tetrahedrons, long3 *minimal_collisions_tids, T *cubic_roots,
                         vec3<T> *all_masses,
                         bool *is_valid_collision,
                         MovingTetrahedron<T> *filtered_tetrahedrons, long3 *filtered_collisions, T *filtered_roots,
                         vec4<T> *filtered_masses, int num_collisions, int *counter) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;

    if (not is_valid_collision[idx]) return;

    int to_idx = atomicAdd(counter, 1);
    filtered_collisions[to_idx] = minimal_collisions_tids[idx];
    filtered_roots[to_idx] = cubic_roots[idx];
    filtered_tetrahedrons[to_idx] = tetrahedrons[idx];

    long3 *tcoll = minimal_collisions_tids + idx;
    masses2quad<T>(tcoll, all_masses, filtered_masses + to_idx);
}


template<typename T>
__global__
void
build_mass_quads(long3 *minimal_collisions_tids,
                 vec3<T> *triangle_masses,
                 vec4<T> *mass_quads, int num_collisions) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;

    long3 *tcoll = minimal_collisions_tids + idx;
    masses2quad<T>(tcoll, triangle_masses, mass_quads + idx);
}


template<typename T>
__device__ void
compute_coeficients_triangle_vertex(vec3<T> &v0, vec3<T> &v1, vec3<T> &v2, vec3<T> &vt, vec3<T> *coefficients,
                                    int idx) {
    vec3<T> e1 = v0 - v1;
    vec3<T> e2 = v2 - v1;
    vec3<T> ep = vt - v1;
    T d00 = dot(e1, e1);
    T d01 = dot(e1, e2);
    T d11 = dot(e2, e2);
    T d20 = dot(ep, e1);
    T d21 = dot(ep, e2);

    T denom = d00 * d11 - d01 * d01;
    T u;
    T w;
    T v;
    if (abs(denom) < 1e-15) {
        u = 1. / 3.;
        w = 1. / 3.;

    } else {
        u = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
    }

    v = 1 - u - w;

    coefficients[idx].x = u;
    coefficients[idx].y = v;
    coefficients[idx].z = w;
}


template<typename T>
__device__ void
compute_normal_direction_triangle_vertex(vec3<T> &v0, vec3<T> &v1, vec3<T> &v2, vec3<T> &vt,
                                         vec3<T> &dv0, vec3<T> &dv1, vec3<T> &dv2, vec3<T> &dvt,
                                         vec3<T> *coefficients, vec3<T> *normal_directions, T *normal_velocity,
                                         T *normal_distance_end, T root,
                                         int idx, bool inverse) {
    coefficients = coefficients + idx;
    vec3<T> p_on_triangle = v0 * coefficients->x + v1 * coefficients->y + v2 * coefficients->z;
    vec3<T> dp_on_triangle = dv0 * coefficients->x + dv1 * coefficients->y + dv2 * coefficients->z;
    vec3<T> endp_on_triangle = p_on_triangle + dp_on_triangle;
    vec3<T> end_vt = vt + dvt;

    // NEW
    vec3<T> v0_coll = v0 + dv0 * root;
    vec3<T> v1_coll = v1 + dv1 * root;
    vec3<T> v2_coll = v2 + dv2 * root;
    vec3<T> vt_coll = vt + dvt * root;

    vec3<T> e0_coll = v1_coll - v0_coll;
    vec3<T> e1_coll = v2_coll - v0_coll;
    vec3<T> normal_direction = normalize(cross(e0_coll, e1_coll));

    T t2v_norm = dot(vt - p_on_triangle, normal_direction);

    if (t2v_norm < 0) {
        normal_direction = normal_direction * -1;
    }
    // NEW END

    // OLD
//    vec3<T> normal_direction = normalize(vt - p_on_triangle);
    // OLD END


    normal_velocity[idx] = dot(dvt - dp_on_triangle, normal_direction);
    normal_distance_end[idx] = dot(end_vt - endp_on_triangle, normal_direction);

    if (inverse) {
        normal_direction = normal_direction * -1;
    }




    normal_directions[idx] = normal_direction;

//        if (checkNanVec<T>(normal_directions[idx])) {
//            vec3<T> p_on_triangle_next = p_on_triangle + dp_on_triangle;
//            vec3<T> vt_next = vt + dvt;
//            normal_directions[idx] = normalize(vt_next - p_on_triangle_next);
//        }

//            if (checkNanVec<T>(normal_directions[idx])) {
//                normal_directions[idx] = normalize(vt - v1);
//
//                if (checkNanVec<T>(normal_directions[idx])) {
//                    normal_directions[idx] = normalize(vt - v2);
//
//
//                    if (checkNanVec<T>(normal_directions[idx])) {
//                        normal_directions[idx] = normalize(vt - v0);
//
//                    }
//                }
//            }
//
//        }



//    if (checkNanVec<T>(p_on_triangle)) {
//        printf("NAN in p_on_triangle!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(vt)) {
//        printf("NAN in vt!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(normal_directions[idx])) {
//        vec3<T> diff = p_on_triangle - vt;
//        vec3<T> normalized = normalize(diff);
//        vec3<T> p_on_triangle_next = p_on_triangle + dp_on_triangle;
//        vec3<T> vt_next = vt + dvt;
//        printf("vt: {%g, %g, %g}\n", vt.x, vt.y, vt.z);
//        printf("p_on_triangle: {%g, %g, %g}\n", p_on_triangle.x, p_on_triangle.y, p_on_triangle.z);
//        printf("dvt: {%g, %g, %g}\n", dvt.x, dvt.y, dvt.z);
//        printf("dp_on_triangle: {%g, %g, %g}\n", dp_on_triangle.x, dp_on_triangle.y, dp_on_triangle.z);
//        printf("vt_next: {%g, %g, %g}\n", vt_next.x, vt_next.y, vt_next.z);
//        printf("p_on_triangle_next: {%g, %g, %g}\n", p_on_triangle_next.x, p_on_triangle_next.y, p_on_triangle_next.z);
//        printf("diff: {%g, %g, %g}\n", diff.x, diff.y, diff.z);
//        printf("normalized: {%g, %g, %g}\n", normalized.x, normalized.y, normalized.z);
//        printf("NAN in normal_directions idx=%d!\n", idx);
//        assert(false);
//    }
//
//    if (checkNanVec<T>(dp_on_triangle)) {
//        printf("NAN in dp_on_triangle!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(dvt)) {
//        printf("NAN in dvt!\n");
////        assert(false);
//    }

}

template<typename T>
__device__ void
compute_coeficients_edge_edge(vec3<T> &v0, vec3<T> &v1, vec3<T> &v2, vec3<T> &v3, vec3<T> *coefficients, int idx) {
    vec3<T> da = v1 - v0;
    vec3<T> db = v3 - v2;
    vec3<T> dc = v2 - v0;

    T denom = norm2(cross(da, db));

    T s;
    T t;

    if (abs(denom) < 1e-15) {
        s = 0.5;
        t = 0.5;
    } else {
        s = dot(cross(dc, db), cross(da, db)) / norm2(cross(da, db));
        t = dot(cross(dc, da), cross(da, db)) / norm2(cross(da, db));
    }

    coefficients[idx].x = s;
    coefficients[idx].y = t;
}

template<typename T>
__device__ void
compute_normal_direction_edge_edge(vec3<T> &v0, vec3<T> &v1, vec3<T> &v2, vec3<T> &v3,
                                   vec3<T> &dv0, vec3<T> &dv1, vec3<T> &dv2, vec3<T> &dv3,
                                   vec3<T> *coefficients, vec3<T> *normal_directions, T *normal_velocity,
                                   T *normal_distance_end, T root,
                                   int idx) {
    coefficients = coefficients + idx;

    vec3<T> p0 = v0 + (v1 - v0) * coefficients->x;
    vec3<T> p1 = v2 + (v3 - v2) * coefficients->y;

    vec3<T> dp0 = dv0 + (dv1 - dv0) * coefficients->x;
    vec3<T> dp1 = dv2 + (dv3 - dv2) * coefficients->y;

    // NEW
    vec3<T> v0_coll = v0 + dv0 * root;
    vec3<T> v1_coll = v1 + dv1 * root;
    vec3<T> v2_coll = v2 + dv2 * root;
    vec3<T> v3_coll = v3 + dv3 * root;

    vec3<T> e0_coll = v1_coll - v0_coll;
    vec3<T> e1_coll = v2_coll - v0_coll;
    vec3<T> normal_direction = normalize(cross(e0_coll, e1_coll));
    T v2v_norm = dot(p1 - p0, normal_direction);

    if (v2v_norm < 0) {
        normal_direction = normal_direction * -1;
    }
    // NEW END

    // OLD
//    vec3<T> normal_direction = normalize(p1 - p0);
    // OLD END



    normal_directions[idx] = normal_direction;

    normal_velocity[idx] = dot(dp1 - dp0, normal_directions[idx]);
    vec3<T> endp0 = p0 + dp0;
    vec3<T> endp1 = p1 + dp1;
    normal_distance_end[idx] = dot(endp1 - endp0, normal_directions[idx]);

    // OLD
//    normal_directions[idx] = normalize(p1 - p0);
    // OLD END

//    if (checkNanVec<T>(normal_directions[idx])) {
//        vec3<T> p0_next = p0 + dp0;
//        vec3<T> p1_next = p1 + dp1;
//        normal_directions[idx] = normalize(p0_next - p1_next);
//        if (checkNanVec<T>(normal_directions[idx])) {
//            normal_directions[idx] = normalize(v2 - v0);
//
//            if (checkNanVec<T>(normal_directions[idx])) {
//                normal_directions[idx] = normalize(v3 - v1);
//
//
//                if (checkNanVec<T>(normal_directions[idx])) {
//                    normal_directions[idx] = normalize(v3 - v0);
//
//                    if (checkNanVec<T>(normal_directions[idx])) {
//                        normal_directions[idx] = normalize(v2 - v1);
//                    }
//                }
//            }
//        }
//
//    }


//    if (checkNanVec<T>(p0)) {
//        printf("NAN in p0!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(p1)) {
//        printf("NAN in p1!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(normal_directions[idx])) {
//        vec3<T> p0_next = p0 + dp0;
//        vec3<T> p1_next = p1 + dp1;
//
//        vec3<T> diff = p1 - p0;
//        vec3<T> normalized = normalize(diff);
//
//        printf("v0: {%g, %g, %g}\n", v0.x, v0.y, v0.z);
//        printf("v1: {%g, %g, %g}\n", v1.x, v1.y, v1.z);
//        printf("v2: {%g, %g, %g}\n", v2.x, v2.y, v2.z);
//        printf("v3: {%g, %g, %g}\n", v3.x, v3.y, v3.z);
//        printf("coefficients: {%g, %g}\n", coefficients->x, coefficients->y);
//
//        printf("p0: {%g, %g, %g}\n", p0.x, p0.y, p0.z);
//        printf("p1: {%g, %g, %g}\n", p1.x, p1.y, p1.z);
//        printf("dp0: {%g, %g, %g}\n", dp0.x, dp0.y, dp0.z);
//        printf("dp1: {%g, %g, %g}\n", dp1.x, dp1.y, dp1.z);
//        printf("p0_next: {%g, %g, %g}\n", p0_next.x, p0_next.y, p0_next.z);
//        printf("p1_next: {%g, %g, %g}\n", p1_next.x, p1_next.y, p1_next.z);
//        printf("diff: {%g, %g, %g}\n", diff.x, diff.y, diff.z);
//        printf("normalized: {%g, %g, %g}\n", normalized.x, normalized.y, normalized.z);
//        printf("NAN in normal_directions idx=%d!\n", idx);
//    }
//
//    if (checkNanVec<T>(dp0)) {
//        printf("NAN in dp0!\n");
//    }
//
//    if (checkNanVec<T>(dp1)) {
//        printf("NAN in dp1!\n");
//    }




}

template<typename T>
__device__ void
compute_masses_edge_edge(vec3<T> *coefficients, vec4<T> *mass_quad, vec2<T> *mass_pair) {
    mass_pair->x = mass_quad->x * (1 - coefficients->x) + mass_quad->y * coefficients->x;
    mass_pair->y = mass_quad->z * (1 - coefficients->y) + mass_quad->w * coefficients->y;
}

template<typename T>
__device__ void
compute_masses_triangle_vertex(vec3<T> *coefficients, vec4<T> *mass_quad, vec2<T> *mass_pair, bool inverse) {
    if (inverse) {
        mass_pair->x = mass_quad->x;
        mass_pair->y =
                mass_quad->y * coefficients->x + mass_quad->z * coefficients->y + +mass_quad->w * coefficients->z;
    } else {
        mass_pair->x =
                mass_quad->x * coefficients->x + mass_quad->y * coefficients->y + +mass_quad->z * coefficients->z;
        mass_pair->y = mass_quad->w;
    }
}

template<typename T>
__global__
void
compute_coeficients_and_directions(MovingTetrahedron<T> *tetrahedrons, long3 *collisions, T *roots, vec4<T> *mass_quads,
                                   vec3<T> *coefficients,
                                   vec3<T> *normal_directions, T *normal_velocity, T *normal_distance_end,
                                   vec2<T> *mass_pairs,
                                   int num_collisions) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;

    long3 *tcoll = collisions + idx;
    MovingTetrahedron<T> *tetrahedron = tetrahedrons + idx;

    T root = roots[idx];
    Tetrahedron<T> thz = Tetrahedron<T>(tetrahedron, root);

    if (tcoll->z < 9) {
        compute_coeficients_edge_edge<T>(thz.v0, thz.v1, thz.v2, thz.v3, coefficients, idx);
        compute_normal_direction_edge_edge<T>(tetrahedron->v0, tetrahedron->v1, tetrahedron->v2, tetrahedron->v3,
                                              tetrahedron->dv0, tetrahedron->dv1, tetrahedron->dv2, tetrahedron->dv3,
                                              coefficients, normal_directions, normal_velocity, normal_distance_end,
                                              root, idx);
        compute_masses_edge_edge<T>(coefficients + idx, mass_quads + idx, mass_pairs + idx);
//        printf("\ntcoll: (%d, %d, %d)\n"
//               "root: %g\n"
//               "s: %g\tt: %f\n"
//               "normal direction: (%g, %g, %g)\n"
//               "normal_velocity: %g\n",
//               (int) tcoll->x, (int) tcoll->y, (int) tcoll->z,
//               root, coefficients[idx].x, coefficients[idx].y,
//               normal_directions[idx].x, normal_directions[idx].y, normal_directions[idx].z,
//               normal_velocity[idx]);
    } else if (tcoll->z < 12) {
        compute_coeficients_triangle_vertex<T>(thz.v0, thz.v1, thz.v2, thz.v3, coefficients, idx);
        compute_normal_direction_triangle_vertex<T>(tetrahedron->v0, tetrahedron->v1, tetrahedron->v2, tetrahedron->v3,
                                                    tetrahedron->dv0, tetrahedron->dv1, tetrahedron->dv2,
                                                    tetrahedron->dv3,
                                                    coefficients, normal_directions, normal_velocity,
                                                    normal_distance_end, root, idx, false);
        compute_masses_triangle_vertex<T>(coefficients + idx, mass_quads + idx, mass_pairs + idx, false);
//        printf("\ntcoll: (%d, %d, %d)\n"
//               "u: %g\tv: %f\tw: %f\n"
//               "normal direction: (%g, %g, %g)\n"
//               "normal_velocity: %g\n",
//               (int) tcoll->x, (int) tcoll->y, (int) tcoll->z,
//               coefficients[idx].x, coefficients[idx].y, coefficients[idx].z,
//               normal_directions[idx].x, normal_directions[idx].y, normal_directions[idx].z,
//               normal_velocity[idx]);
    } else {
        compute_coeficients_triangle_vertex<T>(thz.v1, thz.v2, thz.v3, thz.v0, coefficients, idx);
        compute_normal_direction_triangle_vertex<T>(tetrahedron->v1, tetrahedron->v2, tetrahedron->v3, tetrahedron->v0,
                                                    tetrahedron->dv1, tetrahedron->dv2, tetrahedron->dv3,
                                                    tetrahedron->dv0,
                                                    coefficients, normal_directions, normal_velocity,
                                                    normal_distance_end, root, idx, true);
        compute_masses_triangle_vertex<T>(coefficients + idx, mass_quads + idx, mass_pairs + idx, true);
//        printf("\ntcoll: (%d, %d, %d)\n"
//               "u: %g\tv: %f\tw: %f\n"
//               "normal direction: (%g, %g, %g)\n"
//               "normal_velocity: %g\n",
//               (int) tcoll->x, (int) tcoll->y, (int) tcoll->z,
//               coefficients[idx].x, coefficients[idx].y, coefficients[idx].z,
//               normal_directions[idx].x, normal_directions[idx].y, normal_directions[idx].z,
//               normal_velocity[idx]);
    }

//    printf("OUTER idx: %d\n",  idx);
//    if (checkNanVec<T>(normal_directions[idx])) {
//        printf("ROOT: %g\n", root);
//        printf("NAN in normal_directions OUTER!\n");
////        assert(false);
//    }
}


template<typename T>
__device__ void
atomicAddVec(vec3<T> &v0, vec3<T> &v1) {
//    bool isnan_input_v0 = checkNanVec<T>(v0);
//    bool isnan_input_v1 =
//            checkNanVec<T>(v1);

    atomicAdd(&v0.x, v1.x);
    atomicAdd(&v0.y, v1.y);
    atomicAdd(&v0.z, v1.z);
//
//    bool isnan_output =
//            checkNanVec<T>(v0);
//
//    if (isnan_input_v0) {
//        printf("NAN in atomicAdd input v0!\n");
//    }
//
//    if (isnan_input_v1) {
//        printf("NAN in atomicAdd input v1!\n");
//    }
//
//    if (isnan_output) {
//        printf("NAN in atomicAdd output!\n");
//    }
}


template<typename T>
__device__ bool
checkNanTriangle(Triangle<T> *tri) {
//    return false;
    return checkNanVec<T>(tri->v0) or checkNanVec<T>(tri->v1) or checkNanVec<T>(tri->v0);
}

template<typename T>
__device__ void
initTriangle(Triangle<T> *tri) {

    tri->v0 = {0., 0., 0.};
    tri->v1 = {0., 0., 0.};
    tri->v2 = {0., 0., 0.};
}

template<typename T>
__global__
void
init_impulses(Triangle<T> *impulses_dv, Triangle<T> *impulses_dx, int num_collisions) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;

    Triangle<T> *t0_dv = impulses_dv + idx;
    Triangle<T> *t1_dv = impulses_dv + idx;
    Triangle<T> *t0_dx = impulses_dx + idx;
    Triangle<T> *t1_dx = impulses_dx + idx;

    initTriangle(t0_dv);
    initTriangle(t1_dv);
    initTriangle(t0_dx);
    initTriangle(t1_dx);
}


template<typename T>
__device__ void
extrapolate_impulses_dv(MovingTetrahedron<T> *tetrahedron, T root, T offset,
                        Triangle<T> *triangle0_dv, Triangle<T> *triangle1_dv,
                        Triangle<T> *triangle0_dx, Triangle<T> *triangle1_dx,
                        int3 *triangle0_counter, int3 *triangle1_counter,
                        vec3<T> &v0, vec3<T> &v1, vec3<T> &v2,
                        vec3<T> &v3, int collision_id) {

    vec3<T> nv0 = (tetrahedron->dv0 + v0);
    vec3<T> nv1 = (tetrahedron->dv1 + v1);
    vec3<T> nv2 = (tetrahedron->dv2 + v2);
    vec3<T> nv3 = (tetrahedron->dv3 + v3);

    T root_offset = max(root - offset, 0.);


    vec3<T> imp0_dx = (nv0 - tetrahedron->dv0) * (1 - root_offset);
    vec3<T> imp1_dx = (nv1 - tetrahedron->dv1) * (1 - root_offset);
    vec3<T> imp2_dx = (nv2 - tetrahedron->dv2) * (1 - root_offset);
    vec3<T> imp3_dx = (nv3 - tetrahedron->dv3) * (1 - root_offset);

//    if (checkNanVec<T>(tetrahedron->dv0)) {
//        printf("NAN in dv0!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(tetrahedron->dv1)) {
//        printf("NAN in dv1!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(tetrahedron->dv2)) {
//        printf("NAN in dv2!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(tetrahedron->dv3)) {
//        printf("NAN in dv3!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(v0)) {
//        printf("NAN in v0!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(v1)) {
//        printf("NAN in v1!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(v2)) {
//        printf("NAN in v2!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(v3)) {
//        printf("NAN in v3!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(imp0_dx)) {
//        printf("NAN in imp0_dx!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(imp1_dx)) {
//        printf("NAN in imp1_dx!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(imp2_dx)) {
//        printf("NAN in imp2_dx!\n");
////        assert(false);
//    }
//
//    if (checkNanVec<T>(imp3_dx)) {
//        printf("NAN in imp3_dx!\n");
////        assert(false);
//    }

    switch (collision_id) {
        case 0:
            atomicAddVec<T>(triangle0_dv->v0, v0);
            atomicAddVec<T>(triangle0_dv->v1, v1);
            atomicAddVec<T>(triangle1_dv->v0, v2);
            atomicAddVec<T>(triangle1_dv->v1, v3);

            atomicAddVec<T>(triangle0_dx->v0, imp0_dx);
            atomicAddVec<T>(triangle0_dx->v1, imp1_dx);
            atomicAddVec<T>(triangle1_dx->v0, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v1, imp3_dx);

            atomicAdd(&triangle0_counter->x, 1);
            atomicAdd(&triangle0_counter->y, 1);
            atomicAdd(&triangle1_counter->x, 1);
            atomicAdd(&triangle1_counter->y, 1);
            break;
        case 1:
            atomicAddVec<T>(triangle0_dv->v0, v0);
            atomicAddVec<T>(triangle0_dv->v1, v1);
            atomicAddVec<T>(triangle1_dv->v0, v2);
            atomicAddVec<T>(triangle1_dv->v2, v3);

            atomicAddVec<T>(triangle0_dx->v0, imp0_dx);
            atomicAddVec<T>(triangle0_dx->v1, imp1_dx);
            atomicAddVec<T>(triangle1_dx->v0, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v2, imp3_dx);

            atomicAdd(&triangle0_counter->x, 1);
            atomicAdd(&triangle0_counter->y, 1);
            atomicAdd(&triangle1_counter->x, 1);
            atomicAdd(&triangle1_counter->z, 1);
            break;
        case 2:
            atomicAddVec<T>(triangle0_dv->v0, v0);
            atomicAddVec<T>(triangle0_dv->v1, v1);
            atomicAddVec<T>(triangle1_dv->v1, v2);
            atomicAddVec<T>(triangle1_dv->v2, v3);

            atomicAddVec<T>(triangle0_dx->v0, imp0_dx);
            atomicAddVec<T>(triangle0_dx->v1, imp1_dx);
            atomicAddVec<T>(triangle1_dx->v1, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v2, imp3_dx);

            atomicAdd(&triangle0_counter->x, 1);
            atomicAdd(&triangle0_counter->y, 1);
            atomicAdd(&triangle1_counter->y, 1);
            atomicAdd(&triangle1_counter->z, 1);
            break;
        case 3:
            atomicAddVec<T>(triangle0_dv->v0, v0);
            atomicAddVec<T>(triangle0_dv->v2, v1);
            atomicAddVec<T>(triangle1_dv->v0, v2);
            atomicAddVec<T>(triangle1_dv->v1, v3);

            atomicAddVec<T>(triangle0_dx->v0, imp0_dx);
            atomicAddVec<T>(triangle0_dx->v2, imp1_dx);
            atomicAddVec<T>(triangle1_dx->v0, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v1, imp3_dx);

            atomicAdd(&triangle0_counter->x, 1);
            atomicAdd(&triangle0_counter->z, 1);
            atomicAdd(&triangle1_counter->x, 1);
            atomicAdd(&triangle1_counter->y, 1);
            break;
        case 4:
            atomicAddVec<T>(triangle0_dv->v0, v0);
            atomicAddVec<T>(triangle0_dv->v2, v1);
            atomicAddVec<T>(triangle1_dv->v0, v2);
            atomicAddVec<T>(triangle1_dv->v2, v3);

            atomicAddVec<T>(triangle0_dx->v0, imp0_dx);
            atomicAddVec<T>(triangle0_dx->v2, imp1_dx);
            atomicAddVec<T>(triangle1_dx->v0, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v2, imp3_dx);

            atomicAdd(&triangle0_counter->x, 1);
            atomicAdd(&triangle0_counter->z, 1);
            atomicAdd(&triangle1_counter->x, 1);
            atomicAdd(&triangle1_counter->z, 1);
            break;
        case 5:
            atomicAddVec<T>(triangle0_dv->v0, v0);
            atomicAddVec<T>(triangle0_dv->v2, v1);
            atomicAddVec<T>(triangle1_dv->v1, v2);
            atomicAddVec<T>(triangle1_dv->v2, v3);

            atomicAddVec<T>(triangle0_dx->v0, imp0_dx);
            atomicAddVec<T>(triangle0_dx->v2, imp1_dx);
            atomicAddVec<T>(triangle1_dx->v1, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v2, imp3_dx);

            atomicAdd(&triangle0_counter->x, 1);
            atomicAdd(&triangle0_counter->z, 1);
            atomicAdd(&triangle1_counter->y, 1);
            atomicAdd(&triangle1_counter->z, 1);
            break;
        case 6:
            atomicAddVec<T>(triangle0_dv->v1, v0);
            atomicAddVec<T>(triangle0_dv->v2, v1);
            atomicAddVec<T>(triangle1_dv->v0, v2);
            atomicAddVec<T>(triangle1_dv->v1, v3);

            atomicAddVec<T>(triangle0_dx->v1, imp0_dx);
            atomicAddVec<T>(triangle0_dx->v2, imp1_dx);
            atomicAddVec<T>(triangle1_dx->v0, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v1, imp3_dx);

            atomicAdd(&triangle0_counter->y, 1);
            atomicAdd(&triangle0_counter->z, 1);
            atomicAdd(&triangle1_counter->x, 1);
            atomicAdd(&triangle1_counter->y, 1);
            break;
        case 7:
            atomicAddVec<T>(triangle0_dv->v1, v0);
            atomicAddVec<T>(triangle0_dv->v2, v1);
            atomicAddVec<T>(triangle1_dv->v0, v2);
            atomicAddVec<T>(triangle1_dv->v2, v3);

            atomicAddVec<T>(triangle0_dx->v1, imp0_dx);
            atomicAddVec<T>(triangle0_dx->v2, imp1_dx);
            atomicAddVec<T>(triangle1_dx->v0, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v2, imp3_dx);

            atomicAdd(&triangle0_counter->y, 1);
            atomicAdd(&triangle0_counter->z, 1);
            atomicAdd(&triangle1_counter->x, 1);
            atomicAdd(&triangle1_counter->z, 1);
            break;
        case 8:
            atomicAddVec<T>(triangle0_dv->v1, v0);
            atomicAddVec<T>(triangle0_dv->v2, v1);
            atomicAddVec<T>(triangle1_dv->v1, v2);
            atomicAddVec<T>(triangle1_dv->v2, v3);

            atomicAddVec<T>(triangle0_dx->v1, imp0_dx);
            atomicAddVec<T>(triangle0_dx->v2, imp1_dx);
            atomicAddVec<T>(triangle1_dx->v1, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v2, imp3_dx);

            atomicAdd(&triangle0_counter->y, 1);
            atomicAdd(&triangle0_counter->z, 1);
            atomicAdd(&triangle1_counter->y, 1);
            atomicAdd(&triangle1_counter->z, 1);
            break;
        case 9:
            atomicAddVec<T>(triangle0_dv->v0, v0);
            atomicAddVec<T>(triangle0_dv->v1, v1);
            atomicAddVec<T>(triangle0_dv->v2, v2);
            atomicAddVec<T>(triangle1_dv->v0, v3);

            atomicAddVec<T>(triangle0_dx->v0, imp0_dx);
            atomicAddVec<T>(triangle0_dx->v1, imp1_dx);
            atomicAddVec<T>(triangle0_dx->v2, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v0, imp3_dx);

            atomicAdd(&triangle0_counter->x, 1);
            atomicAdd(&triangle0_counter->y, 1);
            atomicAdd(&triangle0_counter->z, 1);
            atomicAdd(&triangle1_counter->x, 1);
            break;
        case 10:
            atomicAddVec<T>(triangle0_dv->v0, v0);
            atomicAddVec<T>(triangle0_dv->v1, v1);
            atomicAddVec<T>(triangle0_dv->v2, v2);
            atomicAddVec<T>(triangle1_dv->v1, v3);

            atomicAddVec<T>(triangle0_dx->v0, imp0_dx);
            atomicAddVec<T>(triangle0_dx->v1, imp1_dx);
            atomicAddVec<T>(triangle0_dx->v2, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v1, imp3_dx);

            atomicAdd(&triangle0_counter->x, 1);
            atomicAdd(&triangle0_counter->y, 1);
            atomicAdd(&triangle0_counter->z, 1);
            atomicAdd(&triangle1_counter->y, 1);
            break;
        case 11:
            atomicAddVec<T>(triangle0_dv->v0, v0);
            atomicAddVec<T>(triangle0_dv->v1, v1);
            atomicAddVec<T>(triangle0_dv->v2, v2);
            atomicAddVec<T>(triangle1_dv->v2, v3);

            atomicAddVec<T>(triangle0_dx->v0, imp0_dx);
            atomicAddVec<T>(triangle0_dx->v1, imp1_dx);
            atomicAddVec<T>(triangle0_dx->v2, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v2, imp3_dx);

            atomicAdd(&triangle0_counter->x, 1);
            atomicAdd(&triangle0_counter->y, 1);
            atomicAdd(&triangle0_counter->z, 1);
            atomicAdd(&triangle1_counter->z, 1);
            break;
        case 12:
            atomicAddVec<T>(triangle0_dv->v0, v0);
            atomicAddVec<T>(triangle1_dv->v0, v1);
            atomicAddVec<T>(triangle1_dv->v1, v2);
            atomicAddVec<T>(triangle1_dv->v2, v3);

            atomicAddVec<T>(triangle0_dx->v0, imp0_dx);
            atomicAddVec<T>(triangle1_dx->v0, imp1_dx);
            atomicAddVec<T>(triangle1_dx->v1, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v2, imp3_dx);

            atomicAdd(&triangle0_counter->x, 1);
            atomicAdd(&triangle1_counter->x, 1);
            atomicAdd(&triangle1_counter->y, 1);
            atomicAdd(&triangle1_counter->z, 1);
            break;
        case 13:
            atomicAddVec<T>(triangle0_dv->v1, v0);
            atomicAddVec<T>(triangle1_dv->v0, v1);
            atomicAddVec<T>(triangle1_dv->v1, v2);
            atomicAddVec<T>(triangle1_dv->v2, v3);

            atomicAddVec<T>(triangle0_dx->v1, imp0_dx);
            atomicAddVec<T>(triangle1_dx->v0, imp1_dx);
            atomicAddVec<T>(triangle1_dx->v1, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v2, imp3_dx);

            atomicAdd(&triangle0_counter->y, 1);
            atomicAdd(&triangle1_counter->x, 1);
            atomicAdd(&triangle1_counter->y, 1);
            atomicAdd(&triangle1_counter->z, 1);
            break;
        case 14:
            atomicAddVec<T>(triangle0_dv->v2, v0);
            atomicAddVec<T>(triangle1_dv->v0, v1);
            atomicAddVec<T>(triangle1_dv->v1, v2);
            atomicAddVec<T>(triangle1_dv->v2, v3);

            atomicAddVec<T>(triangle0_dx->v2, imp0_dx);
            atomicAddVec<T>(triangle1_dx->v0, imp1_dx);
            atomicAddVec<T>(triangle1_dx->v1, imp2_dx);
            atomicAddVec<T>(triangle1_dx->v2, imp3_dx);

            atomicAdd(&triangle0_counter->z, 1);
            atomicAdd(&triangle1_counter->x, 1);
            atomicAdd(&triangle1_counter->y, 1);
            atomicAdd(&triangle1_counter->z, 1);
            break;
    }
}


template<typename T>
__global__
void
extrapolate_impulses(MovingTetrahedron<T> *tetrahedrons, T *roots, long3 *collisions, vec3<T> *coefficients,
                     vec3<T> *normal_directions, T *normal_velocity, T *normal_distance_end, vec2<T> *mass_pairs,
                     Triangle<T> *impulses_dv, Triangle<T> *impulses_dx, int3 *counters,
                     int num_collisions) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;

    long3 *tcoll = collisions + idx;
    MovingTetrahedron<T> *tetrahedron = tetrahedrons + idx;

    Triangle<T> *t0_dv = impulses_dv + tcoll->x;
    Triangle<T> *t1_dv = impulses_dv + tcoll->y;
//
//    bool isnan_input_dv = checkNanTriangle<T>(t0_dv) or checkNanTriangle<T>(t1_dv);
//    if (isnan_input_dv) {
//        printf("NAN in input dv!\n");
//    }

    Triangle<T> *t0_dx = impulses_dx + tcoll->x;
    Triangle<T> *t1_dx = impulses_dx + tcoll->y;

    int3 *t0_counter = counters + tcoll->x;
    int3 *t1_counter = counters + tcoll->y;

    coefficients = coefficients + idx;

    vec3<T> imp0_dv;
    vec3<T> imp1_dv;
    vec3<T> imp2_dv;
    vec3<T> imp3_dv;

    T impulse_magnitude = normal_velocity[idx];
    T offset;
    if (abs(normal_velocity[idx]) < 1e-15) {
        offset = 0.1;
    } else {
        offset = 2 * THICKNESS / -normal_velocity[idx];
    }

    offset = min(max(offset, 0.), 1.);

//    offset = 0.1;

    T root = roots[idx];

    T mass_sum = mass_pairs[idx].x + mass_pairs[idx].y;

    T mass_fraction_0;
    T mass_fraction_1;
    if (abs(mass_sum) < 1e-15) {
        mass_fraction_0 = 0.5;
        mass_fraction_1 = 0.5;
    } else {
        mass_fraction_0 = mass_pairs[idx].x / mass_sum;
        mass_fraction_1 = 1 - mass_fraction_0;
    }

//    T impulse_magnitude_0 = impulse_magnitude / 2;
//    T impulse_magnitude_1 = -impulse_magnitude / 2;
//
//    if (isnan(mass_sum)) {
//        printf("NAN in mass_sum!\n");
//    }
//    if (isnan(mass_fraction_1)) {
//        printf("NAN in mass_fraction_1!\n");
//    }
//    if (isnan(mass_fraction_0)) {
//        printf("NAN in mass_fraction_0!\n");
//    }
//
//    if (isnan(impulse_magnitude)) {
//        printf("NAN in impulse_magnitude!\n");
//    }

    T impulse_magnitude_0 = impulse_magnitude * mass_fraction_1;
    T impulse_magnitude_1 = -impulse_magnitude * mass_fraction_0;


    if (tcoll->z < 9) {
        imp0_dv = normal_directions[idx] * impulse_magnitude_0;
        imp1_dv = normal_directions[idx] * impulse_magnitude_0;
        imp2_dv = normal_directions[idx] * impulse_magnitude_1;
        imp3_dv = normal_directions[idx] * impulse_magnitude_1;
    } else if (tcoll->z < 12) {
        imp0_dv = normal_directions[idx] * impulse_magnitude_0;
        imp1_dv = normal_directions[idx] * impulse_magnitude_0;
        imp2_dv = normal_directions[idx] * impulse_magnitude_0;
        imp3_dv = normal_directions[idx] * impulse_magnitude_1;
    } else {
        imp0_dv = normal_directions[idx] * impulse_magnitude_0;
        imp1_dv = normal_directions[idx] * impulse_magnitude_1;
        imp2_dv = normal_directions[idx] * impulse_magnitude_1;
        imp3_dv = normal_directions[idx] * impulse_magnitude_1;
    }

//    if (checkNanVec<T>(normal_directions[idx])) {
//        printf("NAN in normal_directions! idx: %d\n", idx);
//    }
//
//    if (isnan(impulse_magnitude_0)) {
//        printf("NAN in impulse_magnitude_0!\n");
//    }
////
//    if (isnan(impulse_magnitude_1)) {
//        printf("NAN in impulse_magnitude_1!\n");
//    }

    extrapolate_impulses_dv<T>(tetrahedron, root, offset,
                               t0_dv, t1_dv,
                               t0_dx, t1_dx,
                               t0_counter, t1_counter,
                               imp0_dv, imp1_dv, imp2_dv, imp3_dv, tcoll->z);

//    bool isnan_input =
//            checkNanVec<T>(imp0_dv) or checkNanVec<T>(imp1_dv) or checkNanVec<T>(imp2_dv) or checkNanVec<T>(imp3_dv);
//    if (isnan_input) {
//        printf("NAN in impulses!\n");
//    }

//    bool isnan_output_dv = checkNanTriangle<T>(t0_dv) or checkNanTriangle<T>(t1_dv);
//    bool isnan_output_dx = checkNanTriangle<T>(t0_dx) or checkNanTriangle<T>(t1_dx);
//    if (isnan_output_dv) {
//        printf("NAN in dv!\n");
//    }
//    if (isnan_output_dx) {
//        printf("NAN in dx!\n");
//    }

}

void collision_impulses_from_bbox_tree(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                       at::Tensor triangles_next_tensor,
                                       at::Tensor triangles_mass_tensor,
                                       at::Tensor *impulses_dv_tensor_ptr, at::Tensor *impulses_dx_tensor_ptr,
                                       at::Tensor *impulses_counter_tensor_ptr,
                                       int max_candidates_per_triangle,
                                       int max_collisions_per_triangle) {
    const auto batch_size = bboxes_tensor.size(0);
    const auto num_triangles = (bboxes_tensor.size(1) + 1) / 2;
    const auto num_nodes = bboxes_tensor.size(1);
    const auto max_total_candidates = max_candidates_per_triangle * num_triangles;


//    printf("IMP: collision_impulses_from_bbox_tree\n");
//    return;

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
                auto triangle_mass_float_ptr = triangles_mass_tensor.data<scalar_t>();

                for (int bidx = 0; bidx < batch_size; ++bidx) {

                    ////                             cpt-12
                    AABB<scalar_t> *bboxes_ptr =
                            (AABB<scalar_t> *) bboxes_float_ptr +
                            num_nodes * bidx;
                    long4 *tree_ptr = (long4 *) tree_long_ptr +
                                      num_nodes * bidx;


                    reconstruct_bvh<scalar_t>(bboxes_ptr, tree_ptr, internal_nodes.data().get(),
                                              leaf_nodes.data().get(), triangle_ids.data().get(), num_nodes,
                                              num_triangles);
                    cudaCheckError();
//

                    int gridSize = (num_triangles + blockSize - 1) / blockSize;
////                             cpt-10
//                     ================================ findPotentialCollisions
                    thrust::fill(collisionIndices.begin(), collisionIndices.end(),
                                 make_long2(-1, -1));

////                             cpt-10.1
                    findPotentialCollisions<scalar_t><<<gridSize, blockSize>>>(
                            collisionIndices.data().get(),
                            internal_nodes.data().get(),
                            leaf_nodes.data().get(), triangle_ids.data().get(), num_triangles,
                            max_total_candidates, &collision_idx_cnt.data().get()[bidx]);
                    cudaDeviceSynchronize();
////                             cpt-10.2
                    if ((int) collision_idx_cnt[bidx] > max_total_candidates) {
                        printf("Number of candidate collisions exceeds maximal allowed number. Some of the candidate collisions are omitted.\n");
                    }
                    int num_cand_collisions =
                            thrust::reduce(thrust::make_transform_iterator(
                                                   collisionIndices.begin(), is_valid_cnt()),
                                           thrust::make_transform_iterator(
                                                   collisionIndices.end(), is_valid_cnt()));

//                    printf("num_cand_collisions: %d\n", num_cand_collisions);

////                             cpt-9
                    if (num_cand_collisions > 0) {
                        Triangle<scalar_t> *triangles_ptr =
                                (TrianglePtr<scalar_t>) triangle_float_ptr +
                                num_triangles * bidx;
                        Triangle<scalar_t> *triangles_next_ptr =
                                (TrianglePtr<scalar_t>) triangle_next_float_ptr +
                                num_triangles * bidx;
                        vec3<scalar_t> *triangles_mass_ptr =
                                (vec3<scalar_t> *) triangle_mass_float_ptr +
                                num_triangles * bidx;

////                             cpt-8
                        // Keep only the pairs of ids where a bounding box to bounding box
                        // collision was detected.
                        thrust::device_vector <long2> triangle_collisions(num_cand_collisions,
                                                                          make_long2(-1, -1));

                        thrust::copy_if(collisionIndices.begin(), collisionIndices.end(),
                                        triangle_collisions.begin(), is_valid_cnt());

                        cudaCheckError();

                        int tri_grid_size =
                                (num_triangles + blockSize - 1) / blockSize;
//
////                             cpt-7
                        thrust::device_vector <MovingTriangle<scalar_t>> moving_triangles(num_triangles);
                        make_moving_triangles<
                                scalar_t><<<tri_grid_size, blockSize>>>(triangles_ptr, triangles_next_ptr,
                                                                        moving_triangles.data().get(),
                                                                        num_triangles);
                        cudaCheckError();


////                             cpt-6
                        thrust::device_vector<bool> minimal_collision_flags(num_cand_collisions * 15, 0);
////                             cpt-6.1

                        int coll_grid_size = (triangle_collisions.size() + blockSize - 1) / blockSize;
                        dim3 blockGrid(coll_grid_size, 15);

////                             cpt-6.2
                        checkMinimalCollisions<scalar_t><<<blockGrid, blockSize>>>(
                                triangle_collisions.data().get(),
                                moving_triangles.data().get(), minimal_collision_flags.data().get(),
                                num_cand_collisions);
                        cudaCheckError();


////                             cpt-5
                        int num_cminimal_collisions_flags =
                                thrust::reduce(thrust::make_transform_iterator(
                                                       minimal_collision_flags.begin(), is_true()),
                                               thrust::make_transform_iterator(
                                                       minimal_collision_flags.end(), is_true()));
                        thrust::device_vector <long3> minimal_collisions(num_cminimal_collisions_flags,
                                                                         make_long3(-1, -1, -1));
////                             cpt-5.1
                        copyMinimalCollisions<scalar_t><<<blockGrid, blockSize>>>(
                                triangle_collisions.data().get(), minimal_collisions.data().get(),
                                minimal_collision_flags.data().get(),
                                num_cand_collisions, &minimal_collision_idx_cnt.data().get()[bidx]);
                        cudaCheckError();


                        int num_cminimal_collisions = minimal_collision_idx_cnt[bidx];

////                             cpt-4
                        int min_coll_grid_size = (num_cminimal_collisions + blockSize - 1) / blockSize;
                        thrust::device_vector <MovingTetrahedron<scalar_t>> tetrahedrons(num_cminimal_collisions);
////                             cpt-4.1
                        make_tetrahedrons<scalar_t><<<min_coll_grid_size, blockSize>>>(
                                minimal_collisions.data().get(),
                                moving_triangles.data().get(), tetrahedrons.data().get(), num_cminimal_collisions);
                        cudaCheckError();

////                             cpt-3
                        thrust::device_vector <vec3<scalar_t>> cubic_roots_triplets(num_cminimal_collisions,
                                                                                    {-1, -1, -1});
////                             cpt-3.1
                        compute_cubic_roots<scalar_t>(tetrahedrons.data().get(), cubic_roots_triplets.data().get(),
                                                      minimal_collisions.data().get(),
                                                      num_cminimal_collisions);
                        cudaCheckError();

                        int num_valid_roots =
                                thrust::reduce(thrust::make_transform_iterator(
                                                       cubic_roots_triplets.begin(), num_roots<scalar_t>()),
                                               thrust::make_transform_iterator(
                                                       cubic_roots_triplets.end(), num_roots<scalar_t>()));

////                             cpt-2
                        if (num_valid_roots > 0) {

                            thrust::device_vector <MovingTetrahedron<scalar_t>> tetrahedrons_valid(num_valid_roots);
                            thrust::device_vector <long3> minimal_collisions_valid(num_valid_roots,
                                                                                   make_long3(-1, -1, -1));
////                             cpt-2.1
//
//
                            thrust::device_vector <scalar_t> cubic_roots(num_valid_roots, -1);
                            flatten_cubic_roots<scalar_t><<<min_coll_grid_size, blockSize>>>(tetrahedrons.data().get(),
                                                                                             minimal_collisions.data().get(),
                                                                                             cubic_roots_triplets.data().get(),
                                                                                             tetrahedrons_valid.data().get(),
                                                                                             minimal_collisions_valid.data().get(),
                                                                                             cubic_roots.data().get(),
                                                                                             num_cminimal_collisions,
                                                                                             &valid_roots_idx_cnt.data().get()[bidx]);
                            cudaCheckError();
////                             cpt-2.2
                            int valroots_grid_size = (num_valid_roots + blockSize - 1) / blockSize;
                            int num_valid_roots_counter = valid_roots_idx_cnt[bidx];
                            //                            printf("num_valid_roots_counter: %d\n", num_valid_roots_counter);

////                            printf("IMP: num_valid_roots=%d\n", num_valid_roots);
////                            printf("IMP: num_valid_roots_counter=%d\n", num_valid_roots_counter);
////                            printf("IMP: blockSize=%d\n", blockSize);
////                            printf("IMP: valroots_grid_size=%d\n", valroots_grid_size);
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

////                             cpt-1
                            if (num_valid_collisions > 0) {
                                thrust::device_vector <Triangle<scalar_t>> impulses_dv(num_triangles);
                                thrust::device_vector <Triangle<scalar_t>> impulses_dx(num_triangles);
                                thrust::device_vector <int3> impulse_counters(num_triangles, make_int3(0, 0, 0));

                                thrust::device_vector <long3> filtered_collisions(num_valid_collisions,
                                                                                  make_long3(-1, -1, -1));
                                thrust::device_vector <vec4<scalar_t>> mass_quads(num_valid_collisions,
                                                                                  {0., 0., 0., 0.});
                                thrust::device_vector <scalar_t> filtered_cubic_roots(num_valid_collisions);
                                thrust::device_vector <MovingTetrahedron<scalar_t>> filtered_tetrahedrons(
                                        num_valid_collisions);

                                copy_filtered_collisions<scalar_t><<<valroots_grid_size, blockSize>>>(
                                        tetrahedrons_valid.data().get(),
                                        minimal_collisions_valid.data().get(),
                                        cubic_roots.data().get(),
                                        triangles_mass_ptr,
                                        is_valid_collision.data().get(),
                                        filtered_tetrahedrons.data().get(),
                                        filtered_collisions.data().get(),
                                        filtered_cubic_roots.data().get(),
                                        mass_quads.data().get(),
                                        num_valid_roots,
                                        &filtered_minimal_collision_idx_cnt.data().get()[bidx]);
                                //
                                //
                                thrust::device_vector <vec3<scalar_t>> coefficients(num_valid_collisions, {-1, -1, -1});
                                thrust::device_vector <vec3<scalar_t>> normal_directions(num_valid_collisions,
                                                                                         {-1, -1, -1});
                                thrust::device_vector <scalar_t> normal_velocity(num_valid_collisions);
                                thrust::device_vector <scalar_t> normal_distance_end(num_valid_collisions);
                                thrust::device_vector <vec2<scalar_t>> mass_pairs(num_valid_collisions,
                                                                                  {0., 0.});

                                int valid_grid_size = (num_valid_collisions + blockSize - 1) / blockSize;
                                compute_coeficients_and_directions<scalar_t><<<valid_grid_size, blockSize>>>(
                                        filtered_tetrahedrons.data().get(),
                                        filtered_collisions.data().get(),
                                        filtered_cubic_roots.data().get(),
                                        mass_quads.data().get(),
                                        coefficients.data().get(),
                                        normal_directions.data().get(),
                                        normal_velocity.data().get(),
                                        normal_distance_end.data().get(),
                                        mass_pairs.data().get(),
                                        num_valid_collisions);
                                //

                                init_impulses<scalar_t><<<tri_grid_size, blockSize>>>(
                                        impulses_dv.data().get(),
                                        impulses_dx.data().get(),
                                        num_triangles);

//
                                extrapolate_impulses<scalar_t><<<valid_grid_size, blockSize>>>(
                                        filtered_tetrahedrons.data().get(),
                                        filtered_cubic_roots.data().get(),
                                        filtered_collisions.data().get(),
                                        coefficients.data().get(),
                                        normal_directions.data().get(),
                                        normal_velocity.data().get(),
                                        normal_distance_end.data().get(),
                                        mass_pairs.data().get(),
                                        impulses_dv.data().get(),
                                        impulses_dx.data().get(),
                                        impulse_counters.data().get(),
                                        num_valid_collisions);
                                //
                                scalar_t *impulses_dv_ptr = impulses_dv_tensor_ptr->data<scalar_t>();
                                cudaMemcpy(impulses_dv_ptr + bidx * num_triangles * 3 * 3,
                                           (scalar_t *) impulses_dv.data().get(),
                                           num_triangles * 3 * 3 * sizeof(scalar_t),
                                           cudaMemcpyDeviceToDevice);
                                cudaCheckError();

                                scalar_t *impulses_dx_ptr = impulses_dx_tensor_ptr->data<scalar_t>();
                                cudaMemcpy(impulses_dx_ptr + bidx * num_triangles * 3 * 3,
                                           (scalar_t *) impulses_dx.data().get(),
                                           num_triangles * 3 * 3 * sizeof(scalar_t),
                                           cudaMemcpyDeviceToDevice);
                                cudaCheckError();

                                int *impulses_counter_ptr = impulses_counter_tensor_ptr->data<int>();
                                cudaMemcpy(impulses_counter_ptr + bidx * num_triangles * 3,
                                           (int *) impulse_counters.data().get(),
                                           num_triangles * 3 * sizeof(int),
                                           cudaMemcpyDeviceToDevice);
                                cudaCheckError();
                            }
                        }
                    }


                }

            }));

}




void collision_impulses_from_bbox_tree_partial(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                       at::Tensor triangles_next_tensor,
                                       at::Tensor triangles_mass_tensor,
                                       at::Tensor triangles_to_check_tensor,
                                       at::Tensor *impulses_dv_tensor_ptr, at::Tensor *impulses_dx_tensor_ptr,
                                       at::Tensor *impulses_counter_tensor_ptr,
                                       int max_candidates_per_triangle,
                                       int max_collisions_per_triangle) {
    const auto batch_size = bboxes_tensor.size(0);
    const auto num_triangles = (bboxes_tensor.size(1) + 1) / 2;
    const auto num_nodes = bboxes_tensor.size(1);
    const auto max_total_candidates = max_candidates_per_triangle * num_triangles;


//    printf("IMP: collision_impulses_from_bbox_tree\n");
//    return;

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
                auto triangle_mass_float_ptr = triangles_mass_tensor.data<scalar_t>();
                auto triangles_to_check_bool_ptr =   triangles_to_check_tensor.data<bool>();

                for (int bidx = 0; bidx < batch_size; ++bidx) {

                    ////                             cpt-12
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
//

                    int gridSize = (num_triangles + blockSize - 1) / blockSize;
////                             cpt-10
//                     ================================ findPotentialCollisions
                    thrust::fill(collisionIndices.begin(), collisionIndices.end(),
                                 make_long2(-1, -1));

////                             cpt-10.1
                    findPotentialCollisionsPartial<scalar_t><<<gridSize, blockSize>>>(
                            collisionIndices.data().get(),
                            internal_nodes.data().get(),
                            leaf_nodes.data().get(), triangle_ids.data().get(), triangles_to_check_ptr, num_triangles,
                            max_total_candidates, &collision_idx_cnt.data().get()[bidx]);
                    cudaDeviceSynchronize();
////                             cpt-10.2
                    if ((int) collision_idx_cnt[bidx] > max_total_candidates) {
                        printf("Number of candidate collisions exceeds maximal allowed number. Some of the candidate collisions are omitted.\n");
                    }
                    int num_cand_collisions =
                            thrust::reduce(thrust::make_transform_iterator(
                                                   collisionIndices.begin(), is_valid_cnt()),
                                           thrust::make_transform_iterator(
                                                   collisionIndices.end(), is_valid_cnt()));

//                    printf("num_cand_collisions: %d\n", num_cand_collisions);

////                             cpt-9
                    if (num_cand_collisions > 0) {
                        Triangle<scalar_t> *triangles_ptr =
                                (TrianglePtr<scalar_t>) triangle_float_ptr +
                                num_triangles * bidx;
                        Triangle<scalar_t> *triangles_next_ptr =
                                (TrianglePtr<scalar_t>) triangle_next_float_ptr +
                                num_triangles * bidx;
                        vec3<scalar_t> *triangles_mass_ptr =
                                (vec3<scalar_t> *) triangle_mass_float_ptr +
                                num_triangles * bidx;

////                             cpt-8
                        // Keep only the pairs of ids where a bounding box to bounding box
                        // collision was detected.
                        thrust::device_vector <long2> triangle_collisions(num_cand_collisions,
                                                                          make_long2(-1, -1));

                        thrust::copy_if(collisionIndices.begin(), collisionIndices.end(),
                                        triangle_collisions.begin(), is_valid_cnt());

                        cudaCheckError();

                        int tri_grid_size =
                                (num_triangles + blockSize - 1) / blockSize;
//
////                             cpt-7
                        thrust::device_vector <MovingTriangle<scalar_t>> moving_triangles(num_triangles);
                        make_moving_triangles<
                                scalar_t><<<tri_grid_size, blockSize>>>(triangles_ptr, triangles_next_ptr,
                                                                        moving_triangles.data().get(),
                                                                        num_triangles);
                        cudaCheckError();


////                             cpt-6
                        thrust::device_vector<bool> minimal_collision_flags(num_cand_collisions * 15, 0);
////                             cpt-6.1

                        int coll_grid_size = (triangle_collisions.size() + blockSize - 1) / blockSize;
                        dim3 blockGrid(coll_grid_size, 15);

////                             cpt-6.2
                        checkMinimalCollisions<scalar_t><<<blockGrid, blockSize>>>(
                                triangle_collisions.data().get(),
                                moving_triangles.data().get(), minimal_collision_flags.data().get(),
                                num_cand_collisions);
                        cudaCheckError();


////                             cpt-5
                        int num_cminimal_collisions_flags =
                                thrust::reduce(thrust::make_transform_iterator(
                                                       minimal_collision_flags.begin(), is_true()),
                                               thrust::make_transform_iterator(
                                                       minimal_collision_flags.end(), is_true()));
                        thrust::device_vector <long3> minimal_collisions(num_cminimal_collisions_flags,
                                                                         make_long3(-1, -1, -1));
////                             cpt-5.1
                        copyMinimalCollisions<scalar_t><<<blockGrid, blockSize>>>(
                                triangle_collisions.data().get(), minimal_collisions.data().get(),
                                minimal_collision_flags.data().get(),
                                num_cand_collisions, &minimal_collision_idx_cnt.data().get()[bidx]);
                        cudaCheckError();


                        int num_cminimal_collisions = minimal_collision_idx_cnt[bidx];

////                             cpt-4
                        int min_coll_grid_size = (num_cminimal_collisions + blockSize - 1) / blockSize;
                        thrust::device_vector <MovingTetrahedron<scalar_t>> tetrahedrons(num_cminimal_collisions);
////                             cpt-4.1
                        make_tetrahedrons<scalar_t><<<min_coll_grid_size, blockSize>>>(
                                minimal_collisions.data().get(),
                                moving_triangles.data().get(), tetrahedrons.data().get(), num_cminimal_collisions);
                        cudaCheckError();

////                             cpt-3
                        thrust::device_vector <vec3<scalar_t>> cubic_roots_triplets(num_cminimal_collisions,
                                                                                    {-1, -1, -1});
////                             cpt-3.1
                        compute_cubic_roots<scalar_t>(tetrahedrons.data().get(), cubic_roots_triplets.data().get(),
                                                      minimal_collisions.data().get(),
                                                      num_cminimal_collisions);
                        cudaCheckError();

                        int num_valid_roots =
                                thrust::reduce(thrust::make_transform_iterator(
                                                       cubic_roots_triplets.begin(), num_roots<scalar_t>()),
                                               thrust::make_transform_iterator(
                                                       cubic_roots_triplets.end(), num_roots<scalar_t>()));

////                             cpt-2
                        if (num_valid_roots > 0) {

                            thrust::device_vector <MovingTetrahedron<scalar_t>> tetrahedrons_valid(num_valid_roots);
                            thrust::device_vector <long3> minimal_collisions_valid(num_valid_roots,
                                                                                   make_long3(-1, -1, -1));
////                             cpt-2.1
//
//
                            thrust::device_vector <scalar_t> cubic_roots(num_valid_roots, -1);
                            flatten_cubic_roots<scalar_t><<<min_coll_grid_size, blockSize>>>(tetrahedrons.data().get(),
                                                                                             minimal_collisions.data().get(),
                                                                                             cubic_roots_triplets.data().get(),
                                                                                             tetrahedrons_valid.data().get(),
                                                                                             minimal_collisions_valid.data().get(),
                                                                                             cubic_roots.data().get(),
                                                                                             num_cminimal_collisions,
                                                                                             &valid_roots_idx_cnt.data().get()[bidx]);
                            cudaCheckError();
////                             cpt-2.2
                            int valroots_grid_size = (num_valid_roots + blockSize - 1) / blockSize;
                            int num_valid_roots_counter = valid_roots_idx_cnt[bidx];
                            //                            printf("num_valid_roots_counter: %d\n", num_valid_roots_counter);

////                            printf("IMP: num_valid_roots=%d\n", num_valid_roots);
////                            printf("IMP: num_valid_roots_counter=%d\n", num_valid_roots_counter);
////                            printf("IMP: blockSize=%d\n", blockSize);
////                            printf("IMP: valroots_grid_size=%d\n", valroots_grid_size);
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

////                             cpt-1
                            if (num_valid_collisions > 0) {
                                thrust::device_vector <Triangle<scalar_t>> impulses_dv(num_triangles);
                                thrust::device_vector <Triangle<scalar_t>> impulses_dx(num_triangles);
                                thrust::device_vector <int3> impulse_counters(num_triangles, make_int3(0, 0, 0));

                                thrust::device_vector <long3> filtered_collisions(num_valid_collisions,
                                                                                  make_long3(-1, -1, -1));
                                thrust::device_vector <vec4<scalar_t>> mass_quads(num_valid_collisions,
                                                                                  {0., 0., 0., 0.});
                                thrust::device_vector <scalar_t> filtered_cubic_roots(num_valid_collisions);
                                thrust::device_vector <MovingTetrahedron<scalar_t>> filtered_tetrahedrons(
                                        num_valid_collisions);

                                copy_filtered_collisions<scalar_t><<<valroots_grid_size, blockSize>>>(
                                        tetrahedrons_valid.data().get(),
                                        minimal_collisions_valid.data().get(),
                                        cubic_roots.data().get(),
                                        triangles_mass_ptr,
                                        is_valid_collision.data().get(),
                                        filtered_tetrahedrons.data().get(),
                                        filtered_collisions.data().get(),
                                        filtered_cubic_roots.data().get(),
                                        mass_quads.data().get(),
                                        num_valid_roots,
                                        &filtered_minimal_collision_idx_cnt.data().get()[bidx]);
                                //
                                //
                                thrust::device_vector <vec3<scalar_t>> coefficients(num_valid_collisions, {-1, -1, -1});
                                thrust::device_vector <vec3<scalar_t>> normal_directions(num_valid_collisions,
                                                                                         {-1, -1, -1});
                                thrust::device_vector <scalar_t> normal_velocity(num_valid_collisions);
                                thrust::device_vector <scalar_t> normal_distance_end(num_valid_collisions);
                                thrust::device_vector <vec2<scalar_t>> mass_pairs(num_valid_collisions,
                                                                                  {0., 0.});

                                int valid_grid_size = (num_valid_collisions + blockSize - 1) / blockSize;
                                compute_coeficients_and_directions<scalar_t><<<valid_grid_size, blockSize>>>(
                                        filtered_tetrahedrons.data().get(),
                                        filtered_collisions.data().get(),
                                        filtered_cubic_roots.data().get(),
                                        mass_quads.data().get(),
                                        coefficients.data().get(),
                                        normal_directions.data().get(),
                                        normal_velocity.data().get(),
                                        normal_distance_end.data().get(),
                                        mass_pairs.data().get(),
                                        num_valid_collisions);
                                //

                                init_impulses<scalar_t><<<tri_grid_size, blockSize>>>(
                                        impulses_dv.data().get(),
                                        impulses_dx.data().get(),
                                        num_triangles);

//
                                extrapolate_impulses<scalar_t><<<valid_grid_size, blockSize>>>(
                                        filtered_tetrahedrons.data().get(),
                                        filtered_cubic_roots.data().get(),
                                        filtered_collisions.data().get(),
                                        coefficients.data().get(),
                                        normal_directions.data().get(),
                                        normal_velocity.data().get(),
                                        normal_distance_end.data().get(),
                                        mass_pairs.data().get(),
                                        impulses_dv.data().get(),
                                        impulses_dx.data().get(),
                                        impulse_counters.data().get(),
                                        num_valid_collisions);
                                //
                                scalar_t *impulses_dv_ptr = impulses_dv_tensor_ptr->data<scalar_t>();
                                cudaMemcpy(impulses_dv_ptr + bidx * num_triangles * 3 * 3,
                                           (scalar_t *) impulses_dv.data().get(),
                                           num_triangles * 3 * 3 * sizeof(scalar_t),
                                           cudaMemcpyDeviceToDevice);
                                cudaCheckError();

                                scalar_t *impulses_dx_ptr = impulses_dx_tensor_ptr->data<scalar_t>();
                                cudaMemcpy(impulses_dx_ptr + bidx * num_triangles * 3 * 3,
                                           (scalar_t *) impulses_dx.data().get(),
                                           num_triangles * 3 * 3 * sizeof(scalar_t),
                                           cudaMemcpyDeviceToDevice);
                                cudaCheckError();

                                int *impulses_counter_ptr = impulses_counter_tensor_ptr->data<int>();
                                cudaMemcpy(impulses_counter_ptr + bidx * num_triangles * 3,
                                           (int *) impulse_counters.data().get(),
                                           num_triangles * 3 * sizeof(int),
                                           cudaMemcpyDeviceToDevice);
                                cudaCheckError();
                            }
                        }
                    }


                }

            }));

}



void collision_impulses_from_continuous_collisions(at::Tensor collision_tensor, at::Tensor roots_tensor,
                                                   at::Tensor triangles_tensor,
                                                   at::Tensor triangles_next_tensor,
                                                   at::Tensor triangles_mass_tensor,
                                                   at::Tensor *impulses_dv_tensor_ptr,
                                                   at::Tensor *impulses_dx_tensor_ptr,
                                                   at::Tensor *impulses_counter_tensor_ptr) {
    const auto batch_size = triangles_tensor.size(0);
    const auto num_triangles = triangles_tensor.size(1);
    const auto num_collisions = collision_tensor.size(1);

    int blockSize = NUM_THREADS;


    AT_DISPATCH_FLOATING_TYPES(
            triangles_tensor.type(), "bvh_tree_building", ([&] {
//
                auto collisions_long_ptr = collision_tensor.data<long>();
                auto roots_float_ptr = roots_tensor.data<scalar_t>();
                auto triangle_float_ptr = triangles_tensor.data<scalar_t>();
                auto triangle_next_float_ptr = triangles_next_tensor.data<scalar_t>();
                auto triangle_mass_float_ptr = triangles_mass_tensor.data<scalar_t>();
//
                for (int bidx = 0; bidx < batch_size; ++bidx) {


                    Triangle<scalar_t> *triangles_ptr =
                            (TrianglePtr<scalar_t>) triangle_float_ptr +
                            num_triangles * bidx;
                    Triangle<scalar_t> *triangles_next_ptr =
                            (TrianglePtr<scalar_t>) triangle_next_float_ptr +
                            num_triangles * bidx;
                    vec3<scalar_t> *triangles_mass_ptr =
                            (vec3<scalar_t> *) triangle_mass_float_ptr +
                            num_triangles * bidx;
                    long3 *collisions_ptr =
                            (long3 *) collisions_long_ptr +
                            num_collisions * bidx;
                    scalar_t *roots_ptr =
                            (scalar_t *) roots_float_ptr +
                            num_collisions * bidx;


                    int tri_grid_size =
                            (num_triangles + blockSize - 1) / blockSize;
                    thrust::device_vector <MovingTriangle<scalar_t>> moving_triangles(num_triangles);
                    make_moving_triangles<
                            scalar_t><<<tri_grid_size, blockSize>>>(triangles_ptr, triangles_next_ptr,
                                                                    moving_triangles.data().get(),
                                                                    num_triangles);
                    cudaCheckError();
//
//
                    int coll_grid_size = (num_collisions + blockSize - 1) / blockSize;
                    thrust::device_vector <MovingTetrahedron<scalar_t>> tetrahedrons(num_collisions);
                    make_tetrahedrons<scalar_t><<<coll_grid_size, blockSize>>>(
                            collisions_ptr,
                            moving_triangles.data().get(), tetrahedrons.data().get(), num_collisions);
                    cudaCheckError();


                    thrust::device_vector <vec4<scalar_t>> mass_quads(num_collisions,
                                                                      {0., 0., 0., 0.});

                    build_mass_quads<scalar_t><<<coll_grid_size, blockSize>>>(collisions_ptr, triangles_mass_ptr,
                                                                              mass_quads.data().get(), num_collisions);

                    thrust::device_vector <vec3<scalar_t>> coefficients(num_collisions, {-1, -1, -1});
                    thrust::device_vector <vec3<scalar_t>> normal_directions(num_collisions,
                                                                             {-1, -1, -1});
                    thrust::device_vector <scalar_t> normal_velocity(num_collisions);
                    thrust::device_vector <scalar_t> normal_distance_end(num_collisions);
                    thrust::device_vector <vec2<scalar_t>> mass_pairs(num_collisions,
                                                                      {0., 0.});

                    compute_coeficients_and_directions<scalar_t><<<coll_grid_size, blockSize>>>(
                            tetrahedrons.data().get(),
                            collisions_ptr,
                            roots_ptr,
                            mass_quads.data().get(),
                            coefficients.data().get(),
                            normal_directions.data().get(),
                            normal_velocity.data().get(),
                            normal_distance_end.data().get(),
                            mass_pairs.data().get(),
                            num_collisions);
                    cudaCheckError();
//
//
                    thrust::device_vector <Triangle<scalar_t>> impulses_dv(num_triangles);
                    thrust::device_vector <Triangle<scalar_t>> impulses_dx(num_triangles);
                    thrust::device_vector <int3> impulse_counters(num_triangles, make_int3(0, 0, 0));

                    init_impulses<scalar_t><<<tri_grid_size, blockSize>>>(
                            impulses_dv.data().get(),
                            impulses_dx.data().get(),
                            num_triangles);
                    cudaCheckError();

//
                    extrapolate_impulses<scalar_t><<<coll_grid_size, blockSize>>>(
                            tetrahedrons.data().get(),
                            roots_ptr,
                            collisions_ptr,
                            coefficients.data().get(),
                            normal_directions.data().get(),
                            normal_velocity.data().get(),
                            normal_distance_end.data().get(),
                            mass_pairs.data().get(),
                            impulses_dv.data().get(),
                            impulses_dx.data().get(),
                            impulse_counters.data().get(),
                            num_collisions);

                    //
                    cudaCheckError();
                    scalar_t *impulses_dv_ptr = impulses_dv_tensor_ptr->data<scalar_t>();
                    cudaMemcpy(impulses_dv_ptr + bidx * num_triangles * 3 * 3,
                               (scalar_t *) impulses_dv.data().get(),
                               num_triangles * 3 * 3 * sizeof(scalar_t),
                               cudaMemcpyDeviceToDevice);
                    cudaCheckError();

                    scalar_t *impulses_dx_ptr = impulses_dx_tensor_ptr->data<scalar_t>();
                    cudaMemcpy(impulses_dx_ptr + bidx * num_triangles * 3 * 3,
                               (scalar_t *) impulses_dx.data().get(),
                               num_triangles * 3 * 3 * sizeof(scalar_t),
                               cudaMemcpyDeviceToDevice);
                    cudaCheckError();

                    int *impulses_counter_ptr = impulses_counter_tensor_ptr->data<int>();
                    cudaMemcpy(impulses_counter_ptr + bidx * num_triangles * 3,
                               (int *) impulse_counters.data().get(),
                               num_triangles * 3 * sizeof(int),
                               cudaMemcpyDeviceToDevice);
                    cudaCheckError();


                }

            }));

}
