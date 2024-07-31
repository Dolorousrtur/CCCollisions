#pragma once

#import "flags.h"
#import "utils.cu"
#import "bvh.cu"
#import "figures.cu"
#import "build_bvh.cu"
#import "collisions_static.cu"
#import "collisions_continuous.cu"

template<typename T>
__device__ bool
check_minimal_bbox_overlap(long2 *triangle_collisions, Triangle<T> *moving_triangles,
                           int from_idx, int collision_id, T threshold) {


    long2 *coll = triangle_collisions + from_idx;
    Triangle<T> *triangle0 = moving_triangles + coll->x;
    Triangle<T> *triangle1 = moving_triangles + coll->y;



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

    bool is_overlap = checkOverlap<T>(bbox0, bbox1, threshold);

//    printf("coll.x: %d, coll_y: %d\n"
//           "collision_id: %d\n"
//           "is_overlap: %d\n", (int)coll->x, (int)coll->y, (int)collision_id, (int)is_overlap);

    return is_overlap;
}

template<typename T>
__global__
void
checkMinimalCollisions(long2 *triangle_collisions, Triangle<T> *triangles, bool *minimal_collisions_flags,
                       int num_collisions, T threshold) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;

    int collision_id = blockIdx.y;

    bool overlab = false;
    bool overlap = check_minimal_bbox_overlap<T>(triangle_collisions, triangles, idx, collision_id, threshold);
    if (not overlap)
        return;

    int to_idx = idx * 15 + collision_id;
    minimal_collisions_flags[to_idx] = true;

}

template<typename T>
__device__ void
make_tetrahedron(long3 *minimal_collisions_tids, Triangle<T> *moving_triangles,
                 Tetrahedron<T> *tetrahedrons,
                 int idx) {

    long3 *coll = minimal_collisions_tids + idx;
    Triangle<T> *triangle0 = moving_triangles + coll->x;
    Triangle<T> *triangle1 = moving_triangles + coll->y;

    switch (coll->z) {
        case 0 :
//            minimal_collisions_vids[to_idx] = make_long4(0, 1, 0, 1);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v0, triangle0->v1, triangle1->v0, triangle1->v1);
            break;
        case 1:
//            minimal_collisions_vids[to_idx] = make_long4(0, 1, 0, 2);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v0, triangle0->v1, triangle1->v0, triangle1->v2);
            break;
        case 2:
//            minimal_collisions_vids[to_idx] = make_long4(0, 1, 1, 2);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v0, triangle0->v1, triangle1->v1, triangle1->v2);
            break;
        case 3 :
//            minimal_collisions_vids[to_idx] = make_long4(0, 2, 0, 1);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v0, triangle0->v2, triangle1->v0, triangle1->v1);
            break;
        case 4:
//            minimal_collisions_vids[to_idx] = make_long4(0, 2, 0, 2);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v0, triangle0->v2, triangle1->v0, triangle1->v2);
            break;
        case 5:
//            minimal_collisions_vids[to_idx] = make_long4(0, 2, 1, 2);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v0, triangle0->v2, triangle1->v1, triangle1->v2);
            break;
        case 6 :
//            minimal_collisions_vids[to_idx] = make_long4(1, 2, 0, 1);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v1, triangle0->v2, triangle1->v0, triangle1->v1);
            break;
        case 7:
//            minimal_collisions_vids[to_idx] = make_long4(1, 2, 0, 2);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v1, triangle0->v2, triangle1->v0, triangle1->v2);
            break;
        case 8:
//            minimal_collisions_vids[to_idx] = make_long4(1, 2, 1, 2);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v1, triangle0->v2, triangle1->v1, triangle1->v2);
            break;
        case 9 :
//            minimal_collisions_vids[to_idx] = make_long4(0, 1, 2, 0);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v0, triangle0->v1, triangle0->v2, triangle1->v0);
            break;
        case 10:
//            minimal_collisions_vids[to_idx] = make_long4(0, 1, 2, 1);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v0, triangle0->v1, triangle0->v2, triangle1->v1);
            break;
        case 11:
//            minimal_collisions_vids[to_idx] = make_long4(0, 1, 2, 2);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v0, triangle0->v1, triangle0->v2, triangle1->v2);
            break;
        case 12 :
//            minimal_collisions_vids[to_idx] = make_long4(0, 0, 1, 2);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v0, triangle1->v0, triangle1->v1, triangle1->v2);
            break;
        case 13:
//            minimal_collisions_vids[to_idx] = make_long4(1, 0, 1, 2);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v1, triangle1->v0, triangle1->v1, triangle1->v2);
            break;
        case 14:
//            minimal_collisions_vids[to_idx] = make_long4(2, 0, 1, 2);
            tetrahedrons[idx] = Tetrahedron<T>(triangle0->v2, triangle1->v0, triangle1->v1, triangle1->v2);
            break;
    }

}

template<typename T>
__global__
void make_tetrahedrons(long3 *minimal_collisions_tids, Triangle<T> *triangles,
                       Tetrahedron<T> *tetrahedrons, int num_collisions) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;

    make_tetrahedron<T>(minimal_collisions_tids, triangles, tetrahedrons, idx);
}

template<typename T>
__device__ void distance_point_to_segment(const vec3<T>& p, const vec3<T>& a, const vec3<T>& b, vec3<T>* dist_and_coeffs, bool is_set_s, T second_coef) {
    vec3<T> ab = b - a;
    vec3<T> ap = p - a;
    T t = dot(ap, ab) / dot(ab, ab);
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    vec3<T> c = {a.x + t*ab.x, a.y + t*ab.y, a.z + t*ab.z};

    T new_dist = sqrt(vec_sq_diff<T>(p, c));
    T old_dist = dist_and_coeffs->x;

    if (new_dist < old_dist) {
        dist_and_coeffs->x = new_dist;
        if (is_set_s) {
            dist_and_coeffs->y = t;
            dist_and_coeffs->z = second_coef;
        } else {
            dist_and_coeffs->z = t;
            dist_and_coeffs->y = second_coef;
        }
    }

}

template<typename T>
__device__ void compute_distance_parallel_edges(vec3<T> &v0, vec3<T> &v1, vec3<T> &v2, vec3<T> &v3, vec3<T> *dist_and_coeffs) {
    distance_point_to_segment<T>(v0, v2, v3, dist_and_coeffs, false, 0.);
    distance_point_to_segment<T>(v1, v2, v3, dist_and_coeffs, false, 1.);
    distance_point_to_segment<T>(v2, v0, v1, dist_and_coeffs, true, 0.);
    distance_point_to_segment<T>(v3, v0, v1, dist_and_coeffs, true, 1.);

}

template<typename T>
__device__ void
compute_distance_edge_edge(vec3<T> &v0, vec3<T> &v1, vec3<T> &v2, vec3<T> &v3, T *distances, vec3<T> *coefficients,
                           int idx) {

    vec3<T> da = v1 - v0;
    vec3<T> db = v3 - v2;
    vec3<T> dc = v2 - v0;

    T s = dot(cross(dc, db), cross(da, db)) / norm2(cross(da, db));
    T t = dot(cross(dc, da), cross(da, db)) / norm2(cross(da, db));

    if (isnan(s) or isnan(t)) {
        vec3<T> dist_and_coeffs = {FLT_MAX, -1., -1.};

        compute_distance_parallel_edges<T>(v0, v1, v2, v3, &dist_and_coeffs);

        distances[idx] = dist_and_coeffs.x;
        s = dist_and_coeffs.y;
        t = dist_and_coeffs.z;
    }

    bool s_valid = (s >= 0 and s <= 1);
    bool t_valid = (t >= 0 and t <= 1);

//    printf("v0: (%f, %f, %f)\n"
//           "v1: (%f, %f, %f)\n"
//           "v2: (%f, %f, %f)\n"
//           "v3: (%f, %f, %f)\n"
//           "s: %f, t: %f\n",
//           v0.x, v0.y, v0.z,
//           v1.x, v1.y, v1.z,
//           v2.x, v2.y, v2.z,
//           v3.x, v3.y, v3.z,
//           s, t);

    if (not s_valid or not t_valid) {
        distances[idx] = -1;
        coefficients[idx].x = -1;
        coefficients[idx].y = -1;
    } else {
        distances[idx] = sqrt(vec_sq_diff<T>(v0 + da * s, v2 + db * t));
        coefficients[idx].x = s;
        coefficients[idx].y = t;
    }
    return;

//    if (not s_valid or not t_valid) {
//        T clamp1 = 0;
//        T clamp2 = 0;
//        if (not s_valid) {
//            if (s < 0) {
//                clamp1 = -s;
//                s = 0;
//            } else {
//                clamp1 = s - 1;
//                s = 1.;
//            }
//        }
//
//        if (not t_valid) {
//            if (t < 0) {
//                clamp2 = -t;
//                t = 0;
//            } else {
//                clamp2 = t - 1;
//                t = 1;
//            }
//        }
//
//        if (clamp1 > clamp2) {
//            t = dot(da * s - dc, db) / norm2(db);
//            if (t > 1)
//                t = 1;
//            else if (t < 0)
//                t = 0;
//        } else {
//            s = dot(db * t + dc, da) / norm2(da);
//            if (s > 1)
//                s = 1;
//            else if (s < 0)
//                s = 0;
//        }
//
//
//    }


//    distances[idx] = sqrt(vec_sq_diff<T>(v0 + da * s, v2 + db * t));
//    coefficients[idx].x = s;
//    coefficients[idx].y = t;


}

template<typename T>
__device__ void
compute_distance_triangle_node(vec3<T> &v0, vec3<T> &v1, vec3<T> &v2, vec3<T> &vt, T *distances, vec3<T> *coefficients,
                               int idx) {
    vec3<T> e1 = v0 - v1;
    vec3<T> e2 = v2 - v1;
    vec3<T> ep = vt - v1;
    T d00 = dot(e1, e1);
    T d01 = dot(e1, e2);
    T d11 = dot(e2, e2);
    T d20 = dot(ep, e1);
    T d21 = dot(ep, e2);

//    printf("d00: %g\td01: %g\td11: %g\td20: %g\td21: %g\n", d00, d01, d11, d20, d21);

    T denom = d00 * d11 - d01 * d01;
    T u = (d11 * d20 - d01 * d21) / denom;
    T w = (d00 * d21 - d01 * d20) / denom;
    T v = 1 - w - u;

//    printf("u: %g\tv: %g\tw: %g\tdenom: %g\n", u, v, w, denom);

    bool v_valid = v >= 0 and v <= 1;
    bool w_valid = w >= 0 and w <= 1;
    bool u_valid = u >= 0 and u <= 1;
    bool all_valid = v_valid and w_valid and u_valid;

    if (not all_valid) {

        distances[idx] = -1;

        coefficients[idx].x = -1;
        coefficients[idx].y = -1;
        coefficients[idx].z = -1;
    } else {
        vec3<T> p_on_tri = v0 * u + v1 * v + v2 * w;
        distances[idx] = sqrt(vec_sq_diff<T>(vt, p_on_tri));

        coefficients[idx].x = u;
        coefficients[idx].y = v;
        coefficients[idx].z = w;
    }
    return;

//    u: 1.54251	v: -1.19506	w: 0.652547

//    if (not all_valid) {
//        if (u <= 0) {
//            if (v <= 0) {
//                u = 0;
//                v = 0;
//                w = 1;
//            } else if (w <= 0) {
//                u = 0;
//                v = 1;
//                w = 0;
//            } else {
//                u = 0;
//                w = dot(ep, e2) / dot(e2, e2);
//                if (w < 0)
//                    w = 0;
//                else if (w > 1)
//                    w = 1;
//                v = 1 - w;
//            }
//        } else if (v <= 0) {
//            if (w <= 0) {
//                u = 1;
//                v = 0;
//                w = 0;
//            } else {
//                v = 0;
//                w = dot(vt - v0, v2 - v0) / norm2(v2 - v0);
//                if (w < 0)
//                    w = 0;
//                else if (w > 1)
//                    w = 1;
//                u = 1 - w;
//            }
//        } else {
//            w = 0;
//            v = dot(vt - v0, e1 * -1) / norm2(e1);
//            if (v < 0)
//                v = 0;
//            else if (v > 1)
//                v = 1;
//            u = 1 - v;
//        }
//    }

//    vec3<T> p_on_tri = v0 * u + v1 * v + v2 * w;
//    distances[idx] = sqrt(vec_sq_diff<T>(vt, p_on_tri));
//
//    coefficients[idx].x = u;
//    coefficients[idx].y = v;
//    coefficients[idx].z = w;
//
//    return;

}

template<typename T>
__global__
void
computeDistances(Tetrahedron<T> *tetrahedrons, long3 *minimal_collisions_tids,
                 T *distances, vec3<T> *coefficients, bool *is_close, int num_collisions, T threshold) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;

    long3 *tcoll = minimal_collisions_tids + idx;
    Tetrahedron<T> *thz = tetrahedrons + idx;

//    if (tcoll->x != 1 or tcoll->y != 2 or tcoll->z != 12) return;
//
    if (tcoll->z < 9) {
        compute_distance_edge_edge<T>(thz->v0, thz->v1, thz->v2, thz->v3, distances, coefficients, idx);
//        printf("idx: %d\tt0: %d\tt1: %d\ttype: %d\td: %g\ts: %g\tt: %g\n", idx, (int) tcoll->x, (int) tcoll->y,
//               (int) tcoll->z, distances[idx], coefficients[idx].x, coefficients[idx].y);
    } else if (tcoll->z < 12) {
        compute_distance_triangle_node<T>(thz->v0, thz->v1, thz->v2, thz->v3, distances, coefficients, idx);
//        printf("idx: %d\tt0: %d\tt1: %d\ttype: %d\td: %g\tu: %g\tv: %g\tw: %g\n", idx, (int) tcoll->x, (int) tcoll->y,
//               (int) tcoll->z, distances[idx], coefficients[idx].x, coefficients[idx].y, coefficients[idx].z);
    } else {
        compute_distance_triangle_node<T>(thz->v1, thz->v2, thz->v3, thz->v0, distances, coefficients, idx);

        if (tcoll->x == 82 and tcoll->y == 211 and tcoll->z == 14) {
            printf("idx: %d\tt0: %d\tt1: %d\ttype: %d\td: %g\tu: %g\tv: %g\tw: %g\n", idx, (int) tcoll->x, (int) tcoll->y,
                   (int) tcoll->z, distances[idx], coefficients[idx].x, coefficients[idx].y, coefficients[idx].z);

        }
//        printf("idx: %d\tt0: %d\tt1: %d\ttype: %d\td: %g\tu: %g\tv: %g\tw: %g\n", idx, (int) tcoll->x, (int) tcoll->y,
//               (int) tcoll->z, distances[idx], coefficients[idx].x, coefficients[idx].y, coefficients[idx].z);
    }

    is_close[idx] = distances[idx] <= threshold and distances[idx] >= 0;


}

template<typename T>
__global__
void
collectClose(Tetrahedron<T> *tetrahedrons, long3 *minimal_collisions, T *distances, vec3<T> *coefficients,
             bool *is_close,
             long3 *minimal_collisions_close, T *distances_close, vec3<T> *coefficients_close, vec3<T> *contact_points,
             int num_collisions, int *counter) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_collisions) return;
    if (not is_close[idx]) return;

    int to_idx = atomicAdd(counter, 1);
    minimal_collisions_close[to_idx] = minimal_collisions[idx];
    distances_close[to_idx] = distances[idx];
    coefficients_close[to_idx] = coefficients[idx];

    long3 *coll = minimal_collisions + idx;
    Tetrahedron<T> *tetrahedron = tetrahedrons + idx;
    vec3<T> *coeffs = coefficients + idx;


    if (coll->z < 9) {
        contact_points[2 * to_idx] = tetrahedron->v0 + (tetrahedron->v1 - tetrahedron->v0) * coeffs->x;
        contact_points[2 * to_idx + 1] = tetrahedron->v2 + (tetrahedron->v3 - tetrahedron->v2) * coeffs->y;
    } else if (coll->z < 12) {
        contact_points[2 * to_idx] =
                tetrahedron->v0 * coeffs->x + tetrahedron->v1 * coeffs->y + tetrahedron->v2 * coeffs->z;
        contact_points[2 * to_idx + 1] = tetrahedron->v3;
    } else {
        contact_points[2 * to_idx] = tetrahedron->v0;
        contact_points[2 * to_idx + 1] =
                tetrahedron->v1 * coeffs->x + tetrahedron->v2 * coeffs->y + tetrahedron->v3 * coeffs->z;
    }


}

void find_proximity_from_bbox_tree(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                   at::Tensor *collision_tensor_ptr,
                                   int max_candidates_per_triangle, float threshold) {
    const auto batch_size = bboxes_tensor.size(0);
    const auto num_triangles = (bboxes_tensor.size(1) + 1) / 2;
    const auto num_nodes = bboxes_tensor.size(1);
    const auto max_total_candidates = max_candidates_per_triangle * num_triangles;

    // list of pairs of triangle indices
    // each pair represents a triangle-triangle penetration
    // max number of collisions is `num_triangles * max_candidates_per_triangle`; if trying to add more collisions -> error
    thrust::device_vector <long2> collisionIndices(num_triangles * max_candidates_per_triangle);

    thrust::device_vector<int> triangle_ids(num_triangles);
    thrust::sequence(triangle_ids.begin(), triangle_ids.end());


    // int *counter;
    // number of collisions in each sample of the batch
    thrust::device_vector<int> collision_idx_cnt(batch_size);
    thrust::fill(collision_idx_cnt.begin(), collision_idx_cnt.end(), 0);

    thrust::device_vector<int> minimal_collision_idx_cnt(batch_size);
    thrust::fill(minimal_collision_idx_cnt.begin(), minimal_collision_idx_cnt.end(), 0);

    thrust::device_vector<int> close_idx_cnt(batch_size);
    thrust::fill(close_idx_cnt.begin(), close_idx_cnt.end(), 0);

    AT_DISPATCH_FLOATING_TYPES(
            bboxes_tensor.type(), "bvh_tree_building", ([&] {
                thrust::device_vector <BVHNode<scalar_t>> leaf_nodes(num_triangles);
                thrust::device_vector <BVHNode<scalar_t>> internal_nodes(num_triangles - 1);

                auto bboxes_float_ptr = bboxes_tensor.data<scalar_t>();
                auto tree_long_ptr = tree_tensor.data<long>();
                auto triangle_float_ptr = triangles_tensor.data<scalar_t>();

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
//
//                    int tempgridSize = (2 * num_triangles - 1 + blockSize - 1) / blockSize;
//                    print_node<scalar_t><<<tempgridSize, blockSize>>>(internal_nodes.data().get(),
//                                                                      leaf_nodes.data().get(), num_triangles);


//                     ================================ findPotentialCollisions
                    findPotentialCollisions<scalar_t><<<gridSize, blockSize>>>(
                            collisionIndices.data().get(),
                            internal_nodes.data().get(),
                            leaf_nodes.data().get(), triangle_ids.data().get(), num_triangles,
                            max_total_candidates, &collision_idx_cnt.data().get()[bidx], threshold);
                    if ((int) collision_idx_cnt[bidx] > max_total_candidates) {
                        printf("Number of candidate collisions exceeds maximal allowed number. Some of the candidate collisions are omitted.\n");
                    }
                    cudaDeviceSynchronize();


                    int num_cand_collisions =
                            thrust::reduce(thrust::make_transform_iterator(
                                                   collisionIndices.begin(), is_valid_cnt()),
                                           thrust::make_transform_iterator(
                                                   collisionIndices.end(), is_valid_cnt()));

//                    std::cout << "\nnum_cand_collisions " << num_cand_collisions << "\n";
                    if (num_cand_collisions > 0) {

                        Triangle<scalar_t> *triangles_ptr =
                                (TrianglePtr<scalar_t>) triangle_float_ptr +
                                num_triangles * bidx;


                        // Keep only the pairs of ids where a bounding box to bounding box
                        // collision was detected.
                        thrust::device_vector <long2> triangle_collisions(num_cand_collisions,
                                                                          make_long2(-1, -1));
                        thrust::copy_if(collisionIndices.begin(), collisionIndices.end(),
                                        triangle_collisions.begin(), is_valid_cnt());

                        cudaCheckError();
//
                        thrust::device_vector<bool> minimal_collision_flags(num_cand_collisions * 15, 0);

                        int coll_grid_size = (triangle_collisions.size() + blockSize - 1) / blockSize;
                        dim3 blockGrid(coll_grid_size, 15);
                        checkMinimalCollisions<scalar_t><<<blockGrid, blockSize>>>(
                                triangle_collisions.data().get(),
                                triangles_ptr, minimal_collision_flags.data().get(),
                                num_cand_collisions, threshold);
                        cudaCheckError();

                        int num_cminimal_collisions_flags =
                                thrust::reduce(thrust::make_transform_iterator(
                                                       minimal_collision_flags.begin(), is_true()),
                                               thrust::make_transform_iterator(
                                                       minimal_collision_flags.end(), is_true()));
                        thrust::device_vector <long3> minimal_collisions(num_cminimal_collisions_flags,
                                                                         make_long3(-1, -1, -1));

                        copyMinimalCollisions<scalar_t><<<blockGrid, blockSize>>>(
                                triangle_collisions.data().get(), minimal_collisions.data().get(),
                                minimal_collision_flags.data().get(),
                                num_cand_collisions, &minimal_collision_idx_cnt.data().get()[bidx]);
                        cudaCheckError();

                        int num_cminimal_collisions = minimal_collision_idx_cnt[bidx];
//                        printf("num_cminimal_collisions: %d\n", num_cminimal_collisions);

                        int min_coll_grid_size = (num_cminimal_collisions + blockSize - 1) / blockSize;
                        thrust::device_vector <Tetrahedron<scalar_t>> tetrahedrons(num_cminimal_collisions);
                        make_tetrahedrons<scalar_t><<<min_coll_grid_size, blockSize>>>(
                                minimal_collisions.data().get(),
                                triangles_ptr, tetrahedrons.data().get(), num_cminimal_collisions);
                        cudaCheckError();

                        thrust::device_vector <scalar_t> distances(num_cminimal_collisions, -1);
                        thrust::device_vector <vec3<scalar_t>> coefficients(num_cminimal_collisions, {-1, -1, -1});
                        thrust::device_vector<bool> is_close(num_cminimal_collisions, 0);
                        computeDistances<scalar_t><<<min_coll_grid_size, blockSize>>>(tetrahedrons.data().get(),
                                                                                      minimal_collisions.data().get(),
                                                                                      distances.data().get(),
                                                                                      coefficients.data().get(),
                                                                                      is_close.data().get(),
                                                                                      num_cminimal_collisions,
                                                                                      threshold);
                        cudaCheckError();

                        int num_close =
                                thrust::reduce(thrust::make_transform_iterator(
                                                       is_close.begin(), is_true()),
                                               thrust::make_transform_iterator(
                                                       is_close.end(), is_true()));
//                        printf("num_close: %d\n", num_close);


                        thrust::device_vector <scalar_t> distances_close(num_close);
                        thrust::device_vector <vec3<scalar_t>> coefficients_close(num_close);
                        thrust::device_vector <vec3<scalar_t>> contact_points(num_close * 2);
                        thrust::device_vector <long3> minimal_collisions_close(num_close);

                        collectClose<scalar_t><<<min_coll_grid_size, blockSize>>>(tetrahedrons.data().get(),
                                                                                  minimal_collisions.data().get(),
                                                                                  distances.data().get(),
                                                                                  coefficients.data().get(),
                                                                                  is_close.data().get(),
                                                                                  minimal_collisions_close.data().get(),
                                                                                  distances_close.data().get(),
                                                                                  coefficients_close.data().get(),
                                                                                  contact_points.data().get(),
                                                                                  num_cminimal_collisions,
                                                                                  &close_idx_cnt.data().get()[bidx]
                        );
                        cudaCheckError();


////
                        long *dev_ptr = collision_tensor_ptr->data<long>();
                        cudaMemcpy(dev_ptr + bidx * num_triangles * max_candidates_per_triangle * 3,
                                   (long *) minimal_collisions_close.data().get(),
                                   3 * minimal_collisions_close.size() * sizeof(long),
                                   cudaMemcpyDeviceToDevice);
                        cudaCheckError();

                    }


                }

            }));

}