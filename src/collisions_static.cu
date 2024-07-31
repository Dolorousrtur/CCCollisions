#pragma once

#import "flags.h"
#import "utils.cu"
#import "bvh.cu"
#import "figures.cu"
#import "build_bvh.cu"
#import "collisions_static.cu"

template<typename T>
__device__ int traverseBVH(long2 *collisionIndices, BVHNodePtr<T> root,
                           const AABB<T> &queryAABB, int queryObjectIdx,
                           BVHNodePtr<T> leaf, int max_total_collisions,
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

        if (overlapR && childR->isLeaf()) {
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

template<typename T>
__global__ void findPotentialCollisions(long2 *collisionIndices,
                                        BVHNodePtr<T> root,
                                        BVHNodePtr<T> leaves, int *triangle_ids,
                                        int num_primitives,
                                        int max_total_collisions, int *counter, T threshold = 0, bool collision_ordering = true) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_primitives) {
        BVHNodePtr<T> leaf = leaves + idx;
        int triangle_id = triangle_ids[idx];
        int num_collisions =
                traverseBVH<T>(collisionIndices, root, leaf->bbox, triangle_id,
                               leaf, max_total_collisions, counter, threshold, collision_ordering);
    }
    return;
}


template<typename T>
__global__ void checkTriangleIntersections(long2 *collisions,
                                           Triangle<T> *triangles,
                                           long2 *out_collisions,
                                           int num_cand_collisions,
                                           T threshold, int *counter,
                                           int allow_shared_vertices = 0) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_cand_collisions) {
        int first_tri_idx = collisions[idx].x;
        int second_tri_idx = collisions[idx].y;


        Triangle<T> tri1 = triangles[first_tri_idx];
        Triangle<T> tri2 = triangles[second_tri_idx];
        bool do_collide_isect = TriangleTriangleIsectSepAxis<T>(tri1, tri2, threshold);
        int num_shared_verts = countSharedVerts<T>(tri1, tri2);


        // bool shareV = num_shared_verts > allow_shared_vertices;
        bool shareV = num_shared_verts > allow_shared_vertices;

        bool do_collide = do_collide_isect and not shareV;
//        printf("idx: %d\t first_tri_idx %d\tsecond_tri_idx %d\n"
//               "do_collide: %d\n"
//               "do_collide_isect: %d\n"
//               "shareV: %d\n", idx, first_tri_idx, second_tri_idx, do_collide, do_collide_isect, shareV);


        if (do_collide) {
            int to_idx = atomicAdd(counter, 1);
            out_collisions[to_idx] = make_long2(first_tri_idx, second_tri_idx);
        }

    }
    return;
}


// template<typename T>
// __global__ void checkTriangleIntersections(long2 *collisions,
//                                            Triangle<T> *triangles,
//                                            int num_cand_collisions,
//                                            int num_triangles,
//                                            T threshold) {
//     int idx = threadIdx.x + blockDim.x * blockIdx.x;
//     if (idx < num_cand_collisions) {
//         int first_tri_idx = collisions[idx].x;
//         int second_tri_idx = collisions[idx].y;


//         Triangle<T> tri1 = triangles[first_tri_idx];
//         Triangle<T> tri2 = triangles[second_tri_idx];
//         bool do_collide_isect = TriangleTriangleIsectSepAxis<T>(tri1, tri2, threshold);
//         bool shareV = shareVertex<T>(tri1, tri2);
//         bool do_collide = do_collide_isect and not shareV;
// //        printf("idx: %d\t first_tri_idx %d\tsecond_tri_idx %d\n"
// //               "do_collide: %d\n"
// //               "do_collide_isect: %d\n"
// //               "shareV: %d\n", idx, first_tri_idx, second_tri_idx, do_collide, do_collide_isect, shareV);
//         if (do_collide) {
//             collisions[idx] = make_long2(first_tri_idx, second_tri_idx);
//         } 
//     }
//     return;
// }




template<typename T>
__device__ int
get_idx_by_node(BVHNodePtr<T> internal_nodes, BVHNodePtr<T> leaf_nodes, int num_triangles, BVHNodePtr<T> node) {
    if (not node)
        return -1;

    if (node->isLeaf())
        return node - leaf_nodes + num_triangles - 1;
    else
        return node - internal_nodes;
}


void find_collisions_from_bbox_tree(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                    at::Tensor *collision_tensor_ptr, float threshold,
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


    thrust::device_vector<int> valid_collision_idx_cnt(batch_size);
    thrust::fill(valid_collision_idx_cnt.begin(), valid_collision_idx_cnt.end(), 0);


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


//                    std::cout << "\nnum_triangles counter " << num_triangles << "\n";
//                     ================================ findPotentialCollisions
                    findPotentialCollisions<scalar_t><<<gridSize, blockSize>>>(
                            collisionIndices.data().get(),
                            internal_nodes.data().get(),
                            leaf_nodes.data().get(), triangle_ids.data().get(), num_triangles,
                            max_total_candidates, &collision_idx_cnt.data().get()[bidx], threshold);


//                    std::cout << "\nnum_cand_collisions counter " << (int)collision_idx_cnt[bidx] << "\n";
//                    std::cout << "\nmax_total_candidates counter " << (int)max_total_candidates << "\n";
                    if ((int)collision_idx_cnt[bidx] > max_total_candidates) {
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
                        thrust::device_vector <long2> collisions(num_cand_collisions,
                                                                 make_long2(-1, -1));
                        thrust::copy_if(collisionIndices.begin(), collisionIndices.end(),
                                        collisions.begin(), is_valid_cnt());

                        
                        thrust::device_vector <long2> valid_collisions(num_cand_collisions,
                                                                 make_long2(-1, -1));

                        cudaCheckError();

                        int tri_grid_size =
                                (collisions.size() + blockSize - 1) / blockSize;

//                        print_candidates<scalar_t><<<tri_grid_size, blockSize>>>(collisions.data().get(), collisions.size());
                        checkTriangleIntersections<scalar_t><<<tri_grid_size, blockSize>>>( // checkTriangleIntersections
                                collisions.data().get(), triangles_ptr, valid_collisions.data().get(), collisions.size(),  threshold,
                                &valid_collision_idx_cnt.data().get()[bidx]);


                        // checkTriangleIntersections<scalar_t><<<tri_grid_size, blockSize>>>( // checkTriangleIntersections
                        // collisions.data().get(), triangles_ptr, collisions.size(),
                        // num_triangles, threshold);



                        cudaCheckError();

//
                        long *dev_ptr = collision_tensor_ptr->data<long>();
                        cudaMemcpy(dev_ptr + bidx * num_triangles * max_collisions * 2,
                                   (long *) valid_collisions.data().get(),
                                   2 * valid_collisions.size() * sizeof(long),
                                   cudaMemcpyDeviceToDevice);
                        cudaCheckError();

                    }


                }

            }));

}


template<typename T>
__device__
void
addTECollsion(Triangle<T>* triangle, vec3<T>* v0, vec3<T>* v1,
                long2* in_collisions, int from_idx, int collision_id,
                 long4* out_collisions, T* edge_coefs, int* counter) {
    /*
    Check if edge v0-v1 intersects with the triangle
    */

    // printf("collision_id: %d", collision_id);

    vec3<T> edge1 = triangle->v1 - triangle->v0;
    vec3<T> edge2 = triangle->v2 - triangle->v0;
    vec3<T> triangleNormal = normalize(cross(edge1, edge2));


    bool v0_shared = hasVertex<T>(*triangle, *v0);
    bool v1_shared = hasVertex<T>(*triangle, *v1);
    bool is_loop_node = v0_shared or v1_shared;

    // printf("is_loop_node: %d\n"
    // "traingle.v0: (%g, %g, %g)\n"
    // "traingle.v1: (%g, %g, %g)\n"
    // "traingle.v2: (%g, %g, %g)\n"
    // "v0: (%g, %g, %g)\n"
    // "v1: (%g, %g, %g)\n"
    // "x_same: %d\t y_same: %d\t z_same: %d\n"
    // "all same: %d\n"
    // "\n\n",
    // is_loop_node,
    // triangle->v0.x, triangle->v0.y, triangle->v0.z,
    // triangle->v1.x, triangle->v1.y, triangle->v1.z,
    // triangle->v2.x, triangle->v2.y, triangle->v2.z,
    // v0->x, v0->y, v0->z,
    // v1->x, v1->y, v1->z,
    // x_same, y_same, z_same,
    // same);

    if (is_loop_node) {
        if (collision_id > 2) {
            return;
        }

        int shared_node;
        float t;
        if (v0_shared) {
            shared_node = 0;
            t = 0.;
        } else if (v1_shared) {
            // shared_node = 1;
            // t = 1.;
            return;
        }

        int to_id = atomicAdd(counter, 1);
        long2 from_collision = in_collisions[from_idx];
        out_collisions[to_id] = make_long4(from_collision.x, from_collision.y, collision_id, shared_node);
        edge_coefs[to_id] = t;
        return;
        
    }


    vec3<T> dir = *v1 - *v0;
    T en_dot = dot(dir, triangleNormal);

    if (en_dot == 0.0) {
        // Edge and triangle are parallel
        // printf("en_dot == 0.0\n");
        return;
    }

    T d = dot(triangleNormal, triangle->v0);
    T t = (d - dot(triangleNormal, *v0)) / en_dot;

    if (t < 0.0 or t > 1.0) {
        // Edge does not intersect with the triangle
        // printf("t < 0.0 or t > 1.0\n");
        return;
    }

    // vec3<T> intersection = *v0 + t * dir;
    vec3<T> intersection = *v0 + dir * t;


    vec3<T> ep = intersection - triangle->v0;
    T d00 = dot(edge1, edge1);
    T d01 = dot(edge1, edge2);
    T d11 = dot(edge2, edge2);
    T d20 = dot(ep, edge1);
    T d21 = dot(ep, edge2);

    T denom = d00 * d11 - d01 * d01;
    if (denom < 1e-15) {
        // Intersection is outside the triangle
        // printf("denom < 1e-15\n");
        return;
    }

    T v = (d11 * d20 - d01 * d21) / denom;
    T w = (d00 * d21 - d01 * d20) / denom;
    T u = 1.0 - v - w;

    if (u < 0.0 or v < 0.0 or w < 0.0) {
        // Intersection is outside the triangle
        // printf("u < 0.0 or v < 0.0 or w < 0.0\n");
        return;
    }

    int to_id = atomicAdd(counter, 1);

    long2 from_collision = in_collisions[from_idx];
    out_collisions[to_id] = make_long4(from_collision.x, from_collision.y, collision_id, -1);
    edge_coefs[to_id] = t;

}



template<typename T>
__global__
void
collectTEIntersections(long2 *collisions, Triangle<T> *triangles, int num_cand_collisions,
                       long4 *minimal_collisions, T *edge_coefs, int *counter) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    

    if (idx < num_cand_collisions) {
        int collision_id = blockIdx.y;

        int first_tri_idx = collisions[idx].x;
        int second_tri_idx = collisions[idx].y;

        Triangle<T> tri1;
        Triangle<T> tri2;

        if (collision_id < 3) {
            tri1 = triangles[first_tri_idx];
            tri2 = triangles[second_tri_idx];
        } else {
            tri1 = triangles[second_tri_idx];
            tri2 = triangles[first_tri_idx];
        }



        int edge_id = collision_id % 3;

        vec3<T> v1;
        vec3<T> v2;

        switch (edge_id){
            case 0:
                v1 = tri1.v0;
                v2 = tri1.v1;
                break;
            case 1:
                v1 = tri1.v1;
                v2 = tri1.v2;
                break;
            case 2:
                v1 = tri1.v2;
                v2 = tri1.v0;
                break;
        }


    
        addTECollsion<T>(&tri2, &v1, &v2, collisions, idx, collision_id, minimal_collisions, edge_coefs, counter);
    }
}



template<typename T>
__global__
void
collectTECandidates(long2 *collisions, Triangle<T> *triangles, int num_cand_collisions,
                       long4 *minimal_collisions, int *counter) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    

    if (idx < num_cand_collisions) {
        int collision_id = blockIdx.y;

        int first_tri_idx = collisions[idx].x;
        int second_tri_idx = collisions[idx].y;

        Triangle<T> tri1;
        Triangle<T> tri2;

        if (collision_id < 3) {
            tri1 = triangles[first_tri_idx];
            tri2 = triangles[second_tri_idx];
        } else {
            tri1 = triangles[second_tri_idx];
            tri2 = triangles[first_tri_idx];
        }

        int edge_id = collision_id % 3;


        vec3<T> v1;
        vec3<T> v2;
        switch (edge_id){
            case 0:
                v1 = tri1.v0;
                v2 = tri1.v1;
                break;
            case 1:
                v1 = tri1.v1;
                v2 = tri1.v2;
                break;
            case 2:
                v1 = tri1.v2;
                v2 = tri1.v0;
                break;
        }

        int loop_id = -1;
        bool v1_shared = hasVertex<T>(tri2, v1);
        bool v2_shared = hasVertex<T>(tri2, v2);

        if (v1_shared) {
            if (collision_id < 3) {
                loop_id = 0;
                int to_id = atomicAdd(counter, 1);
                long2 from_collision = collisions[idx];
                minimal_collisions[to_id] = make_long4(from_collision.x, from_collision.y, collision_id, loop_id);
            }
            return;
        } else if (v2_shared) {
            return;
        }


        // if (shareVertex(tri1, tri2)) {
        //     if (collision_id < 3 and hasVertex<T>(tri2, v1)) {
        //         loop_id = 0;
        //         int to_id = atomicAdd(counter, 1);
        //         long2 from_collision = collisions[idx];
        //         minimal_collisions[to_id] = make_long4(from_collision.x, from_collision.y, collision_id, loop_id);
        //     }
        //     return;
        // }



        AABB<T> bbox0 = tri2.ComputeBBox();
        AABB<T> bbox1 = tri2.ComputeBBoxEdge(edge_id, (edge_id+1)%3);
        bool overlap = checkOverlap<T>(bbox0, bbox1);




        if (overlap) {
            int to_id = atomicAdd(counter, 1);
            long2 from_collision = collisions[idx];
            minimal_collisions[to_id] = make_long4(from_collision.x, from_collision.y, collision_id, loop_id);
        }


    }
}





void find_triangle_edge_collisions_from_bbox_tree(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                    at::Tensor *collision_tensor_ptr, at::Tensor *coeff_tensor_ptr,
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


    thrust::device_vector<int> valid_collision_idx_cnt(batch_size);
    thrust::fill(valid_collision_idx_cnt.begin(), valid_collision_idx_cnt.end(), 0);


    thrust::device_vector<int> minimal_collision_idx_cnt(batch_size);
    thrust::fill(minimal_collision_idx_cnt.begin(), minimal_collision_idx_cnt.end(), 0);

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


//                    std::cout << "\nnum_triangles counter " << num_triangles << "\n";
//                     ================================ findPotentialCollisions
                    findPotentialCollisions<scalar_t><<<gridSize, blockSize>>>(
                            collisionIndices.data().get(),
                            internal_nodes.data().get(),
                            leaf_nodes.data().get(), triangle_ids.data().get(), num_triangles,
                            max_total_candidates, &collision_idx_cnt.data().get()[bidx], 0.);


//                    std::cout << "\nnum_cand_collisions counter " << (int)collision_idx_cnt[bidx] << "\n";
//                    std::cout << "\nmax_total_candidates counter " << (int)max_total_candidates << "\n";
                    if ((int)collision_idx_cnt[bidx] > max_total_candidates) {
                        printf("Number of candidate collisions exceeds maximal allowed number. Some of the candidate collisions are omitted.\n");
                    }
                    cudaDeviceSynchronize();


                    int num_cand_collisions =
                            thrust::reduce(thrust::make_transform_iterator(
                                                   collisionIndices.begin(), is_valid_cnt()),
                                           thrust::make_transform_iterator(
                                                   collisionIndices.end(), is_valid_cnt()));

//                    std::cout << "\nnum_cand_collisions " << num_cand_collisions << "\n";
                        // printf("num_valid_collisions: %d\n", num_valid_collisions);
                    if (num_cand_collisions > 0) {

                        Triangle<scalar_t> *triangles_ptr =
                                (TrianglePtr<scalar_t>) triangle_float_ptr +
                                num_triangles * bidx;


                        // Keep only the pairs of ids where a bounding box to bounding box
                        // collision was detected.
                        thrust::device_vector <long2> collisions(num_cand_collisions,
                                                                 make_long2(-1, -1));
                        thrust::copy_if(collisionIndices.begin(), collisionIndices.end(),
                                        collisions.begin(), is_valid_cnt());

                        
                        thrust::device_vector <long2> valid_collisions(num_cand_collisions,
                                                                 make_long2(-1, -1));

                        cudaCheckError();

                        int tri_grid_size =
                                (collisions.size() + blockSize - 1) / blockSize;

//                        print_candidates<scalar_t><<<tri_grid_size, blockSize>>>(collisions.data().get(), collisions.size());
                        checkTriangleIntersections<scalar_t><<<tri_grid_size, blockSize>>>( // checkTriangleIntersections
                                collisions.data().get(), triangles_ptr, 
                                valid_collisions.data().get(), collisions.size(), 0.,
                                &valid_collision_idx_cnt.data().get()[bidx], 0);

                        cudaCheckError();

                        int num_valid_collisions = (int)valid_collision_idx_cnt[bidx];

                        if (num_valid_collisions > 0) {

                            // thrust::device_vector <long3> minimal_collisions(num_valid_collisions*6,
                            //                                         make_long3(-1, -1, -1));

                            thrust::device_vector <long4> minimal_collisions(num_valid_collisions*6,
                                        make_long4(-1, -1, -1, -1));

                            thrust::device_vector <scalar_t> edge_coefs(num_valid_collisions*6,
                                                                                        -1.);

                            int te_grid_size =
                                    (num_valid_collisions + blockSize - 1) / blockSize;
                            dim3 blockGrid(te_grid_size, 6);

    // 
                            // // collectTEIntersections<scalar_t><<<blockGrid, blockSize>>>( // checkTriangleIntersections
                            collectTEIntersections<scalar_t><<<blockGrid, blockSize>>>( // checkTriangleIntersections
                                    valid_collisions.data().get(), triangles_ptr, num_valid_collisions,
                                    minimal_collisions.data().get(), edge_coefs.data().get(), 
                                    &minimal_collision_idx_cnt.data().get()[bidx]);
                                    // &valid_collision_idx_cnt.data().get()[bidx]);
                            cudaCheckError();


    // //
                            long *dev_ptr = collision_tensor_ptr->data<long>();
                            cudaMemcpy(dev_ptr + bidx * num_triangles * max_collisions * 3,
                                    (long *) minimal_collisions.data().get(),
                                    4 * minimal_collisions.size() * sizeof(long),
                                    cudaMemcpyDeviceToDevice);
                            cudaCheckError();

                            scalar_t *coeff_ptr = coeff_tensor_ptr->data<scalar_t>();
                            cudaMemcpy(coeff_ptr + bidx * num_triangles * max_collisions,
                                    (scalar_t *) edge_coefs.data().get(),
                                    edge_coefs.size() * sizeof(scalar_t),
                                    cudaMemcpyDeviceToDevice);
                            cudaCheckError();
                        }

                    }
                }

            }));

}


void find_triangle_edge_candidates(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
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


    thrust::device_vector<int> valid_collision_idx_cnt(batch_size);
    thrust::fill(valid_collision_idx_cnt.begin(), valid_collision_idx_cnt.end(), 0);


    thrust::device_vector<int> minimal_collision_idx_cnt(batch_size);
    thrust::fill(minimal_collision_idx_cnt.begin(), minimal_collision_idx_cnt.end(), 0);

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


//                    std::cout << "\nnum_triangles counter " << num_triangles << "\n";
//                     ================================ findPotentialCollisions
                    findPotentialCollisions<scalar_t><<<gridSize, blockSize>>>(
                            collisionIndices.data().get(),
                            internal_nodes.data().get(),
                            leaf_nodes.data().get(), triangle_ids.data().get(), num_triangles,
                            max_total_candidates, &collision_idx_cnt.data().get()[bidx], 0.);


//                    std::cout << "\nnum_cand_collisions counter " << (int)collision_idx_cnt[bidx] << "\n";
//                    std::cout << "\nmax_total_candidates counter " << (int)max_total_candidates << "\n";
                    if ((int)collision_idx_cnt[bidx] > max_total_candidates) {
                        printf("Number of candidate collisions exceeds maximal allowed number. Some of the candidate collisions are omitted.\n");
                    }
                    cudaDeviceSynchronize();


                    int num_cand_collisions =
                            thrust::reduce(thrust::make_transform_iterator(
                                                   collisionIndices.begin(), is_valid_cnt()),
                                           thrust::make_transform_iterator(
                                                   collisionIndices.end(), is_valid_cnt()));

//                    std::cout << "\nnum_cand_collisions " << num_cand_collisions << "\n";
                        // printf("num_valid_collisions: %d\n", num_valid_collisions);
                    if (num_cand_collisions > 0) {

                        Triangle<scalar_t> *triangles_ptr =
                                (TrianglePtr<scalar_t>) triangle_float_ptr +
                                num_triangles * bidx;


                        // Keep only the pairs of ids where a bounding box to bounding box
                        // collision was detected.
                        thrust::device_vector <long2> collisions(num_cand_collisions,
                                                                 make_long2(-1, -1));
                        thrust::copy_if(collisionIndices.begin(), collisionIndices.end(),
                                        collisions.begin(), is_valid_cnt());

                        
                        thrust::device_vector <long2> valid_collisions(num_cand_collisions,
                                                                 make_long2(-1, -1));

                        cudaCheckError();

                        int tri_grid_size =
                                (collisions.size() + blockSize - 1) / blockSize;

//                        print_candidates<scalar_t><<<tri_grid_size, blockSize>>>(collisions.data().get(), collisions.size());
                        checkTriangleIntersections<scalar_t><<<tri_grid_size, blockSize>>>( // checkTriangleIntersections
                                collisions.data().get(), triangles_ptr, 
                                valid_collisions.data().get(), collisions.size(), 0.,
                                &valid_collision_idx_cnt.data().get()[bidx], 1);

                        cudaCheckError();

                        int num_valid_collisions = (int)valid_collision_idx_cnt[bidx];

                        if (num_valid_collisions > 0) {

                            // thrust::device_vector <long3> minimal_collisions(num_valid_collisions*6,
                            //                                         make_long3(-1, -1, -1));

                            thrust::device_vector <long4> minimal_collisions(num_valid_collisions*6,
                                        make_long4(-1, -1, -1, -1));


                            int te_grid_size =
                                    (num_valid_collisions + blockSize - 1) / blockSize;
                            dim3 blockGrid(te_grid_size, 6);


                            // // collectTEIntersections<scalar_t><<<blockGrid, blockSize>>>( // checkTriangleIntersections
                            collectTECandidates<scalar_t><<<blockGrid, blockSize>>>( // checkTriangleIntersections
                                    valid_collisions.data().get(), triangles_ptr, num_valid_collisions,
                                    minimal_collisions.data().get(), 
                                    &minimal_collision_idx_cnt.data().get()[bidx]);
                                    // &valid_collision_idx_cnt.data().get()[bidx]);
                            cudaCheckError();


                            long *dev_ptr = collision_tensor_ptr->data<long>();
                            cudaMemcpy(dev_ptr + bidx * num_triangles * max_collisions * 3,
                                    (long *) minimal_collisions.data().get(),
                                    4 * minimal_collisions.size() * sizeof(long),
                                    cudaMemcpyDeviceToDevice);
                            cudaCheckError();

                        }

                    }
                }

            }));

}