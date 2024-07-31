#pragma once

#import "flags.h"
#import "utils.cu"
#import "bvh.cu"
#import "figures.cu"
#import "build_bvh.cu"



template<typename T>
__global__
void
collectTECandidates2(long2 *collisions, Triangle<T> *triangles, int num_cand_collisions, int2 *binmasks) {
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

                atomicAdd(&binmasks[idx].x, (1 << collision_id));
                atomicAdd(&binmasks[idx].y, (1 << collision_id));

                // binmasks[to_id].x += (1 << collision_id);
                // binmasks[to_id].y += (1 << collision_id);


                // minimal_collisions[to_id] = make_long4(from_collision.x, from_collision.y, collision_id, loop_id);
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
            atomicAdd(&binmasks[idx].x, (1 << collision_id));
            // binmasks[to_id].x += (1 << collision_id);
            // minimal_collisions[to_id] = make_long4(from_collision.x, from_collision.y, collision_id, loop_id);
        }




    }
}

void find_triangle_edge_candidates2(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                    at::Tensor *collision_tensor_ptr, at::Tensor *binmasks_ptr,
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

                            // thrust::device_vector <long2> minimal_collisions(num_valid_collisions,
                            //             make_long2(-1, -1));

                            thrust::device_vector <int2> binmasks(num_valid_collisions,
                                        make_int2(0, 0));
                            // thrust::device_vector <long2> binmasks(num_valid_collisions,
                            //             make_long2(0, 0));

                            // thrust::device_vector <int> binmasks(num_valid_collisions*2, 0);

                            int te_grid_size =
                                    (num_valid_collisions + blockSize - 1) / blockSize;
                            dim3 blockGrid(te_grid_size, 6);


                            // // collectTEIntersections<scalar_t><<<blockGrid, blockSize>>>( // checkTriangleIntersections
                            collectTECandidates2<scalar_t><<<blockGrid, blockSize>>>( // checkTriangleIntersections
                                    valid_collisions.data().get(), triangles_ptr, num_valid_collisions, binmasks.data().get());
                            cudaCheckError();


                            long *dev_ptr = collision_tensor_ptr->data<long>();
                            cudaMemcpy(dev_ptr + bidx * num_triangles * max_collisions * 2,
                                    (long *) valid_collisions.data().get(),
                                    2 * valid_collisions.size() * sizeof(long),
                                    cudaMemcpyDeviceToDevice);
                            cudaCheckError();


                            int *dev_ptr2 = binmasks_ptr->data<int>();
                            cudaMemcpy(dev_ptr2 + bidx * num_triangles * max_collisions * 2,
                                    (int *) binmasks.data().get(),
                                    2 * binmasks.size() * sizeof(int),
                                    cudaMemcpyDeviceToDevice);


                        }

                    }
                }

            }));

}