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
#include "figures.cu"
#include "aabb.cu"
#include "bvh.cu"


template<typename T>
__global__ void ComputeMortonCodes(Triangle<T> *triangles, int num_triangles,
                                   AABB<T> *scene_bb,
                                   MortonCode *morton_codes) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_triangles) {
        // Fetch the current triangle
        Triangle<T> tri = triangles[idx];
        vec3<T> centroid = (tri.v0 + tri.v1 + tri.v2) / (T) 3.0;

        T x = (centroid.x - scene_bb->min_t.x) /
              (scene_bb->max_t.x - scene_bb->min_t.x);
        T y = (centroid.y - scene_bb->min_t.y) /
              (scene_bb->max_t.y - scene_bb->min_t.y);
        T z = (centroid.z - scene_bb->min_t.z) /
              (scene_bb->max_t.z - scene_bb->min_t.z);

        morton_codes[idx] = morton3D<T>(x, y, z);
    }
    return;
}

template<typename T>
__global__ void ComputeMortonCodes(Triangle<T> *triangles1, Triangle<T> *triangles2, int num_triangles,
                                   AABB<T> *scene_bb,
                                   MortonCode *morton_codes) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_triangles) {
        // Fetch the current triangle
        Triangle<T> tri1 = triangles1[idx];
        Triangle<T> tri2 = triangles2[idx];
        vec3<T> centroid1 = (tri1.v0 + tri1.v1 + tri1.v2) / (T) 3.0;
        vec3<T> centroid2 = (tri2.v0 + tri2.v1 + tri2.v2) / (T) 3.0;
        vec3<T> centroid = (centroid1 + centroid2) / (T) 2.0;

        T x = (centroid.x - scene_bb->min_t.x) /
              (scene_bb->max_t.x - scene_bb->min_t.x);
        T y = (centroid.y - scene_bb->min_t.y) /
              (scene_bb->max_t.y - scene_bb->min_t.y);
        T z = (centroid.z - scene_bb->min_t.z) /
              (scene_bb->max_t.z - scene_bb->min_t.z);

        morton_codes[idx] = morton3D<T>(x, y, z);
    }
    return;
}


template<typename T>
__global__ void BuildRadixTree(MortonCode *morton_codes, int num_triangles,
                               int *triangle_ids, BVHNodePtr<T> internal_nodes,
                               BVHNodePtr<T> leaf_nodes) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_triangles - 1)
        return;

    int delta_next = LongestCommonPrefix(idx, idx + 1, morton_codes,
                                         num_triangles, triangle_ids);
    int delta_last = LongestCommonPrefix(idx, idx - 1, morton_codes,
                                         num_triangles, triangle_ids);
    // Find the direction of the range
    int direction = delta_next - delta_last >= 0 ? 1 : -1;

    int delta_min = LongestCommonPrefix(idx, idx - direction, morton_codes,
                                        num_triangles, triangle_ids);

    // Do binary search to compute the upper bound for the length of the range
    int lmax = 2;
    while (LongestCommonPrefix(idx, idx + lmax * direction, morton_codes,
                               num_triangles, triangle_ids) > delta_min) {
        lmax *= 2;
    }

    // Use binary search to find the other end.
    int l = 0;
    int divider = 2;
    for (int t = lmax / divider; t >= 1; divider *= 2) {
        if (LongestCommonPrefix(idx, idx + (l + t) * direction, morton_codes,
                                num_triangles, triangle_ids) > delta_min) {
            l = l + t;
        }
        t = lmax / divider;
    }
    int j = idx + l * direction;

    // Find the length of the longest common prefix for the current node
    int node_delta =
            LongestCommonPrefix(idx, j, morton_codes, num_triangles, triangle_ids);
    int s = 0;
    divider = 2;
    // Search for the split position using binary search.
    for (int t = (l + (divider - 1)) / divider; t >= 1; divider *= 2) {
        if (LongestCommonPrefix(idx, idx + (s + t) * direction, morton_codes,
                                num_triangles, triangle_ids) > node_delta) {
            s = s + t;
        }
        t = (l + (divider - 1)) / divider;
    }
    // gamma in the Karras paper
    int split = idx + s * direction + min(direction, 0);

    // Assign the parent and the left, right children for the current node
    BVHNodePtr<T> curr_node = internal_nodes + idx;
    if (min(idx, j) == split) {
        curr_node->left = leaf_nodes + split;
        (leaf_nodes + split)->parent = curr_node;
    } else {
        curr_node->left = internal_nodes + split;
        (internal_nodes + split)->parent = curr_node;
    }
    if (max(idx, j) == split + 1) {
        curr_node->right = leaf_nodes + split + 1;
        (leaf_nodes + split + 1)->parent = curr_node;
    } else {
        curr_node->right = internal_nodes + split + 1;
        (internal_nodes + split + 1)->parent = curr_node;
    }
}

template<typename T>
__global__ void CreateHierarchy(BVHNodePtr<T> internal_nodes,
                                BVHNodePtr<T> leaf_nodes, int num_triangles,
                                Triangle<T> *triangles, int *triangle_ids,
                                int *atomic_counters, AABB<T> *bboxes) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_triangles)
        return;

    BVHNodePtr<T> leaf = leaf_nodes + idx;
    // Assign the index to the primitive
    leaf->idx = triangle_ids[idx];

    Triangle<T> tri = triangles[triangle_ids[idx]];
    // Assign the bounding box of the triangle to the leaves
//    leaf->bbox = tri.ComputeBBox();
    leaf->bbox = bboxes[triangle_ids[idx]];
//    leaf->bbox = bboxes[idx];

    // the rightmost node that ca be reached from a leaf node is itself
    leaf->rightmost = leaf;


    BVHNodePtr<T> curr_node = leaf->parent;
    int current_idx = curr_node - internal_nodes;

    // Increment the atomic counter
    int curr_counter = atomicAdd(atomic_counters + current_idx, 1);
    while (true) {
        // atomicAdd returns the old value at the specified address. Thus the
        // first thread to reach this point will immediately return
        // So the `curr_node` will be processed only by the second thread to reach here,
        // when both its' children have been processed
        if (curr_counter == 0)
            break;

        // Calculate the bounding box of the current node as the union of the
        // bounding boxes of its children.
        AABB<T> left_bb = curr_node->left->bbox;
        AABB<T> right_bb = curr_node->right->bbox;
        curr_node->bbox = left_bb + right_bb;
        // Store a pointer to the right most node that can be reached from this
        // internal node.

        curr_node->rightmost =
                curr_node->left->rightmost > curr_node->right->rightmost
                ? curr_node->left->rightmost
                : curr_node->right->rightmost;


        // If we have reached the root break
        if (curr_node == internal_nodes)
            break;

        // Proceed to the parent of the node
        curr_node = curr_node->parent;
        // Calculate its position in the flat array
        current_idx = curr_node - internal_nodes;
        // Update the visitation counter
        curr_counter = atomicAdd(atomic_counters + current_idx, 1);

    }

    return;
}


template<typename T>
__global__ void ComputeTriBoundingBoxes(Triangle<T> *triangles,
                                        int num_triangles, AABB<T> *bboxes) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_triangles) {
        bboxes[idx] = triangles[idx].ComputeBBox();
    }
}

template<typename T>
void buildBVH(BVHNodePtr<T> internal_nodes, BVHNodePtr<T> leaf_nodes,
              Triangle<T> *__restrict__ triangles,
              thrust::device_vector<int> *triangle_ids, int num_triangles,
              int batch_size) {


    thrust::device_vector <AABB<T>> bounding_boxes(num_triangles);

    int blockSize = NUM_THREADS;
    int gridSize = (num_triangles + blockSize - 1) / blockSize;

    // Compute the bounding box for all the triangles

    ComputeTriBoundingBoxes<T><<<gridSize, blockSize>>>(
            triangles, num_triangles, bounding_boxes.data().get());

    cudaCheckError();



    // Compute the union of all the bounding boxes
    AABB<T> host_scene_bb = thrust::reduce(
            bounding_boxes.begin(), bounding_boxes.end(), AABB<T>(), MergeAABB<T>());

    cudaCheckError();


    // TODO: Custom reduction ?
    // Copy the bounding box back to the GPU
    AABB<T> *scene_bb_ptr;
    cudaMalloc(&scene_bb_ptr, sizeof(AABB<T>));
    cudaMemcpy(scene_bb_ptr, &host_scene_bb, sizeof(AABB<T>),
               cudaMemcpyHostToDevice);

    thrust::device_vector <MortonCode> morton_codes(num_triangles);

    // Compute the morton codes for the centroids of all the primitives
    ComputeMortonCodes<T><<<gridSize, blockSize>>>(
            triangles, num_triangles, scene_bb_ptr,
            morton_codes.data().get());

    cudaCheckError();


    // Construct an array of triangle ids.
    // (fill triangle_ids with values [0, num_triangles-1])
    thrust::sequence(triangle_ids->begin(), triangle_ids->end());


    // Sort the triangles according to the morton code
    // (sort `triangle_ids` in-place by key `morton_codes`)
    try {
        thrust::sort_by_key(morton_codes.begin(), morton_codes.end(),
                            triangle_ids->begin());

    } catch (thrust::system_error e) {
        std::cout << "Error inside sort: " << e.what() << std::endl;
    }
    // ===============================================
    // Construct the radix tree using the sorted morton code sequence
    // for internal_nodes, only set `left`, `right` and `parent` (for non-root)
    // for `leaf_nodes`, only set `parent`
    // `bbox` and `idx` is left empty for all nodes
    BuildRadixTree<T><<<gridSize, blockSize>>>(
            morton_codes.data().get(), num_triangles, triangle_ids->data().get(),
            internal_nodes, leaf_nodes);

    cudaCheckError();


    // Create an array that contains the atomic counters for each node in the
    // tree
    thrust::device_vector<int> counters(num_triangles);
//
    // Build the Bounding Volume Hierarchy in parallel from the leaves to the
    // root
    CreateHierarchy<T><<<gridSize, blockSize>>>(
            internal_nodes, leaf_nodes, num_triangles, triangles,
            triangle_ids->data().get(), counters.data().get(), bounding_boxes.data().get());


    cudaCheckError();
    cudaFree(scene_bb_ptr);
    return;
}


template<typename T>
__global__ void
CreateBboxesTree(BVHNodePtr<T> internal_nodes, BVHNodePtr<T> leaf_nodes, vec3<T> *bboxes, long4 *bboxes_tree,
                 int num_triangles) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= 2 * num_triangles - 1) {
        return;
    }


    bool is_leaf = idx >= (num_triangles - 1);
    int from_idx = idx;
    if (is_leaf) {
        from_idx -= (num_triangles - 1);
    }

    BVHNodePtr<T> node;
    if (is_leaf) {
        node = leaf_nodes + from_idx;
    } else {
        node = internal_nodes + from_idx;
    }
    AABB<T> bbox = node->bbox;


    int parent_idx = -1;
    int left_idx = -1;
    int right_idx = -1;

    if (node->parent) {
        parent_idx = node->parent - internal_nodes;
    }
//
    if (node->left) {
        if (node->left->isLeaf()) {
            left_idx = num_triangles - 1 + (node->left - leaf_nodes);
        } else {
            left_idx = node->left - internal_nodes;
        }
    }

    if (node->right) {
        if (node->right->isLeaf()) {
            right_idx = num_triangles - 1 + (node->right - leaf_nodes);
        } else {
            right_idx = node->right - internal_nodes;
        }
    }

    int self_idx = node->idx;

//    if (idx == 3024) {
//        printf("\n\nidx: %d\n"
//               "parent_idx: %d, left_idx: %d, right_idx: %d, self_idx: %d\n"
//               "min_t: (%g, %g, %g)\n"
//               "max_t: (%g, %g, %g)\n,"
//               "FLT: %g\n"
//               "DBL: %g\n", idx,
//               parent_idx, left_idx, right_idx, self_idx,
//               bbox.min_t.x, bbox.min_t.y, bbox.min_t.z,
//               bbox.max_t.x, bbox.max_t.y, bbox.max_t.z,
//               FLT_MAX, DBL_MAX);
//    }

    bboxes_tree[idx] = make_long4(parent_idx, left_idx, right_idx, self_idx);
    bboxes[2 * idx] = bbox.min_t;
    bboxes[2 * idx + 1] = bbox.max_t;


}


void build_bvh(at::Tensor triangles, at::Tensor *bboxes_tensor_ptr, at::Tensor *tree_tensor_ptr) {
    const auto batch_size = triangles.size(0);
    const auto num_triangles = triangles.size(1);

    thrust::device_vector<int> triangle_ids(num_triangles);

    // int *counter;
    // number of collisions in each sample of the batch
    thrust::device_vector<int> collision_idx_cnt(batch_size);
    thrust::fill(collision_idx_cnt.begin(), collision_idx_cnt.end(), 0);

    // Construct the bvh tree
    AT_DISPATCH_FLOATING_TYPES(
            triangles.type(), "bvh_tree_building", ([&] {
                thrust::device_vector <BVHNode<scalar_t>> leaf_nodes(num_triangles);
                thrust::device_vector <BVHNode<scalar_t>> internal_nodes(num_triangles - 1);

                auto triangle_float_ptr = triangles.data<scalar_t>();

                for (int bidx = 0; bidx < batch_size; ++bidx) {
                    Triangle<scalar_t> *triangles_ptr =
                            (TrianglePtr<scalar_t>) triangle_float_ptr +
                            num_triangles * bidx;
//
                    // ================================== buildBVH call
                    // internal_nodes: non-initialized vector of `BVHNode` of size `num_triangles - 1`
                    // leaf_nodes: non-initialized vector of `BVHNode` of size `num_triangles`
                    // triangles_ptr: pointer Triangl<scalar_t>* to triangles (triplets of vertex inds)
                    // triangle_ids: non-initialized vector of int of size `num_triangles`
                    // num_triangles: int
                    // batch_size: int
                    buildBVH<scalar_t>(internal_nodes.data().get(),
                                       leaf_nodes.data().get(), triangles_ptr,
                                       &triangle_ids, num_triangles, batch_size);
//
//
                    int num_bboxes = 2 * num_triangles - 1;
                    thrust::device_vector <vec3<scalar_t>> bboxes(num_bboxes * 2);
                    thrust::device_vector <long4> bboxes_tree(num_bboxes);

                    int blockSize = NUM_THREADS;
                    int gridSize = (num_bboxes + blockSize - 1) / blockSize;
                    thrust::device_vector<int> counters(num_triangles);
                    CreateBboxesTree<scalar_t><<<gridSize, blockSize>>>(internal_nodes.data().get(),
                                                                        leaf_nodes.data().get(),
                                                                        bboxes.data().get(), bboxes_tree.data().get(),
                                                                        num_triangles);

                    scalar_t *bboxes_dev_ptr = bboxes_tensor_ptr->data<scalar_t>();
                    cudaMemcpy(bboxes_dev_ptr + bidx * num_bboxes * 6,
                               (scalar_t *) bboxes.data().get(),
                               3 * bboxes.size() * sizeof(scalar_t),
                               cudaMemcpyDeviceToDevice);


                    long *tree_dev_ptr = tree_tensor_ptr->data<long>();
                    cudaMemcpy(tree_dev_ptr + bidx * num_bboxes * 4,
                               (long *) bboxes_tree.data().get(),
                               4 * bboxes_tree.size() * sizeof(long),
                               cudaMemcpyDeviceToDevice);
                    cudaCheckError();

                    cudaDeviceSynchronize();
                }
            }));


    cudaCheckError();
}


template<typename T>
__global__ void MergeTriBoundingBoxes(AABB<T> *bboxes1, AABB<T> *bboxes2, AABB<T> *bboxes_out,
                                      int num_triangles) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_triangles)
        return;

    bboxes_out[idx] = MergeAABB<T>()(bboxes1[idx], bboxes2[idx]);
}

//template<typename T>
//__global__ void InitializeBVHNodes(BVHNodePtr<T> nodes,
//                                      int num_nodes) {
//    int idx = threadIdx.x + blockDim.x * blockIdx.x;
//    if (idx >= num_nodes)
//        return;
//
////    BVHNodePtr<T> node = nodes + idx;
////    node ->
//
//
//    bboxes_out[idx] = MergeAABB<T>()(bboxes1[idx], bboxes2[idx]);
//}

template<typename T>
void buildBVH(BVHNodePtr<T> internal_nodes, BVHNodePtr<T> leaf_nodes,
              Triangle<T> *__restrict__ triangles, Triangle<T> *__restrict__ triangles_next,
              thrust::device_vector<int> *triangle_ids, int num_triangles,
              int batch_size) {


    thrust::device_vector <AABB<T>> bounding_boxes_curr(num_triangles);
    thrust::device_vector <AABB<T>> bounding_boxes_next(num_triangles);
    thrust::device_vector <AABB<T>> bounding_boxes(num_triangles);

    int blockSize = NUM_THREADS;
    int gridSize = (num_triangles + blockSize - 1) / blockSize;

    // Compute the bounding box for all the triangles

    ComputeTriBoundingBoxes<T><<<gridSize, blockSize>>>(
            triangles, num_triangles, bounding_boxes_curr.data().get());
    ComputeTriBoundingBoxes<T><<<gridSize, blockSize>>>(
            triangles_next, num_triangles, bounding_boxes_next.data().get());
    MergeTriBoundingBoxes<T><<<gridSize, blockSize>>>(bounding_boxes_curr.data().get(),
                                                      bounding_boxes_next.data().get(),
                                                      bounding_boxes.data().get(), num_triangles);

    cudaCheckError();



    // Compute the union of all the bounding boxes
    AABB<T> host_scene_bb = thrust::reduce(
            bounding_boxes.begin(), bounding_boxes.end(), AABB<T>(), MergeAABB<T>());

    cudaCheckError();


    // TODO: Custom reduction ?
    // Copy the bounding box back to the GPU
    AABB<T> *scene_bb_ptr;
    cudaMalloc(&scene_bb_ptr, sizeof(AABB<T>));
    cudaMemcpy(scene_bb_ptr, &host_scene_bb, sizeof(AABB<T>),
               cudaMemcpyHostToDevice);

    thrust::device_vector <MortonCode> morton_codes(num_triangles);

    // Compute the morton codes for the centroids of all the primitives
    ComputeMortonCodes<T><<<gridSize, blockSize>>>(
            triangles, triangles_next, num_triangles, scene_bb_ptr,
            morton_codes.data().get());

    cudaCheckError();


    // Construct an array of triangle ids.
    // (fill triangle_ids with values [0, num_triangles-1])
    thrust::sequence(triangle_ids->begin(), triangle_ids->end());


    // Sort the triangles according to the morton code
    // (sort `triangle_ids` in-place by key `morton_codes`)
    try {
        thrust::sort_by_key(morton_codes.begin(), morton_codes.end(),
                            triangle_ids->begin());

    } catch (thrust::system_error e) {
        std::cout << "Error inside sort: " << e.what() << std::endl;
    }
    // ===============================================
    // Construct the radix tree using the sorted morton code sequence
    // for internal_nodes, only set `left`, `right` and `parent` (for non-root)
    // for `leaf_nodes`, only set `parent`
    // `bbox` and `idx` is left empty for all nodes
    BuildRadixTree<T><<<gridSize, blockSize>>>(
            morton_codes.data().get(), num_triangles, triangle_ids->data().get(),
            internal_nodes, leaf_nodes);

    cudaCheckError();


    // Create an array that contains the atomic counters for each node in the
    // tree
    thrust::device_vector<int> counters(num_triangles);
//
    // Build the Bounding Volume Hierarchy in parallel from the leaves to the
    // root
    CreateHierarchy<T><<<gridSize, blockSize>>>(
            internal_nodes, leaf_nodes, num_triangles, triangles,
            triangle_ids->data().get(), counters.data().get(), bounding_boxes.data().get());


    cudaCheckError();
    cudaFree(scene_bb_ptr);
    return;
}

void build_bvh(at::Tensor triangles, at::Tensor triangles_next, at::Tensor *bboxes_tensor_ptr,
               at::Tensor *tree_tensor_ptr) {
    const auto batch_size = triangles.size(0);
    const auto num_triangles = triangles.size(1);

    thrust::device_vector<int> triangle_ids(num_triangles);


    // int *counter;
    // number of collisions in each sample of the batch
    thrust::device_vector<int> collision_idx_cnt(batch_size);
    thrust::fill(collision_idx_cnt.begin(), collision_idx_cnt.end(), 0);

    // Construct the bvh tree
    AT_DISPATCH_FLOATING_TYPES(
            triangles.type(), "bvh_tree_building", ([&] {
                thrust::device_vector <BVHNode<scalar_t>> leaf_nodes(num_triangles);
                thrust::device_vector <BVHNode<scalar_t>> internal_nodes(num_triangles - 1);

                auto triangle_float_ptr = triangles.data<scalar_t>();
                auto triangle_next_float_ptr = triangles_next.data<scalar_t>();

                for (int bidx = 0; bidx < batch_size; ++bidx) {
                    Triangle<scalar_t> *triangles_ptr =
                            (TrianglePtr<scalar_t>) triangle_float_ptr +
                            num_triangles * bidx;

                    Triangle<scalar_t> *triangles_next_ptr =
                            (TrianglePtr<scalar_t>) triangle_next_float_ptr +
                            num_triangles * bidx;

//
                    // ================================== buildBVH call
                    // internal_nodes: non-initialized vector of `BVHNode` of size `num_triangles - 1`
                    // leaf_nodes: non-initialized vector of `BVHNode` of size `num_triangles`
                    // triangles_ptr: pointer Triangl<scalar_t>* to triangles (triplets of vertex inds)
                    // triangle_ids: non-initialized vector of int of size `num_triangles`
                    // num_triangles: int
                    // batch_size: int
                    buildBVH<scalar_t>(internal_nodes.data().get(),
                                       leaf_nodes.data().get(), triangles_ptr, triangles_next_ptr,
                                       &triangle_ids, num_triangles, batch_size);
//
//
                    int num_bboxes = 2 * num_triangles - 1;
                    thrust::device_vector <vec3<scalar_t>> bboxes(num_bboxes * 2);
                    thrust::device_vector <long4> bboxes_tree(num_bboxes);

                    int blockSize = NUM_THREADS;
                    int gridSize = (num_bboxes + blockSize - 1) / blockSize;
                    thrust::device_vector<int> counters(num_triangles);
                    CreateBboxesTree<scalar_t><<<gridSize, blockSize>>>(internal_nodes.data().get(),
                                                                        leaf_nodes.data().get(),
                                                                        bboxes.data().get(), bboxes_tree.data().get(),
                                                                        num_triangles);

                    scalar_t *bboxes_dev_ptr = bboxes_tensor_ptr->data<scalar_t>();
                    cudaMemcpy(bboxes_dev_ptr + bidx * num_bboxes * 6,
                               (scalar_t *) bboxes.data().get(),
                               3 * bboxes.size() * sizeof(scalar_t),
                               cudaMemcpyDeviceToDevice);


                    long *tree_dev_ptr = tree_tensor_ptr->data<long>();
                    cudaMemcpy(tree_dev_ptr + bidx * num_bboxes * 4,
                               (long *) bboxes_tree.data().get(),
                               4 * bboxes_tree.size() * sizeof(long),
                               cudaMemcpyDeviceToDevice);
                    cudaCheckError();

                    blockSize = NUM_THREADS;
                    gridSize = (num_triangles + blockSize - 1) / blockSize;
                    cudaDeviceSynchronize();
                }
            }));


    cudaCheckError();
}


template<typename T>
__device__ BVHNodePtr<T>
get_node_by_idx(BVHNodePtr<T> internal_nodes, BVHNodePtr<T> leaf_nodes, int num_triangles, int idx) {
    if (idx < 0) {
        return nullptr;
    }

    if (idx >= (num_triangles - 1)) {
        idx -= (num_triangles - 1);
        return leaf_nodes + idx;
    } else {
        return internal_nodes + idx;
    }
}

template<typename T>
__global__ void
build_bvh_from_array_tree(AABB<T> *bboxes, long4 *tree, BVHNodePtr<T> internal_nodes, BVHNodePtr<T> leaf_nodes,
                          int *triangle_ids,
                          int num_nodes, int num_triangles) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_nodes) return;

    long4 *long_node_ptr = tree + idx;
    AABB<T> *bbox_ptr = bboxes + idx;

    bool is_leaf = idx >= (num_triangles - 1);
    int to_idx = idx;
    BVHNodePtr<T> node;
    if (is_leaf) {
        to_idx -= (num_triangles - 1);
        node = leaf_nodes + to_idx;
        node->idx = to_idx;
    } else {
        node = internal_nodes + to_idx;
    }

    node->bbox = *bbox_ptr;

    BVHNodePtr<T> parent_node = get_node_by_idx(internal_nodes, leaf_nodes, num_triangles, long_node_ptr->x);
    BVHNodePtr<T> left_node = get_node_by_idx(internal_nodes, leaf_nodes, num_triangles, long_node_ptr->y);
    BVHNodePtr<T> right_node = get_node_by_idx(internal_nodes, leaf_nodes, num_triangles, long_node_ptr->z);

    node->parent = parent_node;
    node->left = left_node;
    node->right = right_node;


    node->idx = long_node_ptr->w;
    if (is_leaf) {
        triangle_ids[to_idx] = long_node_ptr->w;
    }

    return;
}

template<typename T>
__global__ void
set_rightmost_nodes(BVHNodePtr<T> internal_nodes, BVHNodePtr<T> leaf_nodes, int num_triangles, int *atomic_counters) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_triangles)
        return;

    BVHNodePtr<T> leaf = leaf_nodes + idx;
    leaf->rightmost = leaf;

    BVHNodePtr<T> curr_node = leaf->parent;
    int current_idx = curr_node - internal_nodes;
    int curr_counter = atomicAdd(atomic_counters + current_idx, 1);

    while (true) {
        // atomicAdd returns the old value at the specified address. Thus the
        // first thread to reach this point will immediately return
        if (curr_counter == 0)
            break;

        curr_node->rightmost =
                curr_node->left->rightmost > curr_node->right->rightmost
                ? curr_node->left->rightmost
                : curr_node->right->rightmost;

        // If we have reached the root break
        if (curr_node == internal_nodes)
            break;

        // Proceed to the parent of the node
        curr_node = curr_node->parent;
        // Calculate its position in the flat array
        current_idx = curr_node - internal_nodes;
        // Update the visitation counter
        curr_counter = atomicAdd(atomic_counters + current_idx, 1);
    }
}

template<typename T>
void
reconstruct_bvh(AABB<T> *bboxes, long4 *tree, BVHNodePtr<T> internal_nodes, BVHNodePtr<T> leaf_nodes,
                int *triangle_ids,
                int num_nodes, int num_triangles) {

    int blockSize = NUM_THREADS;
    int gridSize = (num_nodes + blockSize - 1) / blockSize;
    build_bvh_from_array_tree<T><<<gridSize, blockSize>>>(bboxes, tree,
                                                          internal_nodes,
                                                          leaf_nodes, triangle_ids, num_nodes,
                                                          num_triangles);

    gridSize = (num_triangles + blockSize - 1) / blockSize;
    thrust::device_vector<int> counters(num_triangles);
    set_rightmost_nodes<T><<<gridSize, blockSize>>>(internal_nodes,
                                                    leaf_nodes, num_triangles,
                                                    counters.data().get());
}