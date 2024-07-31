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

#include <torch/extension.h>

#include <vector>
#include <cmath>

#define AT_CHECK TORCH_CHECK

void build_bvh(at::Tensor triangles, at::Tensor *bboxes_tensor_ptr, at::Tensor *tree_tensor_ptr);

void build_bvh(at::Tensor triangles, at::Tensor triangles_next, at::Tensor *bboxes_tensor_ptr,
               at::Tensor *tree_tensor_ptr);

void find_collisions_from_bbox_tree(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                    at::Tensor *collision_tensor_ptr, float threshold,
                                    int max_collisions);

void find_collisions_from_bbox_tree(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                    at::Tensor triangles_next_tensor,
                                    at::Tensor *collision_tensor_ptr,
                                    at::Tensor *time_tensor_ptr,
                                    int max_candidates_per_triangle,
                                    int max_collisions_per_triangle);

void find_minimal_candidates_from_bbox_tree(at::Tensor bboxes_tensor, at::Tensor tree_tensor,
                                            at::Tensor triangles_tensor,
                                            at::Tensor triangles_next_tensor,
                                            at::Tensor *collision_tensor_ptr,
                                            int max_candidates_per_triangle);

void find_proximity_from_bbox_tree(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                   at::Tensor *collision_tensor_ptr,
                                   int max_collisions, float threshold);

void collision_impulses_from_bbox_tree(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                       at::Tensor triangles_next_tensor, at::Tensor triangles_mass_tensor,
                                       at::Tensor *collision_tensor_ptr, at::Tensor *time_tensor_ptr,
                                       at::Tensor *impulses_tensor_ptr,
                                       int max_candidates_per_triangle,
                                       int max_collisions_per_triangle);

void find_collisions_from_bbox_tree_partial(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                            at::Tensor triangles_next_tensor,
                                            at::Tensor triangles_to_check_tensor,
                                            at::Tensor *collision_tensor_ptr, at::Tensor *time_tensor_ptr,
                                            int max_candidates_per_triangle,
                                            int max_collisions_per_triangle);


void collision_impulses_from_continuous_collisions(at::Tensor collision_tensor, at::Tensor roots_tensor, at::Tensor triangles_tensor,
                                                   at::Tensor triangles_next_tensor,
                                                   at::Tensor triangles_mass_tensor,
                                                   at::Tensor *impulses_dv_tensor_ptr, at::Tensor *impulses_dx_tensor_ptr,
                                                   at::Tensor *impulses_counter_tensor_ptr);


void collision_impulses_from_bbox_tree_partial(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                               at::Tensor triangles_next_tensor,
                                               at::Tensor triangles_mass_tensor,
                                               at::Tensor triangles_to_check_tensor,
                                               at::Tensor *impulses_dv_tensor_ptr, at::Tensor *impulses_dx_tensor_ptr,
                                               at::Tensor *impulses_counter_tensor_ptr,
                                               int max_candidates_per_triangle,
                                               int max_collisions_per_triangle);


void find_triangle_edge_collisions_from_bbox_tree(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                    at::Tensor *collision_tensor_ptr, at::Tensor *coeff_tensor_ptr,
                                    int max_collisions);


void find_triangle_edge_candidates(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                    at::Tensor *collision_tensor_ptr,
                                    int max_collisions);

void find_triangle_edge_candidates2(at::Tensor bboxes_tensor, at::Tensor tree_tensor, at::Tensor triangles_tensor,
                                    at::Tensor *collision_tensor_ptr, at::Tensor *binmasks_ptr,
                                    int max_collisions);



#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector <torch::Tensor> bvh_forward(at::Tensor triangles) {
    CHECK_INPUT(triangles);

    at::Tensor bboxes_tensor = -1 * at::ones({triangles.size(0),
                                              triangles.size(1) * 2 - 1, 6},
                                             at::device(triangles.device()).dtype(triangles.dtype()));

    at::Tensor tree_tensor = -1 * at::ones({triangles.size(0),
                                            triangles.size(1) * 2 - 1, 4},
                                           at::device(triangles.device()).dtype(at::kLong));

    build_bvh(triangles, &bboxes_tensor, &tree_tensor);

    return {torch::autograd::make_variable(bboxes_tensor, false), torch::autograd::make_variable(tree_tensor, false)};
}

std::vector <torch::Tensor> bvh_forward_motion(at::Tensor triangles, at::Tensor triangles_next) {
    CHECK_INPUT(triangles);

    at::Tensor bboxes_tensor = -1 * at::ones({triangles.size(0),
                                              triangles.size(1) * 2 - 1, 6},
                                             at::device(triangles.device()).dtype(triangles.dtype()));

    at::Tensor tree_tensor = -1 * at::ones({triangles.size(0),
                                            triangles.size(1) * 2 - 1, 4},
                                           at::device(triangles.device()).dtype(at::kLong));
//
    build_bvh(triangles, triangles_next, &bboxes_tensor, &tree_tensor);

    return {torch::autograd::make_variable(bboxes_tensor, false), torch::autograd::make_variable(tree_tensor, false)};
}

at::Tensor find_collisions(at::Tensor bboxes, at::Tensor tree, at::Tensor triangles, float threshold=0., int max_collisions = 16) {
    CHECK_INPUT(bboxes);
    CHECK_INPUT(tree);
    CHECK_INPUT(triangles);

    int num_triangles = (bboxes.size(1) + 1) / 2;

    at::Tensor collisionTensor = -1 * at::ones({bboxes.size(0),
                                                num_triangles * max_collisions, 2},
                                               at::device(bboxes.device()).dtype(at::kLong));

    find_collisions_from_bbox_tree(bboxes, tree, triangles, &collisionTensor, threshold, max_collisions);

    return torch::autograd::make_variable(collisionTensor, false);
}


std::vector <torch::Tensor> find_triangle_edge_collisions(at::Tensor bboxes, at::Tensor tree, at::Tensor triangles, float threshold=0., int max_collisions = 16) {
    CHECK_INPUT(bboxes);
    CHECK_INPUT(tree);
    CHECK_INPUT(triangles);

    int num_triangles = (bboxes.size(1) + 1) / 2;

    at::Tensor collisionTensor = -1 * at::ones({bboxes.size(0),
                                                num_triangles * max_collisions, 4},
                                               at::device(bboxes.device()).dtype(at::kLong));


    at::Tensor coeffsTensor = -1 * at::ones({bboxes.size(0),
                                           num_triangles * max_collisions, 1},
                                          at::device(bboxes.device()).dtype(bboxes.dtype()));


    find_triangle_edge_collisions_from_bbox_tree(bboxes, tree, triangles,
                                    &collisionTensor, &coeffsTensor,max_collisions);


    return {torch::autograd::make_variable(collisionTensor, false), torch::autograd::make_variable(coeffsTensor, false)};
}


torch::Tensor find_triangle_edge_candidates_cpp(at::Tensor bboxes, at::Tensor tree, at::Tensor triangles, float threshold=0., int max_collisions = 16) {
    CHECK_INPUT(bboxes);
    CHECK_INPUT(tree);
    CHECK_INPUT(triangles);

    int num_triangles = (bboxes.size(1) + 1) / 2;

    at::Tensor collisionTensor = -1 * at::ones({bboxes.size(0),
                                                num_triangles * max_collisions, 4},
                                               at::device(bboxes.device()).dtype(at::kLong));



    find_triangle_edge_candidates(bboxes, tree, triangles,
                                    &collisionTensor,max_collisions);


    return torch::autograd::make_variable(collisionTensor, false);
}


std::vector <torch::Tensor> find_triangle_edge_candidates2_cpp(at::Tensor bboxes, at::Tensor tree, at::Tensor triangles, float threshold=0., int max_collisions = 16) {
    CHECK_INPUT(bboxes);
    CHECK_INPUT(tree);
    CHECK_INPUT(triangles);

    int num_triangles = (bboxes.size(1) + 1) / 2;

    at::Tensor collisionTensor = -1 * at::ones({bboxes.size(0),
                                                num_triangles * max_collisions, 2},
                                               at::device(bboxes.device()).dtype(at::kLong));
    at::Tensor binmaskTensor = at::zeros({bboxes.size(0),
                                    num_triangles * max_collisions, 2},
                                    at::device(bboxes.device()).dtype(at::kInt));



    find_triangle_edge_candidates2(bboxes, tree, triangles,
                                    &collisionTensor, &binmaskTensor, max_collisions);


    return {torch::autograd::make_variable(collisionTensor, false), torch::autograd::make_variable(binmaskTensor, false)};
}

std::vector <torch::Tensor>
find_collisions_continuous(at::Tensor bboxes, at::Tensor tree, at::Tensor triangles, at::Tensor triangles_next,
                           int max_candidates_per_triangle = 16, int max_collisions_per_triangle = 16) {
    CHECK_INPUT(bboxes);
    CHECK_INPUT(tree);
    CHECK_INPUT(triangles);
    CHECK_INPUT(triangles_next);


    int num_triangles = (bboxes.size(1) + 1) / 2;

    at::Tensor collisionTensor = -1 * at::ones({bboxes.size(0),
                                                num_triangles * max_collisions_per_triangle, 3},
                                               at::device(bboxes.device()).dtype(at::kLong));

    at::Tensor timeTensor = -1 * at::ones({bboxes.size(0),
                                           num_triangles * max_collisions_per_triangle, 1},
                                          at::device(bboxes.device()).dtype(bboxes.dtype()));

    find_collisions_from_bbox_tree(bboxes, tree, triangles,
                                   triangles_next, &collisionTensor, &timeTensor, max_candidates_per_triangle,
                                   max_collisions_per_triangle);


    return {torch::autograd::make_variable(collisionTensor, false), torch::autograd::make_variable(timeTensor, false)};
}

std::vector <torch::Tensor>
find_collisions_continuous_partial(at::Tensor bboxes, at::Tensor tree, at::Tensor triangles, at::Tensor triangles_next, at::Tensor triangles_to_check,
                           int max_candidates_per_triangle = 16, int max_collisions_per_triangle = 16) {
    CHECK_INPUT(bboxes);
    CHECK_INPUT(tree);
    CHECK_INPUT(triangles);
    CHECK_INPUT(triangles_next);
    CHECK_INPUT(triangles_to_check);


    int num_triangles = (bboxes.size(1) + 1) / 2;

    at::Tensor collisionTensor = -1 * at::ones({bboxes.size(0),
                                                num_triangles * max_collisions_per_triangle, 3},
                                               at::device(bboxes.device()).dtype(at::kLong));

    at::Tensor timeTensor = -1 * at::ones({bboxes.size(0),
                                           num_triangles * max_collisions_per_triangle, 1},
                                          at::device(bboxes.device()).dtype(bboxes.dtype()));

    find_collisions_from_bbox_tree_partial(bboxes, tree, triangles,
                                   triangles_next, triangles_to_check, &collisionTensor, &timeTensor, max_candidates_per_triangle,
                                   max_collisions_per_triangle);


    return {torch::autograd::make_variable(collisionTensor, false), torch::autograd::make_variable(timeTensor, false)};
}


std::vector <torch::Tensor>
find_proximity(at::Tensor bboxes, at::Tensor tree, at::Tensor triangles, float threshold,
               int max_candidates_per_triangle = 16) {
//find_proximity(at::Tensor bboxes, at::Tensor tree, at::Tensor triangles) {
    CHECK_INPUT(bboxes);
    CHECK_INPUT(tree);
    CHECK_INPUT(triangles);

    int num_triangles = (bboxes.size(1) + 1) / 2;

    at::Tensor collisionTensor = -1 * at::ones({bboxes.size(0),
                                                num_triangles * max_candidates_per_triangle, 3},
                                               at::device(bboxes.device()).dtype(at::kLong));

    find_proximity_from_bbox_tree(bboxes, tree, triangles,
                                  &collisionTensor, max_candidates_per_triangle,
                                  threshold);


    return {torch::autograd::make_variable(collisionTensor, false)};
}


std::vector <torch::Tensor>
compute_impulses(at::Tensor bboxes, at::Tensor tree, at::Tensor triangles, at::Tensor triangles_next, at::Tensor triangles_masses,
                 int max_candidates_per_triangle = 16, int max_collisions_per_triangle = 16) {

    CHECK_INPUT(bboxes);
    CHECK_INPUT(tree);
    CHECK_INPUT(triangles);
    CHECK_INPUT(triangles_next);
    CHECK_INPUT(triangles_masses);

    int num_triangles = (bboxes.size(1) + 1) / 2;

    at::Tensor impulsesDvTensor = at::zeros({bboxes.size(0),
                                             num_triangles, 3, 3},
                                            at::device(bboxes.device()).dtype(bboxes.dtype()));

    at::Tensor impulsesDxTensor = at::zeros({bboxes.size(0),
                                             num_triangles, 3, 3},
                                            at::device(bboxes.device()).dtype(bboxes.dtype()));

    at::Tensor impulsesCounterTensor = at::zeros({bboxes.size(0),
                                                  num_triangles, 3},
                                                 at::device(bboxes.device()).dtype(at::kInt));

    collision_impulses_from_bbox_tree(bboxes, tree, triangles, triangles_next, triangles_masses,
                                      &impulsesDvTensor, &impulsesDxTensor, &impulsesCounterTensor,
                                      max_candidates_per_triangle,
                                      max_collisions_per_triangle);
    fflush(stdout);


    return {torch::autograd::make_variable(impulsesDvTensor, false),
            torch::autograd::make_variable(impulsesDxTensor, false),
            torch::autograd::make_variable(impulsesCounterTensor, false)};
}


std::vector <torch::Tensor>
compute_impulses_partial(at::Tensor bboxes, at::Tensor tree, at::Tensor triangles, at::Tensor triangles_next, at::Tensor triangles_masses, at::Tensor triangles_to_check,
                 int max_candidates_per_triangle = 16, int max_collisions_per_triangle = 16) {

    CHECK_INPUT(bboxes);
    CHECK_INPUT(tree);
    CHECK_INPUT(triangles);
    CHECK_INPUT(triangles_next);
    CHECK_INPUT(triangles_masses);

    int num_triangles = (bboxes.size(1) + 1) / 2;

    at::Tensor impulsesDvTensor = at::zeros({bboxes.size(0),
                                             num_triangles, 3, 3},
                                            at::device(bboxes.device()).dtype(bboxes.dtype()));

    at::Tensor impulsesDxTensor = at::zeros({bboxes.size(0),
                                             num_triangles, 3, 3},
                                            at::device(bboxes.device()).dtype(bboxes.dtype()));

    at::Tensor impulsesCounterTensor = at::zeros({bboxes.size(0),
                                                  num_triangles, 3},
                                                 at::device(bboxes.device()).dtype(at::kInt));

    collision_impulses_from_bbox_tree_partial(bboxes, tree, triangles, triangles_next, triangles_masses, triangles_to_check,
                                      &impulsesDvTensor, &impulsesDxTensor, &impulsesCounterTensor,
                                      max_candidates_per_triangle,
                                      max_collisions_per_triangle);
    fflush(stdout);


    return {torch::autograd::make_variable(impulsesDvTensor, false),
            torch::autograd::make_variable(impulsesDxTensor, false),
            torch::autograd::make_variable(impulsesCounterTensor, false)};
}



std::vector <torch::Tensor>
compute_impulses_from_collisions(at::Tensor collisions, at::Tensor roots, at::Tensor triangles, at::Tensor triangles_next, at::Tensor triangles_masses) {
    CHECK_INPUT(collisions);
    CHECK_INPUT(roots);
    CHECK_INPUT(triangles);
    CHECK_INPUT(triangles_next);
    CHECK_INPUT(triangles_masses);

    int num_triangles = triangles.size(1);

    at::Tensor impulsesDvTensor = at::zeros({triangles.size(0),
                                             num_triangles, 3, 3},
                                            at::device(triangles.device()).dtype(triangles.dtype()));

    at::Tensor impulsesDxTensor = at::zeros({triangles.size(0),
                                             num_triangles, 3, 3},
                                            at::device(triangles.device()).dtype(triangles.dtype()));

    at::Tensor impulsesCounterTensor = at::zeros({triangles.size(0),
                                                  num_triangles, 3},
                                                 at::device(triangles.device()).dtype(at::kInt));

    collision_impulses_from_continuous_collisions(collisions, roots, triangles, triangles_next, triangles_masses,
                                      &impulsesDvTensor, &impulsesDxTensor, &impulsesCounterTensor);


    return {torch::autograd::make_variable(impulsesDvTensor, false),
            torch::autograd::make_variable(impulsesDxTensor, false),
            torch::autograd::make_variable(impulsesCounterTensor, false)};
}

torch::Tensor
find_mincol_cand(at::Tensor bboxes, at::Tensor tree, at::Tensor triangles, at::Tensor triangles_next,
                 int max_collisions = 16) {
    CHECK_INPUT(bboxes);
    CHECK_INPUT(tree);
    CHECK_INPUT(triangles);
    CHECK_INPUT(triangles_next);

    int num_triangles = (bboxes.size(1) + 1) / 2;

    at::Tensor collisionTensor = -1 * at::ones({bboxes.size(0),
                                                num_triangles * max_collisions * 15, 3},
                                               at::device(bboxes.device()).dtype(at::kLong));
    find_minimal_candidates_from_bbox_tree(bboxes, tree, triangles,
                                           triangles_next, &collisionTensor, max_collisions);


    return torch::autograd::make_variable(collisionTensor, false);
}
//

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("bvh", &bvh_forward, "BVH collision forward (CUDA)",py::arg("triangles"));
m.def("bvh_motion", &bvh_forward_motion, "BVH collision forward (CUDA)",py::arg("triangles"),py::arg("triangles_next"));

m.def("find_collisions", &find_collisions, "",
py::arg("bboxes"), py::arg("tree"), py::arg("triangles"), py::arg("threshold") = 0., py::arg("max_collisions") = 16);

m.def("find_triangle_edge_collisions", &find_triangle_edge_collisions, "",
py::arg("bboxes"), py::arg("tree"), py::arg("triangles"), py::arg("threshold") = 0., py::arg("max_collisions") = 16);

m.def("find_triangle_edge_candidates", &find_triangle_edge_candidates_cpp, "",
py::arg("bboxes"), py::arg("tree"), py::arg("triangles"), py::arg("threshold") = 0., py::arg("max_collisions") = 16);

m.def("find_triangle_edge_candidates2", &find_triangle_edge_candidates2_cpp, "",
py::arg("bboxes"), py::arg("tree"), py::arg("triangles"), py::arg("threshold") = 0., py::arg("max_collisions") = 16);


m.def("find_collisions_continuous", &find_collisions_continuous, "",
py::arg("bboxes"), py::arg("tree"), py::arg("triangles"), py::arg("triangles_next"), py::arg("max_candidates_per_triangle") = 16, py::arg("max_collisions_per_triangle") = 16);

m.def("find_collisions_continuous_partial", &find_collisions_continuous_partial, "",
py::arg("bboxes"), py::arg("tree"), py::arg("triangles"), py::arg("triangles_next"), py::arg("triangles_to_check"), py::arg("max_candidates_per_triangle") = 16, py::arg("max_collisions_per_triangle") = 16);


m.def("find_proximity", &find_proximity, "",
py::arg("bboxes"), py::arg("tree"), py::arg("triangles"), py::arg("threshold"), py::arg("max_candidates_per_triangle") = 16);


m.def("find_mincol_cand", &find_mincol_cand, "",
py::arg("bboxes"), py::arg("tree"), py::arg("triangles"), py::arg("triangles_next"), py::arg("max_collisions") = 16);

m.def("compute_impulses", &compute_impulses, "",
py::arg("bboxes"), py::arg("tree"), py::arg("triangles"), py::arg("triangles_next"),  py::arg("triangles_mass"), py::arg("max_candidates_per_triangle") = 16, py::arg("max_collisions_per_triangle") = 16);


m.def("compute_impulses_partial", &compute_impulses_partial, "",
py::arg("bboxes"), py::arg("tree"), py::arg("triangles"), py::arg("triangles_next"),  py::arg("triangles_mass"),  py::arg("triangles_to_check"), py::arg("max_candidates_per_triangle") = 16, py::arg("max_collisions_per_triangle") = 16);


m.def("compute_impulses_from_collisions", &compute_impulses_from_collisions, "",
py::arg("collisions"), py::arg("roots"), py::arg("triangles"), py::arg("triangles_next"),  py::arg("triangles_mass"));
}
