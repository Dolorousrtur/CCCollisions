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

template<typename T>
struct AABB {
public:
    __host__ __device__ AABB() {
        min_t.x = std::is_same<T, float>::value ? -FLT_MAX : -DBL_MAX;
        min_t.y = std::is_same<T, float>::value ? -FLT_MAX : -DBL_MAX;
        min_t.z = std::is_same<T, float>::value ? -FLT_MAX : -DBL_MAX;

        max_t.x = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;
        max_t.y = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;
        max_t.z = std::is_same<T, float>::value ? FLT_MAX : DBL_MAX;
    };


    __host__ __device__ AABB(const vec3<T> &min_t, const vec3<T> &max_t)
            : min_t(min_t), max_t(max_t) {};

    __host__ __device__ ~AABB() {};

    __host__ __device__ AABB(T min_t_x, T min_t_y, T min_t_z, T max_t_x,
                             T max_t_y, T max_t_z) {
        min_t.x = min_t_x;
        min_t.y = min_t_y;
        min_t.z = min_t_z;
        max_t.x = max_t_x;
        max_t.y = max_t_y;
        max_t.z = max_t_z;
    }

    __host__ __device__ AABB<T> operator+(const AABB<T> &bbox2) const {
        return AABB<T>(
                min(this->min_t.x, bbox2.min_t.x), min(this->min_t.y, bbox2.min_t.y),
                min(this->min_t.z, bbox2.min_t.z), max(this->max_t.x, bbox2.max_t.x),
                max(this->max_t.y, bbox2.max_t.y), max(this->max_t.z, bbox2.max_t.z));
    };

    __host__ __device__ T operator*(const AABB<T> &bbox2) const {
        return (min(this->max_t.x, bbox2.max_t.x) -
                max(this->min_t.x, bbox2.min_t.x)) *
               (min(this->max_t.y, bbox2.max_t.y) -
                max(this->min_t.y, bbox2.min_t.y)) *
               (min(this->max_t.z, bbox2.max_t.z) -
                max(this->min_t.z, bbox2.min_t.z));
    };

    vec3<T> min_t;
    vec3<T> max_t;
};

template<typename T>
std::ostream &operator<<(std::ostream &os, const AABB<T> &x);

template<typename T>
struct MergeAABB {

public:
    __host__ __device__ MergeAABB() {};

    // Create an operator Struct that will be used by thrust::reduce
    // to calculate the bounding box of the scene.
    __host__ __device__ AABB<T> operator()(const AABB<T> &bbox1,
                                           const AABB<T> &bbox2);
};

template<typename T>
std::ostream &operator<<(std::ostream &os, const AABB<T> &x) {
    os << x.min_t << std::endl;
    os << x.max_t << std::endl;
    return os;
}


template<typename T>
__host__ __device__ AABB<T> MergeAABB<T>::operator()(const AABB<T> &bbox1,
                                                     const AABB<T> &bbox2) {
    return bbox1 + bbox2;
};


template<typename T>
__device__
#if FORCE_INLINE == 1
__forceinline__
#endif
bool
checkOverlap(const AABB<T> &bbox1, const AABB<T> &bbox2, T threshold = 0) {
    return (bbox1.min_t.x <= (bbox2.max_t.x + threshold)) && (bbox1.max_t.x >= (bbox2.min_t.x - threshold)) &&
           (bbox1.min_t.y <= (bbox2.max_t.y + threshold)) && (bbox1.max_t.y >= (bbox2.min_t.y - threshold)) &&
           (bbox1.min_t.z <= (bbox2.max_t.z + threshold)) && (bbox1.max_t.z >= (bbox2.min_t.z - threshold));
}