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
#import "flags.h"

std::ostream &operator<<(std::ostream &os, const vec3<float> &x) {
    os << x.x << ", " << x.y << ", " << x.z;
    return os;
}


std::ostream &operator<<(std::ostream &os, const vec3<double> &x) {
    os << x.x << ", " << x.y << ", " << x.z;
    return os;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const vec3 <T> &x) {
    os << x.x << ", " << x.y << ", " << x.z;
    return os;
}


template<typename T>
std::ostream &operator<<(std::ostream &os, vec3 <T> x) {
    os << x.x << ", " << x.y << ", " << x.z;
    return os;
}

__host__ __device__ inline double3 fmin(const double3 &a, const double3 &b) {
    return make_double3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
}

__host__ __device__ inline double3 fmax(const double3 &a, const double3 &b) {
    return make_double3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));
}

template<typename T>
__host__ __device__ __forceinline__ float vec_abs_diff(const vec3 <T> &vec1,
                                                       const vec3 <T> &vec2) {
    return fabs(vec1.x - vec2.x) + fabs(vec1.y - vec2.y) + fabs(vec1.z - vec2.z);
}

template<typename T>
__host__ __device__ __forceinline__ float vec_sq_diff(const vec3 <T> &vec1,
                                                      const vec3 <T> &vec2) {
    return dot(vec1 - vec2, vec1 - vec2);
}


template<typename T>
__host__ __device__
vec3 <T> operator*(const vec3 <T> &vec, T mult) {
    vec3 <T> out;
    out.x = vec.x * mult;
    out.y = vec.y * mult;
    out.z = vec.z * mult;
    return out;
}

__host__ __device__ inline double norm2(const double3 &a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__host__ __device__ inline float norm2(const float3 &a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

struct is_valid_cnt : public thrust::unary_function<long2, int> {
public:
    __host__ __device__ int operator()(long2 vec) const {
        return vec.x >= 0 && vec.y >= 0;
    }
};

inline __device__ double3 normalize(double3 a) {
    return a*rsqrt(a.x*a.x+a.y*a.y+a.z*a.z);
}