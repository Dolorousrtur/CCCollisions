#pragma once

#include <ATen/ATen.h>


#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <vector>
#include <iostream>
#include <string>
#include <type_traits>

#include "../include/double_vec_ops.h"
#include "helper_math.h"

// Size of the stack used to traverse the Bounding Volume Hierarchy tree
#ifndef STACK_SIZE
#define STACK_SIZE 64
#endif /* ifndef STACK_SIZE */

// Upper bound for the number of possible collisions
#ifndef MAX_COLLISIONS
#define MAX_COLLISIONS 16
#endif

#ifndef EPSILON
#define EPSILON 1e-16
#endif /* ifndef EPSILON */

// Number of threads per block for CUDA kernel launch
#ifndef NUM_THREADS
#define NUM_THREADS 128
#endif

#ifndef COLLISION_ORDERING
#define COLLISION_ORDERING 1
#endif

#ifndef FORCE_INLINE
#define FORCE_INLINE 1
#endif /* ifndef FORCE_INLINE */

#ifndef ERROR_CHECKING
#define ERROR_CHECKING 1
#endif /* ifndef ERROR_CHECKING */

#ifndef SCALE
#define SCALE 1e+0
#endif



// Macro for checking cuda errors following a cuda launch or api call
#if ERROR_CHECKING == 1
#define cudaCheckError()                                                       \
  {                                                                            \
    cudaDeviceSynchronize();                                                   \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      exit(0);                                                                 \
    }                                                                          \
  }
#else
#define cudaCheckError()
#endif


typedef unsigned int MortonCode;

template<typename T>
using vec4 = typename std::conditional<std::is_same<T, float>::value, float4,
        double4>::type;

template<typename T>
using vec3 = typename std::conditional<std::is_same<T, float>::value, float3,
        double3>::type;

template<typename T>
using vec2 = typename std::conditional<std::is_same<T, float>::value, float2,
        double2>::type;
