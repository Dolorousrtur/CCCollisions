#pragma once

#include "flags.h"
#include "aabb.cu"

template<typename T>
struct BVHNode {
public:
    AABB<T> bbox;

    BVHNode<T> *left;
    BVHNode<T> *right;
    BVHNode<T> *parent;
    // Stores the rightmost leaf node that can be reached from the current
    // node.
    BVHNode<T> *rightmost;

    __host__ __device__ inline bool isLeaf() { return !left && !right; };

    // The index of the object contained in the node
    int idx = -1;
};

template<typename T> using BVHNodePtr = BVHNode<T> *;


// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__
#if FORCE_INLINE == 1
__forceinline__
#endif
MortonCode
expandBits(MortonCode v) {
    // Shift 16
    v = (v * 0x00010001u) & 0xFF0000FFu;
    // Shift 8
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    // Shift 4
    v = (v * 0x00000011u) & 0xC30C30C3u;
    // Shift 2
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
template<typename T>
__device__
#if FORCE_INLINE == 1
__forceinline__
#endif
MortonCode
morton3D(T x, T y, T z) {
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    MortonCode xx = expandBits((MortonCode) x);
    MortonCode yy = expandBits((MortonCode) y);
    MortonCode zz = expandBits((MortonCode) z);
    return xx * 4 + yy * 2 + zz;
}


__device__
#if FORCE_INLINE == 1
__forceinline__
#endif
int
LongestCommonPrefix(int i, int j, MortonCode *morton_codes,
                    int num_triangles, int *triangle_ids) {
    // This function will be called for i - 1, i, i + 1, so we might go beyond
    // the array limits
    if (i < 0 || i > num_triangles - 1 || j < 0 || j > num_triangles - 1)
        return -1;

    MortonCode key1 = morton_codes[i];
    MortonCode key2 = morton_codes[j];

    if (key1 == key2) {
        // Duplicate key:__clzll(key1 ^ key2) will be equal to the number of
        // bits in key[1, 2]. Add the number of leading zeros between the
        // indices
        return __clz(key1 ^ key2) + __clz(triangle_ids[i] ^ triangle_ids[j]);
    } else {
        // Keys are different
        return __clz(key1 ^ key2);
    }
}
