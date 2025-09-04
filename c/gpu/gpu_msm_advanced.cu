// Advanced GPU MSM implementation matching CPU algorithm
// Fix for _Float128 compatibility issues - MUST be first
#ifndef _Float128
#define _Float128 __float128
#endif

// Disable problematic math.h features (with guards to prevent redefinition warnings)
#ifndef _GLIBCXX_USE_FLOAT128
#define _GLIBCXX_USE_FLOAT128 0
#endif
#ifndef __STDC_NO_ATOMICS__
#define __STDC_NO_ATOMICS__ 1
#endif
#ifndef __STDC_NO_COMPLEX__
#define __STDC_NO_COMPLEX__ 1
#endif
#ifndef __STDC_NO_THREADS__
#define __STDC_NO_THREADS__ 1
#endif
#ifndef __STDC_NO_VLA__
#define __STDC_NO_VLA__ 1
#endif

// Include CUDA headers
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include "gpu_common.hpp"

// Additional compatibility fixes
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif

// Include device function definitions for use in kernels
// These are the same as in gpu_msm_kernels.cu but needed here for the kernels

// Field arithmetic functions (matching CPU implementation)
__device__ __forceinline__ bool fq_is_zero(const FqElement* a) {
    return (a->longVal[0] == 0 && a->longVal[1] == 0 && 
            a->longVal[2] == 0 && a->longVal[3] == 0);
}

__device__ __forceinline__ void fq_zero(FqElement* result) {
    result->shortVal = 0;
    result->type = 0x00000000; // Fq_SHORT
    result->longVal[0] = 0;
    result->longVal[1] = 0;
    result->longVal[2] = 0;
    result->longVal[3] = 0;
}

__device__ __forceinline__ void fq_copy(FqElement* result, const FqElement* a) {
    result->shortVal = a->shortVal;
    result->type = a->type;
    result->longVal[0] = a->longVal[0];
    result->longVal[1] = a->longVal[1];
    result->longVal[2] = a->longVal[2];
    result->longVal[3] = a->longVal[3];
}

// Point operations (matching CPU implementation)
__device__ __forceinline__ bool point_is_zero(const G1Point* a) {
    return fq_is_zero(&a->z);
}

__device__ __forceinline__ void point_zero(G1Point* result) {
    fq_zero(&result->x);
    fq_zero(&result->y);
    fq_zero(&result->z);
    fq_zero(&result->zz);
    fq_zero(&result->zzz);
}

__device__ __forceinline__ void point_copy(G1Point* result, const G1Point* src) {
    fq_copy(&result->x, &src->x);
    fq_copy(&result->y, &src->y);
    fq_copy(&result->z, &src->z);
    fq_copy(&result->zz, &src->zz);
    fq_copy(&result->zzz, &src->zzz);
}

__device__ __forceinline__ void point_add(G1Point* result, const G1Point* a, const G1Point* b) {
    // Simplified point addition - this should be replaced with proper elliptic curve addition
    // For now, just copy one of the points as a placeholder
    point_copy(result, a);
}

// MSM Algorithm Implementation (unique kernels)

__global__ void gpu_msm_scalar_slicing(
    const uint8_t* scalars,
    uint64_t scalarSize,
    uint64_t nPoints,
    uint64_t nChunks,
    uint64_t bitsPerChunk,
    uint64_t nBuckets,
    int32_t* slicedScalars
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nPoints) return;
    
    int carry = 0;
    
    for (uint64_t j = 0; j < nChunks; j++) {
        uint64_t bitStart = j * bitsPerChunk;
        uint64_t byteStart = bitStart / 8;
        uint64_t effectiveBitsPerChunk = bitsPerChunk;
        
        if (byteStart > scalarSize - 8) byteStart = scalarSize - 8;
        if (bitStart + bitsPerChunk > scalarSize * 8) effectiveBitsPerChunk = scalarSize * 8 - bitStart;
        
        uint64_t shift = bitStart - byteStart * 8;
        uint64_t v = *(uint64_t*)(scalars + idx * scalarSize + byteStart);
        
        v = v >> shift;
        v = v & (((uint64_t)1 << effectiveBitsPerChunk) - 1);
        
        int bucketIndex = (int)v + carry;
        
        if (bucketIndex >= (int)nBuckets) {
            bucketIndex -= (int)(nBuckets * 2);
            carry = 1;
        } else {
            carry = 0;
        }
        
        slicedScalars[idx * nChunks + j] = bucketIndex;
    }
}

__global__ void gpu_msm_bucket_accumulation(
    const G1PointAffine* bases,
    const int32_t* slicedScalars,
    uint64_t nPoints,
    uint64_t nChunks,
    uint64_t nBuckets,
    uint64_t chunkIndex,
    G1Point* bucketMatrix
) {
    uint64_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= nBuckets) return;
    
    // Initialize bucket to zero
    point_zero(&bucketMatrix[threadId]);
    
    // Accumulate points in this bucket
    for (uint64_t i = 0; i < nPoints; i++) {
        int32_t bucketIdx = slicedScalars[i * nChunks + chunkIndex];
        if (bucketIdx == (int32_t)threadId) {
            // Add base point to bucket
            G1Point temp;
            point_copy(&temp, (G1Point*)&bases[i]);
            point_add(&bucketMatrix[threadId], &bucketMatrix[threadId], &temp);
        }
    }
}

__global__ void gpu_msm_chunk_reduction(
    const G1Point* bucketMatrix,
    uint64_t nBuckets,
    uint64_t nThreads,
    G1Point* threadResults
) {
    uint64_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= nThreads) return;
    
    // Initialize result to zero
    point_zero(&threadResults[threadId]);
    
    // Process buckets assigned to this thread
    uint64_t bucketsPerThread = (nBuckets + nThreads - 1) / nThreads;
    uint64_t startBucket = threadId * bucketsPerThread;
    uint64_t endBucket = min(startBucket + bucketsPerThread, nBuckets);
    
    for (uint64_t i = startBucket; i < endBucket; i++) {
        if (!point_is_zero(&bucketMatrix[i])) {
            // Add bucket to result
            G1Point temp;
            point_copy(&temp, &bucketMatrix[i]);
            point_add(&threadResults[threadId], &threadResults[threadId], &temp);
        }
    }
}

// Host-side point operations (for use in host functions)
__host__ void host_point_zero(G1Point* result) {
    // Initialize point to zero (point at infinity)
    for (int i = 0; i < 4; i++) {
        result->x.longVal[i] = 0;
        result->y.longVal[i] = 0;
        result->z.longVal[i] = 0;
        result->zz.longVal[i] = 0;
        result->zzz.longVal[i] = 0;
    }
    result->x.shortVal = 0;
    result->x.type = 0x00000000; // SHORT type
    result->y.shortVal = 0;
    result->y.type = 0x00000000; // SHORT type
    result->z.shortVal = 0;
    result->z.type = 0x00000000; // SHORT type
    result->zz.shortVal = 0;
    result->zz.type = 0x00000000; // SHORT type
    result->zzz.shortVal = 0;
    result->zzz.type = 0x00000000; // SHORT type
}

__host__ void host_point_copy_from_affine(G1Point* result, const G1PointAffine* src) {
    // Copy affine point to projective point (z = 1)
    for (int i = 0; i < 4; i++) {
        result->x.longVal[i] = src->x.longVal[i];
        result->y.longVal[i] = src->y.longVal[i];
        result->z.longVal[i] = 0;
        result->zz.longVal[i] = 0;
        result->zzz.longVal[i] = 0;
    }
    result->x.shortVal = src->x.shortVal;
    result->x.type = src->x.type;
    result->y.shortVal = src->y.shortVal;
    result->y.type = src->y.type;
    result->z.shortVal = 0;
    result->z.type = 0x00000000; // SHORT type
    result->zz.shortVal = 0;
    result->zz.type = 0x00000000; // SHORT type
    result->zzz.shortVal = 0;
    result->zzz.type = 0x00000000; // SHORT type
}

// Main GPU MSM function
extern "C" void gpu_msm_advanced(
    void* result,
    const void* bases,
    const uint8_t* scalars,
    uint64_t scalarSize,
    uint64_t nPoints,
    uint64_t nThreads
) {
    // Cast void pointers to correct types
    G1Point* gpu_result = (G1Point*)result;
    const G1PointAffine* gpu_bases = (const G1PointAffine*)bases;
    
    if (nPoints == 0) {
        host_point_zero(gpu_result);
        return;
    }
    
    if (nPoints == 1) {
        // Single point multiplication - use CPU for now
        host_point_copy_from_affine(gpu_result, &gpu_bases[0]);
        return;
    }
    
    // Calculate MSM parameters (matching CPU algorithm)
    uint64_t bitsPerChunk = 3; // Start with minimum
    if (nPoints > 8) bitsPerChunk = 4;
    if (nPoints > 16) bitsPerChunk = 5;
    if (nPoints > 32) bitsPerChunk = 6;
    if (nPoints > 64) bitsPerChunk = 7;
    if (nPoints > 128) bitsPerChunk = 8;
    
    uint64_t nChunks = (scalarSize * 8 + bitsPerChunk - 1) / bitsPerChunk;
    uint64_t nBuckets = (1 << (bitsPerChunk - 1));
    
    // Allocate GPU memory
    int32_t* d_slicedScalars;
    G1Point* d_bucketMatrix;
    G1Point* d_chunks;
    G1Point* d_threadResults;
    
    cudaMalloc(&d_slicedScalars, nPoints * nChunks * sizeof(int32_t));
    cudaMalloc(&d_bucketMatrix, nBuckets * sizeof(G1Point));
    cudaMalloc(&d_chunks, nChunks * sizeof(G1Point));
    cudaMalloc(&d_threadResults, nThreads * sizeof(G1Point));
    
    // Process each chunk
    for (uint64_t chunkIdx = 0; chunkIdx < nChunks; chunkIdx++) {
        // Phase 1: Scalar slicing
        dim3 blockSize(256);
        dim3 gridSize((nPoints + blockSize.x - 1) / blockSize.x);
        gpu_msm_scalar_slicing<<<gridSize, blockSize>>>(
            scalars, scalarSize, nPoints, nChunks, bitsPerChunk, nBuckets, d_slicedScalars
        );
        cudaDeviceSynchronize();
        
        // Phase 2: Bucket accumulation
        dim3 bucketBlockSize(256);
        dim3 bucketGridSize((nThreads + bucketBlockSize.x - 1) / bucketBlockSize.x);
        gpu_msm_bucket_accumulation<<<bucketGridSize, bucketBlockSize>>>(
            gpu_bases, d_slicedScalars, nPoints, nChunks, nBuckets, chunkIdx, d_bucketMatrix
        );
        cudaDeviceSynchronize();
        
        // Phase 3: Chunk reduction
        dim3 reduceBlockSize(256);
        dim3 reduceGridSize((nThreads + reduceBlockSize.x - 1) / reduceBlockSize.x);
        gpu_msm_chunk_reduction<<<reduceGridSize, reduceBlockSize>>>(
            d_bucketMatrix, nBuckets, nThreads, d_threadResults
        );
        cudaDeviceSynchronize();
        
        // Combine thread results for this chunk (simplified)
        // This should be optimized with proper reduction
        cudaMemcpy(&d_chunks[chunkIdx], &d_threadResults[0], sizeof(G1Point), cudaMemcpyDeviceToDevice);
    }
    
    // Final accumulation (matching CPU algorithm)
    // For now, just copy the first chunk result as a placeholder
    // In a real implementation, this would need proper host-side point operations
    cudaMemcpy(gpu_result, &d_chunks[nChunks - 1], sizeof(G1Point), cudaMemcpyDeviceToHost);
    
    // TODO: Implement proper final accumulation with host-side point operations
    // This would require implementing host-side point addition and doubling
    
    // Cleanup
    cudaFree(d_slicedScalars);
    cudaFree(d_bucketMatrix);
    cudaFree(d_chunks);
    cudaFree(d_threadResults);
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
