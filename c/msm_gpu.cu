#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include "msm_gpu.hpp"
#include <iostream>
#include <chrono>

namespace MSM_GPU {

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        return false; \
    } \
} while(0)

// GPU MSM Context Implementation
GPUMSMContext::GPUMSMContext() : initialized(false), cudaContext(nullptr), 
                                computeStream(nullptr), startEvent(nullptr), endEvent(nullptr) {
    deviceInfo.available = false;
}

GPUMSMContext::~GPUMSMContext() {
    if (initialized) {
        if (computeStream) cudaStreamDestroy(*computeStream);
        if (startEvent) cudaEventDestroy(*startEvent);
        if (endEvent) cudaEventDestroy(*endEvent);
        delete computeStream;
        delete startEvent;
        delete endEvent;
    }
}

bool GPUMSMContext::initialize() {
    // Check CUDA availability
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }
    
    // Get device 0 (primary GPU)
    deviceInfo.deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceInfo.deviceId));
    
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceInfo.deviceId));
    
    deviceInfo.totalMemory = prop.totalGlobalMem;
    deviceInfo.computeCapability = prop.major * 10 + prop.minor;
    deviceInfo.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    deviceInfo.maxBlocksPerGrid = prop.maxGridSize[0];
    
    // Get free memory
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    deviceInfo.freeMemory = free;
    
    // Create CUDA stream and events
    computeStream = new cudaStream_t;
    startEvent = new cudaEvent_t;
    endEvent = new cudaEvent_t;
    
    CUDA_CHECK(cudaStreamCreate(computeStream));
    CUDA_CHECK(cudaEventCreate(startEvent));
    CUDA_CHECK(cudaEventCreate(endEvent));
    
    deviceInfo.available = true;
    initialized = true;
    
    std::cerr << "GPU initialized: " << prop.name 
              << " (CC " << deviceInfo.computeCapability 
              << ", " << (deviceInfo.totalMemory / (1024*1024*1024)) << " GB)" << std::endl;
    
    return true;
}

GPUDevice GPUMSMContext::getDeviceInfo() const {
    return deviceInfo;
}

bool GPUMSMContext::isAvailable() const {
    return deviceInfo.available;
}

void GPUMSMContext::getOptimalConfig(uint64_t nPoints, int& threadsPerBlock, int& blocksPerGrid) const {
    // Optimal configuration based on GPU capabilities
    if (deviceInfo.computeCapability >= 70) { // Volta/Turing/Ampere
        threadsPerBlock = 256;
        blocksPerGrid = (nPoints + threadsPerBlock - 1) / threadsPerBlock;
        if (blocksPerGrid > deviceInfo.maxBlocksPerGrid) {
            blocksPerGrid = deviceInfo.maxBlocksPerGrid;
        }
    } else {
        threadsPerBlock = 128;
        blocksPerGrid = (nPoints + threadsPerBlock - 1) / threadsPerBlock;
        if (blocksPerGrid > deviceInfo.maxBlocksPerGrid) {
            blocksPerGrid = deviceInfo.maxBlocksPerGrid;
        }
    }
}

// CUDA Kernel Implementations

// Scalar slicing kernel - processes scalars in parallel
__global__ void scalarSlicingKernel_kernel(int32_t* slicedScalars, uint8_t* scalars, 
                                          uint64_t nPoints, uint64_t nChunks, uint64_t nBuckets,
                                          uint64_t bitsPerChunk, uint64_t scalarSize) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nPoints) return;
    
    uint64_t baseOffset = idx * nChunks;
    uint8_t* scalarPtr = scalars + idx * scalarSize;
    
    int carry = 0;
    for (uint64_t j = 0; j < nChunks; j++) {
        uint64_t bitStart = j * bitsPerChunk;
        uint64_t byteStart = bitStart / 8;
        uint64_t effectiveBitsPerChunk = bitsPerChunk;
        
        if (byteStart > scalarSize - 8) byteStart = scalarSize - 8;
        if (bitStart + bitsPerChunk > scalarSize * 8) {
            effectiveBitsPerChunk = scalarSize * 8 - bitStart;
        }
        
        uint64_t shift = bitStart - byteStart * 8;
        uint64_t v = *(uint64_t*)(scalarPtr + byteStart);
        
        v = v >> shift;
        v = v & (((uint64_t)1 << effectiveBitsPerChunk) - 1);
        
        int bucketIndex = int32_t(v) + carry;
        
        if (bucketIndex >= nBuckets) {
            bucketIndex -= nBuckets * 2;
            carry = 1;
        } else {
            carry = 0;
        }
        
        slicedScalars[baseOffset + j] = bucketIndex;
    }
}

// Bucket filling kernel - adds points to buckets in parallel
__global__ void bucketFillingKernel_kernel(void* buckets, void* bases, int32_t* slicedScalars,
                                          uint64_t nPoints, uint64_t nChunks, uint64_t nBuckets) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nPoints) return;
    
    // This is a simplified version - in practice, we'd need proper elliptic curve arithmetic
    // For now, we'll use atomic operations on bucket indices
    uint64_t chunkIdx = blockIdx.y;
    int32_t bucketIndex = slicedScalars[idx * nChunks + chunkIdx];
    
    if (bucketIndex != 0) {
        // Note: This is a placeholder - actual elliptic curve addition would be implemented here
        // using proper field arithmetic and curve operations
        atomicAdd((int*)buckets + abs(bucketIndex) - 1, 1);
    }
}

// Bucket accumulation kernel - reduces buckets to chunks
__global__ void bucketAccumulationKernel_kernel(void* chunks, void* buckets, 
                                               uint64_t nChunks, uint64_t nBuckets) {
    uint64_t chunkIdx = blockIdx.x;
    if (chunkIdx >= nChunks) return;
    
    // This is a simplified version - actual implementation would use proper elliptic curve arithmetic
    // For now, we'll use a simple reduction pattern
    
    // Note: This kernel would implement the actual bucket accumulation logic
    // using proper field arithmetic and curve operations
}

// Final accumulation kernel - combines chunks into final result
__global__ void finalAccumulationKernel_kernel(void* result, void* chunks, 
                                              uint64_t nChunks, uint64_t bitsPerChunk) {
    // This kernel would implement the final accumulation logic
    // using proper elliptic curve arithmetic (doubling and addition)
    
    // Note: This is a placeholder - actual implementation would use proper field arithmetic
}

// Kernel launcher functions
extern "C" void scalarSlicingKernel(int32_t* slicedScalars, uint8_t* scalars, 
                                    uint64_t nPoints, uint64_t nChunks, uint64_t nBuckets,
                                    uint64_t bitsPerChunk, uint64_t scalarSize) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (nPoints + threadsPerBlock - 1) / threadsPerBlock;
    
    scalarSlicingKernel_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        slicedScalars, scalars, nPoints, nChunks, nBuckets, bitsPerChunk, scalarSize);
    
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void bucketFillingKernel(void* buckets, void* bases, int32_t* slicedScalars,
                                    uint64_t nPoints, uint64_t nChunks, uint64_t nBuckets) {
    int threadsPerBlock = 256;
    dim3 blocksPerGrid((nPoints + threadsPerBlock - 1) / threadsPerBlock, nChunks);
    
    bucketFillingKernel_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        buckets, bases, slicedScalars, nPoints, nChunks, nBuckets);
    
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void bucketAccumulationKernel(void* chunks, void* buckets, 
                                         uint64_t nChunks, uint64_t nBuckets) {
    int threadsPerBlock = 256;
    int blocksPerGrid = nChunks;
    
    bucketAccumulationKernel_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        chunks, buckets, nChunks, nBuckets);
    
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void finalAccumulationKernel(void* result, void* chunks, 
                                        uint64_t nChunks, uint64_t bitsPerChunk) {
    int threadsPerBlock = 256;
    int blocksPerGrid = 1; // Single block for final accumulation
    
    finalAccumulationKernel_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        result, chunks, nChunks, bitsPerChunk);
    
    CUDA_CHECK(cudaGetLastError());
}

} // namespace MSM_GPU
