#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include "msm.hpp"

// Forward declarations for CUDA
struct cudaStream_t;
struct cudaEvent_t;

namespace MSM_GPU {

// GPU device information
struct GPUDevice {
    int deviceId;
    size_t totalMemory;
    size_t freeMemory;
    int computeCapability;
    int maxThreadsPerBlock;
    int maxBlocksPerGrid;
    bool available;
};

// GPU MSM context
class GPUMSMContext {
public:
    GPUMSMContext();
    ~GPUMSMContext();
    
    // Initialize GPU context
    bool initialize();
    
    // Get device information
    GPUDevice getDeviceInfo() const;
    
    // Check if GPU is available
    bool isAvailable() const;
    
    // Get optimal thread/block configuration
    void getOptimalConfig(uint64_t nPoints, int& threadsPerBlock, int& blocksPerGrid) const;

private:
    GPUDevice deviceInfo;
    bool initialized;
    
    // CUDA context management
    void* cudaContext;
    cudaStream_t* computeStream;
    cudaEvent_t* startEvent;
    cudaEvent_t* endEvent;
};

// GPU-accelerated MSM implementation
template <typename Curve, typename BaseField>
class GPUMSM {
public:
    GPUMSM();
    ~GPUMSM();
    
    // Initialize GPU resources
    bool initialize();
    
    // Run MSM on GPU
    void run(typename Curve::Point &r,
             typename Curve::PointAffine *_bases,
             uint8_t* _scalars,
             uint64_t _scalarSize,
             uint64_t _n,
             uint64_t _nThreads = 0);
    
    // Run batch MSM on GPU
    void runBatch(std::vector<typename Curve::Point> &results,
                  std::vector<typename Curve::PointAffine*> _basesArray,
                  std::vector<uint8_t*> _scalarsArray,
                  std::vector<uint64_t> _scalarSizes,
                  std::vector<uint64_t> _nArray,
                  uint64_t _nThreads = 0);

private:
    GPUMSMContext gpuContext;
    bool gpuInitialized;
    
    // GPU memory buffers
    void* d_bases;
    void* d_scalars;
    void* d_results;
    void* d_buckets;
    void* d_chunks;
    void* d_slicedScalars;
    
    // Memory sizes
    size_t maxBasesSize;
    size_t maxScalarsSize;
    size_t maxResultsSize;
    
    // Allocate GPU memory
    bool allocateGPUMemory(uint64_t maxPoints, uint64_t maxScalarSize);
    void freeGPUMemory();
    
    // Memory transfer helpers
    bool copyBasesToGPU(typename Curve::PointAffine* bases, uint64_t nPoints);
    bool copyScalarsToGPU(uint8_t* scalars, uint64_t nPoints, uint64_t scalarSize);
    bool copyResultsFromGPU(typename Curve::Point &r);
    
    // CUDA kernel launchers
    void launchScalarSlicingKernel(uint64_t nPoints, uint64_t nChunks, uint64_t nBuckets, 
                                   uint64_t bitsPerChunk, uint64_t scalarSize);
    void launchBucketFillingKernel(uint64_t nPoints, uint64_t nChunks, uint64_t nBuckets);
    void launchBucketAccumulationKernel(uint64_t nChunks, uint64_t nBuckets);
    void launchFinalAccumulationKernel(uint64_t nChunks, uint64_t bitsPerChunk);
};

// CUDA kernel declarations (implemented in .cu files)
extern "C" {
    // Scalar slicing kernel
    void scalarSlicingKernel(int32_t* slicedScalars, uint8_t* scalars, 
                             uint64_t nPoints, uint64_t nChunks, uint64_t nBuckets,
                             uint64_t bitsPerChunk, uint64_t scalarSize);
    
    // Bucket filling kernel
    void bucketFillingKernel(void* buckets, void* bases, int32_t* slicedScalars,
                             uint64_t nPoints, uint64_t nChunks, uint64_t nBuckets);
    
    // Bucket accumulation kernel
    void bucketAccumulationKernel(void* chunks, void* buckets, 
                                  uint64_t nChunks, uint64_t nBuckets);
    
    // Final accumulation kernel
    void finalAccumulationKernel(void* result, void* chunks, 
                                 uint64_t nChunks, uint64_t bitsPerChunk);
}

} // namespace MSM_GPU
