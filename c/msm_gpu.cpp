#include "msm_gpu.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

namespace MSM_GPU {

// GPU MSM Implementation
template <typename Curve, typename BaseField>
GPUMSM<Curve, BaseField>::GPUMSM() : gpuInitialized(false), 
                                    d_bases(nullptr), d_scalars(nullptr), d_results(nullptr),
                                    d_buckets(nullptr), d_chunks(nullptr), d_slicedScalars(nullptr),
                                    maxBasesSize(0), maxScalarsSize(0), maxResultsSize(0) {
}

template <typename Curve, typename BaseField>
GPUMSM<Curve, BaseField>::~GPUMSM() {
    freeGPUMemory();
}

template <typename Curve, typename BaseField>
bool GPUMSM<Curve, BaseField>::initialize() {
    if (!gpuContext.initialize()) {
        std::cerr << "            GPU MSM: Failed to initialize GPU context" << std::endl;
        return false;
    }
    
    gpuInitialized = true;
    std::cerr << "            GPU MSM: Successfully initialized GPU acceleration" << std::endl;
    
    return true;
}

template <typename Curve, typename BaseField>
bool GPUMSM<Curve, BaseField>::allocateGPUMemory(uint64_t maxPoints, uint64_t maxScalarSize) {
    if (!gpuInitialized) return false;
    
    // Calculate memory requirements
    size_t basesSize = maxPoints * sizeof(typename Curve::PointAffine);
    size_t scalarsSize = maxPoints * maxScalarSize;
    size_t resultsSize = sizeof(typename Curve::Point);
    size_t bucketsSize = maxPoints * sizeof(typename Curve::Point);
    size_t chunksSize = maxPoints * sizeof(typename Curve::Point);
    size_t slicedScalarsSize = maxPoints * maxPoints * sizeof(int32_t);
    
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_bases, basesSize));
    CUDA_CHECK(cudaMalloc(&d_scalars, scalarsSize));
    CUDA_CHECK(cudaMalloc(&d_results, resultsSize));
    CUDA_CHECK(cudaMalloc(&d_buckets, bucketsSize));
    CUDA_CHECK(cudaMalloc(&d_chunks, chunksSize));
    CUDA_CHECK(cudaMalloc(&d_slicedScalars, slicedScalarsSize));
    
    maxBasesSize = basesSize;
    maxScalarsSize = scalarsSize;
    maxResultsSize = resultsSize;
    
    std::cerr << "            GPU MSM: Allocated " << (basesSize + scalarsSize + resultsSize + 
                                                       bucketsSize + chunksSize + slicedScalarsSize) / (1024*1024) 
              << " MB GPU memory" << std::endl;
    
    return true;
}

template <typename Curve, typename BaseField>
void GPUMSM<Curve, BaseField>::freeGPUMemory() {
    if (d_bases) cudaFree(d_bases);
    if (d_scalars) cudaFree(d_scalars);
    if (d_results) cudaFree(d_results);
    if (d_buckets) cudaFree(d_buckets);
    if (d_chunks) cudaFree(d_chunks);
    if (d_slicedScalars) cudaFree(d_slicedScalars);
    
    d_bases = d_scalars = d_results = d_buckets = d_chunks = d_slicedScalars = nullptr;
}

template <typename Curve, typename BaseField>
bool GPUMSM<Curve, BaseField>::copyBasesToGPU(typename Curve::PointAffine* bases, uint64_t nPoints) {
    size_t size = nPoints * sizeof(typename Curve::PointAffine);
    CUDA_CHECK(cudaMemcpy(d_bases, bases, size, cudaMemcpyHostToDevice));
    return true;
}

template <typename Curve, typename BaseField>
bool GPUMSM<Curve, BaseField>::copyScalarsToGPU(uint8_t* scalars, uint64_t nPoints, uint64_t scalarSize) {
    size_t size = nPoints * scalarSize;
    CUDA_CHECK(cudaMemcpy(d_scalars, scalars, size, cudaMemcpyHostToDevice));
    return true;
}

template <typename Curve, typename BaseField>
bool GPUMSM<Curve, BaseField>::copyResultsFromGPU(typename Curve::Point &r) {
    CUDA_CHECK(cudaMemcpy(&r, d_results, sizeof(typename Curve::Point), cudaMemcpyDeviceToHost));
    return true;
}

template <typename Curve, typename BaseField>
void GPUMSM<Curve, BaseField>::launchScalarSlicingKernel(uint64_t nPoints, uint64_t nChunks, uint64_t nBuckets, 
                                                         uint64_t bitsPerChunk, uint64_t scalarSize) {
    scalarSlicingKernel((int32_t*)d_slicedScalars, (uint8_t*)d_scalars, 
                        nPoints, nChunks, nBuckets, bitsPerChunk, scalarSize);
}

template <typename Curve, typename BaseField>
void GPUMSM<Curve, BaseField>::launchBucketFillingKernel(uint64_t nPoints, uint64_t nChunks, uint64_t nBuckets) {
    bucketFillingKernel(d_buckets, d_bases, (int32_t*)d_slicedScalars, nPoints, nChunks, nBuckets);
}

template <typename Curve, typename BaseField>
void GPUMSM<Curve, BaseField>::launchBucketAccumulationKernel(uint64_t nChunks, uint64_t nBuckets) {
    bucketAccumulationKernel(d_chunks, d_buckets, nChunks, nBuckets);
}

template <typename Curve, typename BaseField>
void GPUMSM<Curve, BaseField>::launchFinalAccumulationKernel(uint64_t nChunks, uint64_t bitsPerChunk) {
    finalAccumulationKernel(d_results, d_chunks, nChunks, bitsPerChunk);
}

template <typename Curve, typename BaseField>
void GPUMSM<Curve, BaseField>::run(typename Curve::Point &r,
                                   typename Curve::PointAffine *_bases,
                                   uint8_t* _scalars,
                                   uint64_t _scalarSize,
                                   uint64_t _n,
                                   uint64_t _nThreads) {
    if (!gpuInitialized) {
        std::cerr << "            GPU MSM: GPU not initialized, falling back to CPU" << std::endl;
        return;
    }
    
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    const uint64_t nPoints = _n;
    
    // Allocate GPU memory if needed
    if (nPoints * sizeof(typename Curve::PointAffine) > maxBasesSize ||
        nPoints * _scalarSize > maxScalarsSize) {
        freeGPUMemory();
        if (!allocateGPUMemory(nPoints, _scalarSize)) {
            std::cerr << "            GPU MSM: Failed to allocate GPU memory, falling back to CPU" << std::endl;
            return;
        }
    }
    
    auto gpuTransferStart = std::chrono::high_resolution_clock::now();
    
    // Copy data to GPU
    if (!copyBasesToGPU(_bases, nPoints) || !copyScalarsToGPU(_scalars, nPoints, _scalarSize)) {
        std::cerr << "            GPU MSM: Failed to copy data to GPU, falling back to CPU" << std::endl;
        return;
    }
    
    auto gpuTransferEnd = std::chrono::high_resolution_clock::now();
    auto gpuTransferDuration = std::chrono::duration_cast<std::chrono::microseconds>(gpuTransferEnd - gpuTransferStart);
    std::cerr << "            GPU MSM: Data transfer to GPU: " << gpuTransferDuration.count() << " μs" << std::endl;
    
    auto gpuComputeStart = std::chrono::high_resolution_clock::now();
    
    // Calculate parameters
    uint64_t bitsPerChunk = 16; // Fixed for GPU optimization
    uint64_t nChunks = (_scalarSize * 8 + bitsPerChunk - 1) / bitsPerChunk;
    uint64_t nBuckets = (1ULL << (bitsPerChunk - 1)) * 2;
    
    // Launch GPU kernels
    launchScalarSlicingKernel(nPoints, nChunks, nBuckets, bitsPerChunk, _scalarSize);
    launchBucketFillingKernel(nPoints, nChunks, nBuckets);
    launchBucketAccumulationKernel(nChunks, nBuckets);
    launchFinalAccumulationKernel(nChunks, bitsPerChunk);
    
    // Synchronize GPU
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto gpuComputeEnd = std::chrono::high_resolution_clock::now();
    auto gpuComputeDuration = std::chrono::duration_cast<std::chrono::microseconds>(gpuComputeEnd - gpuComputeStart);
    std::cerr << "            GPU MSM: GPU computation: " << gpuComputeDuration.count() << " μs" << std::endl;
    
    auto gpuResultStart = std::chrono::high_resolution_clock::now();
    
    // Copy result back to CPU
    if (!copyResultsFromGPU(r)) {
        std::cerr << "            GPU MSM: Failed to copy result from GPU, falling back to CPU" << std::endl;
        return;
    }
    
    auto gpuResultEnd = std::chrono::high_resolution_clock::now();
    auto gpuResultDuration = std::chrono::duration_cast<std::chrono::microseconds>(gpuResultEnd - gpuResultStart);
    std::cerr << "            GPU MSM: Result transfer from GPU: " << gpuResultDuration.count() << " μs" << std::endl;
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(totalEnd - totalStart);
    
    std::cerr << "            GPU MSM: Total GPU time: " << totalDuration.count() << " μs" << std::endl;
    std::cerr << "            GPU MSM: Summary - Transfer: " << gpuTransferDuration.count() 
              << " μs, Compute: " << gpuComputeDuration.count() 
              << " μs, Result: " << gpuResultDuration.count() << " μs" << std::endl;
}

template <typename Curve, typename BaseField>
void GPUMSM<Curve, BaseField>::runBatch(std::vector<typename Curve::Point> &results,
                                        std::vector<typename Curve::PointAffine*> _basesArray,
                                        std::vector<uint8_t*> _scalarsArray,
                                        std::vector<uint64_t> _scalarSizes,
                                        std::vector<uint64_t> _nArray,
                                        uint64_t _nThreads) {
    if (!gpuInitialized) {
        std::cerr << "            GPU MSM Batch: GPU not initialized, falling back to CPU" << std::endl;
        return;
    }
    
    std::cerr << "            GPU MSM Batch: Processing " << _basesArray.size() << " operations on GPU" << std::endl;
    
    // For now, process each operation individually on GPU
    // Future optimization: implement true batch processing on GPU
    results.resize(_basesArray.size());
    
    for (size_t i = 0; i < _basesArray.size(); i++) {
        run(results[i], _basesArray[i], _scalarsArray[i], _scalarSizes[i], _nArray[i], _nThreads);
    }
}

// Explicit template instantiations for common curve types
// Note: These would need to be adapted based on the actual curve types used in the project
template class GPUMSM<void, void>; // Placeholder - replace with actual curve types

} // namespace MSM_GPU
