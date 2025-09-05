#include <memory>
#include <chrono>
#include <iostream>
#include "msm.hpp"
#include "misc.hpp"
#include "gpu/gpu_integration.hpp"
#include <cuda_runtime.h>

// Forward declaration of GPU kernel
extern "C" void gpu_bucket_accumulation_kernel(
    const void* bases,
    const int32_t* slicedScalars,
    uint64_t nPoints,
    uint64_t nChunks,
    uint64_t nBuckets,
    uint64_t nThreads,
    void* bucketMatrix,
    void* chunks
);

template <typename Curve, typename BaseField>
void MSM<Curve, BaseField>::run(typename Curve::Point &r,
                                typename Curve::PointAffine *_bases,
                                uint8_t* _scalars,
                                uint64_t _scalarSize,
                                uint64_t _n,
                                uint64_t _nThreads)
{
    
    std::cerr << "MSM: Using CPU algorithm for " << _n << " points" << std::endl;
    
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    ThreadPool &threadPool = ThreadPool::defaultPool();

    auto setupStart = std::chrono::high_resolution_clock::now();
    
    const uint64_t nThreads = threadPool.getThreadCount();
    const uint64_t nPoints = _n;

    scalars = _scalars;
    scalarSize = _scalarSize;

#ifdef MSM_BITS_PER_CHUNK
    bitsPerChunk = MSM_BITS_PER_CHUNK;
#else
    bitsPerChunk = calcBitsPerChunk(nPoints, scalarSize);
#endif

    auto setupEnd = std::chrono::high_resolution_clock::now();
    auto setupDuration = std::chrono::duration_cast<std::chrono::microseconds>(setupEnd - setupStart);
    std::cerr << "            MSM Setup: " << setupDuration.count() << " μs" << std::endl;

    if (nPoints == 0) {
        g.copy(r, g.zero());
        auto totalEnd = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(totalEnd - totalStart);
        std::cerr << "            MSM Total (0 points): " << totalDuration.count() << " μs" << std::endl;
        return;
    }
    if (nPoints == 1) {
        auto singlePointStart = std::chrono::high_resolution_clock::now();
        g.mulByScalar(r, _bases[0], scalars, scalarSize);
        auto singlePointEnd = std::chrono::high_resolution_clock::now();
        auto singlePointDuration = std::chrono::duration_cast<std::chrono::microseconds>(singlePointEnd - singlePointStart);
        std::cerr << "            MSM Single point: " << singlePointDuration.count() << " μs" << std::endl;
        
        auto totalEnd = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(totalEnd - totalStart);
        std::cerr << "            MSM Total (1 point): " << totalDuration.count() << " μs" << std::endl;
        return;
    }

    auto paramCalcStart = std::chrono::high_resolution_clock::now();
    
    const uint64_t nChunks = calcChunkCount(scalarSize, bitsPerChunk);
    const uint64_t nBuckets = calcBucketCount(bitsPerChunk);
    const uint64_t matrixSize = nThreads * nBuckets;
    const uint64_t nSlices = nChunks*nPoints;

    auto paramCalcEnd = std::chrono::high_resolution_clock::now();
    auto paramCalcDuration = std::chrono::duration_cast<std::chrono::microseconds>(paramCalcEnd - paramCalcStart);
    std::cerr << "            MSM Parameter calculation: " << paramCalcDuration.count() << " μs" << std::endl;

    auto memoryAllocStart = std::chrono::high_resolution_clock::now();
    
    std::unique_ptr<typename Curve::Point[]> bucketMatrix(new typename Curve::Point[matrixSize]);
    std::unique_ptr<typename Curve::Point[]> chunks(new typename Curve::Point[nChunks]);
    std::unique_ptr<int32_t[]> slicedScalars(new int32_t[nSlices]);
    
    auto memoryAllocEnd = std::chrono::high_resolution_clock::now();
    auto memoryAllocDuration = std::chrono::duration_cast<std::chrono::microseconds>(memoryAllocEnd - memoryAllocStart);
    std::cerr << "            MSM Memory allocation: " << memoryAllocDuration.count() << " μs" << std::endl;

    auto scalarSlicingStart = std::chrono::high_resolution_clock::now();
    
    threadPool.parallelFor(0, nPoints, [&] (int begin, int end, int numThread) {

        for (int i = begin; i < end; i++) {
            int carry = 0;

            for (int j = 0; j < nChunks; j++) {
                int bucketIndex = getBucketIndex(i, j) + carry;

                if (bucketIndex >= nBuckets) {
                    bucketIndex -= nBuckets*2;
                    carry = 1;
                } else {
                    carry = 0;
                }

                slicedScalars[i*nChunks + j] = bucketIndex;
            }
        }
    });
    
    auto scalarSlicingEnd = std::chrono::high_resolution_clock::now();
    auto scalarSlicingDuration = std::chrono::duration_cast<std::chrono::microseconds>(scalarSlicingEnd - scalarSlicingStart);
    std::cerr << "            MSM Scalar slicing: " << scalarSlicingDuration.count() << " μs" << std::endl;

    auto bucketAccumulationStart = std::chrono::high_resolution_clock::now();
    
    // Try GPU acceleration for bucket accumulation
    if (ffiasm_gpu::GPUIntegration::isGPUAccelerationAvailable() && nPoints > 10000) {
        std::cerr << "            MSM: Using GPU acceleration for bucket accumulation" << std::endl;
        
        // GPU-accelerated bucket accumulation
        
        // Allocate GPU memory for all chunks at once
        typename Curve::PointAffine* d_bases;
        int32_t* d_slicedScalars;
        typename Curve::Point* d_bucketMatrix;
        typename Curve::Point* d_chunks;
        
        cudaMalloc(&d_bases, nPoints * sizeof(typename Curve::PointAffine));
        cudaMalloc(&d_slicedScalars, nChunks * nPoints * sizeof(int32_t));
        cudaMalloc(&d_bucketMatrix, nChunks * nBuckets * sizeof(typename Curve::Point));
        cudaMalloc(&d_chunks, nChunks * sizeof(typename Curve::Point));
        
        // Copy data to GPU
        cudaMemcpy(d_bases, _bases, nPoints * sizeof(typename Curve::PointAffine), cudaMemcpyHostToDevice);
        cudaMemcpy(d_slicedScalars, slicedScalars.get(), nChunks * nPoints * sizeof(int32_t), cudaMemcpyHostToDevice);
        
        // Launch GPU kernel for bucket accumulation
        gpu_bucket_accumulation_kernel(
            d_bases, d_slicedScalars, nPoints, nChunks, nBuckets, nThreads, d_bucketMatrix, d_chunks
        );
        cudaDeviceSynchronize();
        
        // Copy results back to CPU
        cudaMemcpy(chunks.get(), d_chunks, nChunks * sizeof(typename Curve::Point), cudaMemcpyDeviceToHost);
        
        // Cleanup GPU memory
        cudaFree(d_bases);
        cudaFree(d_slicedScalars);
        cudaFree(d_bucketMatrix);
        cudaFree(d_chunks);
    } else {
        // Fallback to CPU implementation
        std::cerr << "            MSM: Using CPU implementation for bucket accumulation" << std::endl;
        
        threadPool.parallelFor(0, nChunks, [&] (int begin, int end, int numThread) {

            for (int j = begin; j < end; j++) {

                typename Curve::Point *buckets = &bucketMatrix[numThread*nBuckets];

                for (int i = 0; i < nBuckets; i++) {
                    g.copy(buckets[i], g.zero());
                }

                for (int i = 0; i < nPoints; i++) {
                    const int bucketIndex = slicedScalars[i*nChunks + j];

                    if (bucketIndex > 0) {
                        g.add(buckets[bucketIndex-1], buckets[bucketIndex-1], _bases[i]);

                    } else if (bucketIndex < 0) {
                        g.sub(buckets[-bucketIndex-1], buckets[-bucketIndex-1], _bases[i]);
                    }
                }

                typename Curve::Point t, tmp;

                g.copy(t, buckets[nBuckets - 1]);
                g.copy(tmp, t);

                for (int i = nBuckets - 2; i >= 0 ; i--) {
                    g.add(tmp, tmp, buckets[i]);
                    g.add(t, t, tmp);
                }

                chunks[j] = t;
            }
        });
    }
    
    auto bucketAccumulationEnd = std::chrono::high_resolution_clock::now();
    auto bucketAccumulationDuration = std::chrono::duration_cast<std::chrono::microseconds>(bucketAccumulationEnd - bucketAccumulationStart);
    std::cerr << "            MSM Bucket accumulation: " << bucketAccumulationDuration.count() << " μs" << std::endl;

    auto finalAccumulationStart = std::chrono::high_resolution_clock::now();
    
    g.copy(r, chunks[nChunks - 1]);

    for (int j = nChunks - 2; j >= 0; j--) {
        for (int i = 0; i < bitsPerChunk; i++) {
            g.dbl(r, r);
        }
        g.add(r, r, chunks[j]);
    }
    
    auto finalAccumulationEnd = std::chrono::high_resolution_clock::now();
    auto finalAccumulationDuration = std::chrono::duration_cast<std::chrono::microseconds>(finalAccumulationEnd - finalAccumulationStart);
    std::cerr << "            MSM Final accumulation: " << finalAccumulationDuration.count() << " μs" << std::endl;
    
    // Calculate and print total time
    auto totalEnd = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(totalEnd - totalStart);
    
    std::cerr << "            MSM Total time: " << totalDuration.count() << " μs" << std::endl;
    std::cerr << "            MSM Summary - Setup: " << setupDuration.count() 
              << " μs, Params: " << paramCalcDuration.count() 
              << " μs, Alloc: " << memoryAllocDuration.count() 
              << " μs, Slicing: " << scalarSlicingDuration.count() 
              << " μs, Buckets: " << bucketAccumulationDuration.count() 
              << " μs, Final: " << finalAccumulationDuration.count() << " μs" << std::endl;
}
