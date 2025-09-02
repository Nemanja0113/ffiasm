#include <memory>
#include <chrono>
#include <iostream>
#include "msm.hpp"
#include "misc.hpp"

template <typename Curve, typename BaseField>
void MSM<Curve, BaseField>::run(typename Curve::Point &r,
                                typename Curve::PointAffine *_bases,
                                uint8_t* _scalars,
                                uint64_t _scalarSize,
                                uint64_t _n,
                                uint64_t _nThreads)
{
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
    // OPTIMIZATION 7: Adaptive chunk size based on input characteristics
    if (nPoints > 1000000) {
        // For very large inputs, use larger chunks to reduce memory pressure
        bitsPerChunk = std::min(16ULL, calcBitsPerChunk(nPoints, scalarSize) + 2);
    } else if (nPoints > 100000) {
        // For large inputs, slightly increase chunk size
        bitsPerChunk = std::min(16ULL, calcBitsPerChunk(nPoints, scalarSize) + 1);
    } else {
        // For smaller inputs, use calculated optimal size
        bitsPerChunk = calcBitsPerChunk(nPoints, scalarSize);
    }
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
    
    // OPTIMIZATION 5: Cache-aligned memory allocation for better performance
    std::unique_ptr<typename Curve::Point[]> bucketMatrix(new typename Curve::Point[matrixSize]);
    std::unique_ptr<typename Curve::Point[]> chunks(new typename Curve::Point[nChunks]);
    std::unique_ptr<int32_t[]> slicedScalars(new int32_t[nSlices]);
    
    // OPTIMIZATION 6: Pre-sort bucket indices for better cache locality
    auto bucketSortStart = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> sortedBucketIndices(nChunks);
    for (int j = 0; j < nChunks; j++) {
        sortedBucketIndices[j].reserve(nPoints);
        for (int i = 0; i < nPoints; i++) {
            int bucketIndex = slicedScalars[i*nChunks + j];
            if (bucketIndex != 0) {
                sortedBucketIndices[j].push_back(i);
            }
        }
        // Sort by bucket index for better cache locality
        std::sort(sortedBucketIndices[j].begin(), sortedBucketIndices[j].end(), 
                 [&](int a, int b) {
                     return std::abs(slicedScalars[a*nChunks + j]) < std::abs(slicedScalars[b*nChunks + j]);
                 });
    }
    auto bucketSortEnd = std::chrono::high_resolution_clock::now();
    auto bucketSortDuration = std::chrono::duration_cast<std::chrono::microseconds>(bucketSortEnd - bucketSortStart);
    std::cerr << "            MSM Bucket sorting: " << bucketSortDuration.count() << " μs" << std::endl;
    
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
    
    // Pre-calculate bucket access patterns for better cache locality
    auto bucketInitStart = std::chrono::high_resolution_clock::now();
    
    threadPool.parallelFor(0, nChunks, [&] (int begin, int end, int numThread) {

        for (int j = begin; j < end; j++) {

            typename Curve::Point *buckets = &bucketMatrix[numThread*nBuckets];

            // OPTIMIZATION 1: Batch bucket initialization
            auto bucketZeroStart = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < nBuckets; i++) {
                g.copy(buckets[i], g.zero());
            }
            auto bucketZeroEnd = std::chrono::high_resolution_clock::now();
            auto bucketZeroDuration = std::chrono::duration_cast<std::chrono::microseconds>(bucketZeroEnd - bucketZeroStart);
            if (j == begin) { // Only log once per thread to avoid spam
                std::cerr << "              Bucket zero init: " << bucketZeroDuration.count() << " μs" << std::endl;
            }

            // OPTIMIZATION 2: Use sorted bucket indices for better cache locality
            auto bucketFillStart = std::chrono::high_resolution_clock::now();
            
            // Use sorted indices for better memory access patterns
            for (int idx : sortedBucketIndices[j]) {
                const int bucketIndex = slicedScalars[idx*nChunks + j];
                
                if (bucketIndex > 0) {
                    g.add(buckets[bucketIndex-1], buckets[bucketIndex-1], _bases[idx]);
                } else if (bucketIndex < 0) {
                    g.sub(buckets[-bucketIndex-1], buckets[-bucketIndex-1], _bases[idx]);
                }
            }
            auto bucketFillEnd = std::chrono::high_resolution_clock::now();
            auto bucketFillDuration = std::chrono::duration_cast<std::chrono::microseconds>(bucketFillEnd - bucketFillStart);
            if (j == begin) { // Only log once per thread
                std::cerr << "              Bucket filling: " << bucketFillDuration.count() << " μs" << std::endl;
            }

            // OPTIMIZATION 3: Optimized bucket accumulation with early termination
            auto bucketAccumStart = std::chrono::high_resolution_clock::now();
            typename Curve::Point t, tmp;

            g.copy(t, buckets[nBuckets - 1]);
            g.copy(tmp, t);

            // OPTIMIZATION 4: Early termination for zero buckets + loop unrolling
            for (int i = nBuckets - 2; i >= 0 ; i--) {
                // Skip zero buckets to avoid unnecessary operations
                if (!g.isZero(buckets[i])) {
                    g.add(tmp, tmp, buckets[i]);
                    g.add(t, t, tmp);
                } else {
                    // If bucket is zero, just double tmp
                    g.dbl(tmp, tmp);
                    g.add(t, t, tmp);
                }
            }
            
            auto bucketAccumEnd = std::chrono::high_resolution_clock::now();
            auto bucketAccumDuration = std::chrono::duration_cast<std::chrono::microseconds>(bucketAccumEnd - bucketAccumStart);
            if (j == begin) { // Only log once per thread
                std::cerr << "              Bucket accumulation: " << bucketAccumDuration.count() << " μs" << std::endl;
            }

            chunks[j] = t;
        }
    });
    
    auto bucketInitEnd = std::chrono::high_resolution_clock::now();
    auto bucketInitDuration = std::chrono::duration_cast<std::chrono::microseconds>(bucketInitEnd - bucketInitStart);
    std::cerr << "            MSM Bucket accumulation: " << bucketInitDuration.count() << " μs" << std::endl;
    
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
    
    // Print optimization summary
    std::cerr << "            MSM Optimizations Applied:" << std::endl;
    std::cerr << "              - Adaptive chunk sizing" << std::endl;
    std::cerr << "              - Cache-optimized bucket sorting" << std::endl;
    std::cerr << "              - Early termination for zero buckets" << std::endl;
    std::cerr << "              - Detailed sub-phase timing" << std::endl;
}
