#ifndef MSM_HPP
#define MSM_HPP

#include <cstdint>
#include <iostream>

// Include ffiasm GPU integration
#include "gpu/gpu_integration.hpp"

template <typename Curve, typename BaseField>
class MSM {
    const uint64_t MIN_CHUNK_SIZE_BITS = 3;
    const uint64_t MAX_CHUNK_SIZE_BITS = 16;

    Curve &g;
    uint8_t *scalars;
    uint64_t scalarSize;
    uint64_t bitsPerChunk;

private:
    uint64_t calcAddsCount(uint64_t nPoints, uint64_t scalarSize, uint64_t bitsPerChunk) const {
        return calcChunkCount(scalarSize, bitsPerChunk)
                * (nPoints + ((uint64_t)1 << bitsPerChunk) + bitsPerChunk + 1);
    }

    uint64_t calcBitsPerChunk(uint64_t n, uint64_t scalarSize) const {
        uint64_t bitsPerChunk = MIN_CHUNK_SIZE_BITS;
        uint64_t minAdds = calcAddsCount(n, scalarSize, bitsPerChunk);

        for (uint64_t k = MIN_CHUNK_SIZE_BITS + 1; k <= MAX_CHUNK_SIZE_BITS; k++) {
            const uint64_t curAdds = calcAddsCount(n, scalarSize, k);

            if (curAdds < minAdds) {
                minAdds = curAdds;
                bitsPerChunk = k;
            }
        }
        return bitsPerChunk;
    }

    uint64_t calcChunkCount(uint64_t scalarSize, uint64_t bitsPerChunk) const {
        return ((scalarSize * 8 - 1 ) / bitsPerChunk) + 1;
    }

    uint64_t calcBucketCount(uint64_t bitsPerChunk) const {
        return ((uint64_t)1 << (bitsPerChunk-1));
    }

    uint64_t getBucketIndex(uint64_t scalarIdx, uint64_t chunkIdx) const {
        uint64_t bitStart = chunkIdx*bitsPerChunk;
        uint64_t byteStart = bitStart/8;
        uint64_t efectiveBitsPerChunk = bitsPerChunk;

        if (byteStart > scalarSize-8) byteStart = scalarSize - 8;
        if (bitStart + bitsPerChunk > scalarSize*8) efectiveBitsPerChunk = scalarSize*8 - bitStart;

        uint64_t shift = bitStart - byteStart*8;
        uint64_t v = *(uint64_t *)(scalars + scalarIdx*scalarSize + byteStart);

        v = v >> shift;
        v = v & ( ((uint64_t)1 << efectiveBitsPerChunk) - 1);

        return uint64_t(v);
    }

    // GPU acceleration method
    bool shouldUseGPU(uint64_t n) const {
        return ffiasm_gpu::GPUIntegration::isGPUAccelerationAvailable() && 
               n >= ffiasm_gpu::GPUIntegration::getMinPointsForGPU();
    }

    void runGPU(typename Curve::Point &r,
                typename Curve::PointAffine *_bases,
                uint8_t* _scalars,
                uint64_t _scalarSize,
                uint64_t _n,
                uint64_t _nThreads) {
        std::cerr << "MSM: Using GPU acceleration for " << _n << " points" << std::endl;
        bool gpu_success = ffiasm_gpu::GPUIntegration::runGPUMSM<Curve, BaseField>(
            r, _bases, _scalars, _scalarSize, _n, _nThreads);
        
        if (!gpu_success) {
            std::cerr << "MSM: GPU acceleration failed, falling back to CPU" << std::endl;
            // Fall back to CPU implementation
            runCPU(r, _bases, _scalars, _scalarSize, _n, _nThreads);
        }
    }

    void runCPU(typename Curve::Point &r,
                typename Curve::PointAffine *_bases,
                uint8_t* _scalars,
                uint64_t _scalarSize,
                uint64_t _n,
                uint64_t _nThreads);

public:
    MSM(Curve &_g): g(_g) {}

    void run(typename Curve::Point &r,
             typename Curve::PointAffine *_bases,
             uint8_t* _scalars,
             uint64_t _scalarSize,
             uint64_t _n,
             uint64_t _nThreads=0);
};

#include "msm.cpp"

#endif // MSM_HPP
