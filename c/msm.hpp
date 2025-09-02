#ifndef MSM_HPP
#define MSM_HPP
#define ENABLE_CUDA

#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <cstdlib>
#include <string>

// Forward declaration for GPU acceleration
#ifdef ENABLE_CUDA
namespace MSM_GPU {
    template <typename Curve, typename BaseField>
    class GPUMSM;
}
#endif

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

public:
    MSM(Curve &_g): g(_g), gpuEnabled(false) 
#ifdef ENABLE_CUDA
        , gpuMSM(nullptr)
#endif
    {
        // Check if GPU is globally enabled for this instance
#ifdef ENABLE_CUDA
        // Check environment variable for GPU acceleration
        const char* gpuEnv = std::getenv("ENABLE_GPU_ACCELERATION");
        if (gpuEnv && std::string(gpuEnv) == "1") {
            // Try to enable GPU acceleration automatically
            if (enableGlobalGPU()) {
                gpuEnabled = gpuGloballyEnabled;
                // Note: gpuMSM will be nullptr, but we'll use global GPU state
            }
        }
#endif
    }

    void run(typename Curve::Point &r,
             typename Curve::PointAffine *_bases,
             uint8_t* _scalars,
             uint64_t _scalarSize,
             uint64_t _n,
             uint64_t _nThreads=0);

    // New batch MSM function for combining multiple MSM operations
    void runBatch(std::vector<typename Curve::Point> &results,
                  std::vector<typename Curve::PointAffine*> _basesArray,
                  std::vector<uint8_t*> _scalarsArray,
                  std::vector<uint64_t> _scalarSizes,
                  std::vector<uint64_t> _nArray,
                  uint64_t _nThreads=0);
    
    // GPU acceleration methods
#ifdef ENABLE_CUDA
    bool enableGPU();
    bool isGPUEnabled() const;
    void disableGPU();
    
    // Static GPU control methods
    static bool enableGlobalGPU();
    static bool isGlobalGPUEnabled();
    static void disableGlobalGPU();
#else
    bool enableGPU();
    bool isGPUEnabled() const;
    void disableGPU();
    
    // Static GPU control methods (stubs)
    static bool enableGlobalGPU();
    static bool isGlobalGPUEnabled();
    static void disableGlobalGPU();
#endif

private:
    // Helper function for batch MSM scalar processing
    int32_t getBucketIndexForOperation(uint64_t scalarIdx, uint64_t chunkIdx, 
                                      uint8_t* scalars, uint64_t scalarSize, 
                                      uint64_t bitsPerChunk) const;
    
    // GPU acceleration
#ifdef ENABLE_CUDA
    std::unique_ptr<MSM_GPU::GPUMSM<Curve, BaseField>> gpuMSM;
    bool gpuEnabled;
    
    // Static GPU state shared across all MSM instances
    static bool gpuGloballyEnabled;
    static std::unique_ptr<MSM_GPU::GPUMSM<Curve, BaseField>> gpuGlobalMSM;
    static std::once_flag gpuInitFlag;
#else
    bool gpuEnabled;
#endif
};

// Static member definitions
#ifdef ENABLE_CUDA
template <typename Curve, typename BaseField>
bool MSM<Curve, BaseField>::gpuGloballyEnabled = false;

template <typename Curve, typename BaseField>
std::unique_ptr<MSM_GPU::GPUMSM<Curve, BaseField>> MSM<Curve, BaseField>::gpuGlobalMSM = nullptr;

template <typename Curve, typename BaseField>
std::once_flag MSM<Curve, BaseField>::gpuInitFlag;
#endif

#include "msm.cpp"

#endif // MSM_HPP
