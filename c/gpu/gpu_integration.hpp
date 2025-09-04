#ifndef GPU_INTEGRATION_HPP
#define GPU_INTEGRATION_HPP

#include <cstdint>
#include <string>
#include <memory>

namespace ffiasm_gpu {

// Forward declarations
template<typename Curve, typename BaseField>
class GPUMSM;

// GPU integration interface for ffiasm
class GPUIntegration {
public:
    // Initialization and cleanup
    static void initializeGPUAcceleration();
    static void cleanupGPUAcceleration();
    
    // Availability check
    static bool isGPUAccelerationAvailable();
    
    // Configuration
    static uint64_t getMinPointsForGPU();
    static void setMinPointsForGPU(uint64_t minPoints);
    
    // Performance statistics
    static void printGPUPerformanceStats();
    
    // MSM execution
    template<typename Curve, typename BaseField>
    static bool runGPUMSM(
        typename Curve::Point& result,
        typename Curve::PointAffine* bases,
        uint8_t* scalars,
        uint64_t scalarSize,
        uint64_t n,
        uint64_t nThreads
    );
    
private:
    static bool gpu_initialized;
    static uint64_t min_points_for_gpu;
};

} // namespace ffiasm_gpu

#endif // GPU_INTEGRATION_HPP
