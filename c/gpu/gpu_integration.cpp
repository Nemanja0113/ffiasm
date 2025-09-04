#include "gpu_integration.hpp"
#include "gpu_msm.hpp"
#include <iostream>
#include <cuda_runtime.h>

namespace ffiasm_gpu {

// Static member definitions
bool GPUIntegration::gpu_initialized = false;
uint64_t GPUIntegration::min_points_for_gpu = 50000;

void GPUIntegration::initializeGPUAcceleration() {
    if (gpu_initialized) {
        return;
    }
    
    std::cerr << "ffiasm GPU: Initializing GPU acceleration..." << std::endl;
    
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "ffiasm GPU: Failed to initialize CUDA device" << std::endl;
        return;
    }
    
    // Check GPU memory
    size_t free_mem, total_mem;
    cudaStatus = cudaMemGetInfo(&free_mem, &total_mem);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "ffiasm GPU: Failed to get GPU memory info" << std::endl;
        return;
    }
    
    std::cerr << "ffiasm GPU: GPU memory - Free: " << (free_mem / 1024 / 1024) 
              << " MB, Total: " << (total_mem / 1024 / 1024) << " MB" << std::endl;
    
    gpu_initialized = true;
    std::cerr << "ffiasm GPU: GPU acceleration initialized successfully" << std::endl;
}

void GPUIntegration::cleanupGPUAcceleration() {
    if (!gpu_initialized) {
        return;
    }
    
    std::cerr << "ffiasm GPU: Cleaning up GPU acceleration..." << std::endl;
    cudaDeviceReset();
    gpu_initialized = false;
}

bool GPUIntegration::isGPUAccelerationAvailable() {
    if (!gpu_initialized) {
        return false;
    }
    
    // Check if CUDA is available
    int deviceCount;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess || deviceCount == 0) {
        return false;
    }
    
    return true;
}

uint64_t GPUIntegration::getMinPointsForGPU() {
    return min_points_for_gpu;
}

void GPUIntegration::setMinPointsForGPU(uint64_t minPoints) {
    min_points_for_gpu = minPoints;
    std::cerr << "ffiasm GPU: Set minimum points for GPU to " << minPoints << std::endl;
}

void GPUIntegration::printGPUPerformanceStats() {
    if (!gpu_initialized) {
        std::cerr << "ffiasm GPU: GPU not initialized" << std::endl;
        return;
    }
    
    size_t free_mem, total_mem;
    cudaError_t cudaStatus = cudaMemGetInfo(&free_mem, &total_mem);
    if (cudaStatus == cudaSuccess) {
        std::cerr << "ffiasm GPU: GPU memory - Free: " << (free_mem / 1024 / 1024) 
                  << " MB, Total: " << (total_mem / 1024 / 1024) << " MB" << std::endl;
    }
}

// Template specialization for G1 points
template<>
bool GPUIntegration::runGPUMSM<struct AltBn128::Engine::G1, struct AltBn128::Engine::F1>(
    struct AltBn128::Engine::G1Point& result,
    struct AltBn128::Engine::G1PointAffine* bases,
    uint8_t* scalars,
    uint64_t scalarSize,
    uint64_t n,
    uint64_t nThreads
) {
    if (!isGPUAccelerationAvailable()) {
        return false;
    }
    
    std::cerr << "ffiasm GPU: Running GPU MSM for " << n << " G1 points" << std::endl;
    
    // Use the advanced GPU MSM implementation
    extern "C" void gpu_msm_advanced(
        struct AltBn128::Engine::G1Point* result,
        const struct AltBn128::Engine::G1PointAffine* bases,
        const uint8_t* scalars,
        uint64_t scalarSize,
        uint64_t nPoints,
        uint64_t nThreads
    );
    
    gpu_msm_advanced(&result, bases, scalars, scalarSize, n, nThreads);
    return true;
}

// Template specialization for G2 points
template<>
bool GPUIntegration::runGPUMSM<struct AltBn128::Engine::G2, struct AltBn128::Engine::F2>(
    struct AltBn128::Engine::G2Point& result,
    struct AltBn128::Engine::G2PointAffine* bases,
    uint8_t* scalars,
    uint64_t scalarSize,
    uint64_t n,
    uint64_t nThreads
) {
    if (!isGPUAccelerationAvailable()) {
        return false;
    }
    
    std::cerr << "ffiasm GPU: Running GPU MSM for " << n << " G2 points" << std::endl;
    
    // For now, G2 GPU acceleration is not implemented
    // This would require separate G2 GPU kernels
    std::cerr << "ffiasm GPU: G2 GPU acceleration not yet implemented" << std::endl;
    return false;
}

} // namespace ffiasm_gpu
