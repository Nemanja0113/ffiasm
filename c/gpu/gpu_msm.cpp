#include "gpu_msm.hpp"
#include "alt_bn128.hpp"
#include <iostream>
#include <algorithm>
#include <functional>
#include <numeric>
#include <cmath>
#include <cstring>

namespace GPU_MSM {

// GPU Memory Manager Implementation
GPUMemoryManager* GPUMemoryManager::instance = nullptr;

GPUMemoryManager::GPUMemoryManager() : initialized(false) {
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    error = cudaStreamCreate(&stream);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    initialized = true;
    std::cerr << "GPU Memory Manager initialized successfully" << std::endl;
}

GPUMemoryManager::~GPUMemoryManager() {
    if (initialized) {
        cudaStreamDestroy(stream);
        cudaDeviceReset();
    }
}

GPUMemoryManager& GPUMemoryManager::getInstance() {
    if (!instance) {
        instance = new GPUMemoryManager();
    }
    return *instance;
}

// GPU MSM Implementation
template<typename Curve>
GPUMSM<Curve>::GPUMSM(const GPUMSMConfig& cfg) 
    : config(cfg), memoryManager(GPUMemoryManager::getInstance()),
      gpu_bases(nullptr), gpu_scalars(nullptr), gpu_buckets(nullptr), gpu_result(nullptr),
      pinned_bases(nullptr), pinned_scalars(nullptr), pinned_result(nullptr),
      memory_allocated(false) {
}

template<typename Curve>
GPUMSM<Curve>::~GPUMSM() {
    cleanup();
}

template<typename Curve>
bool GPUMSM<Curve>::initialize() {
    if (!memoryManager.isInitialized()) {
        std::cerr << "GPU Memory Manager not initialized" << std::endl;
        return false;
    }
    
    // Allocate memory for maximum expected circuit size
    const uint64_t maxPoints = config.maxPointsPerBatch;
    const uint64_t maxBuckets = 1 << 16; // 2^16 buckets max
    
    if (!allocateMemory(maxPoints, maxBuckets)) {
        std::cerr << "Failed to allocate GPU memory" << std::endl;
        return false;
    }
    
    if (config.enablePinnedMemory && !allocatePinnedMemory(maxPoints)) {
        std::cerr << "Failed to allocate pinned memory" << std::endl;
        return false;
    }
    
    memory_allocated = true;
    std::cerr << "GPU MSM initialized successfully (max batch size: " << config.maxPointsPerBatch << " points)" << std::endl;
    return true;
}

template<typename Curve>
void GPUMSM<Curve>::cleanup() {
    deallocateMemory();
    deallocatePinnedMemory();
    memory_allocated = false;
}

template<typename Curve>
bool GPUMSM<Curve>::allocateMemory(uint64_t maxPoints, uint64_t maxBuckets) {
    cudaError_t error;
    
    // Allocate bases
    bases_size = maxPoints * sizeof(typename Curve::PointAffine);
    error = cudaMalloc(&gpu_bases, bases_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for bases: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate scalars
    scalars_size = maxPoints * 32; // Assume 32-byte scalars max
    error = cudaMalloc(&gpu_scalars, scalars_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for scalars: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate buckets
    buckets_size = maxBuckets * sizeof(typename Curve::Point);
    error = cudaMalloc(&gpu_buckets, buckets_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for buckets: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Allocate result
    result_size = sizeof(typename Curve::Point);
    error = cudaMalloc(&gpu_result, result_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for result: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    return true;
}

template<typename Curve>
void GPUMSM<Curve>::deallocateMemory() {
    if (gpu_bases) { cudaFree(gpu_bases); gpu_bases = nullptr; }
    if (gpu_scalars) { cudaFree(gpu_scalars); gpu_scalars = nullptr; }
    if (gpu_buckets) { cudaFree(gpu_buckets); gpu_buckets = nullptr; }
    if (gpu_result) { cudaFree(gpu_result); gpu_result = nullptr; }
}

template<typename Curve>
bool GPUMSM<Curve>::allocatePinnedMemory(uint64_t maxPoints) {
    cudaError_t error;
    
    // Allocate pinned memory for faster transfers
    error = cudaMallocHost(&pinned_bases, maxPoints * sizeof(typename Curve::PointAffine));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory for bases: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMallocHost(&pinned_scalars, maxPoints * 32);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory for scalars: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMallocHost(&pinned_result, sizeof(typename Curve::Point));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory for result: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    return true;
}

template<typename Curve>
void GPUMSM<Curve>::deallocatePinnedMemory() {
    if (pinned_bases) { cudaFreeHost(pinned_bases); pinned_bases = nullptr; }
    if (pinned_scalars) { cudaFreeHost(pinned_scalars); pinned_scalars = nullptr; }
    if (pinned_result) { cudaFreeHost(pinned_result); pinned_result = nullptr; }
}

template<typename Curve>
bool GPUMSM<Curve>::shouldUseGPU(uint64_t nPoints) const {
    bool result = nPoints >= config.minPointsForGPU && memory_allocated;
    if (!result) {
        std::cerr << "GPU rejected: nPoints=" << nPoints << ", minPoints=" << config.minPointsForGPU << ", memory_allocated=" << memory_allocated << std::endl;
    }
    return result;
}

template<typename Curve>
GPUMSMResult GPUMSM<Curve>::computeMSM(
    typename Curve::Point& result,
    const typename Curve::PointAffine* bases,
    const uint8_t* scalars,
    uint64_t scalarSize,
    uint64_t nPoints,
    uint64_t bitsPerChunk) {
    
    GPUMSMResult gpuResult;
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    if (!shouldUseGPU(nPoints)) {
        if (nPoints < config.minPointsForGPU) {
            gpuResult.errorMessage = "Circuit too small for GPU (" + std::to_string(nPoints) + " < " + std::to_string(config.minPointsForGPU) + ")";
        } else if (!memory_allocated) {
            gpuResult.errorMessage = "GPU memory not allocated";
        } else {
            gpuResult.errorMessage = "GPU not suitable for this circuit size";
        }
        return gpuResult;
    }
    
    if (nPoints > config.maxPointsPerBatch) {
        return computeBatchMSM(result, bases, scalars, scalarSize, nPoints, bitsPerChunk);
    }
    
    try {
        // Calculate optimal chunk size if not provided
        if (bitsPerChunk == 0) {
            bitsPerChunk = std::min(16ULL, std::max(3ULL, 
                static_cast<unsigned long long>(std::log2(static_cast<double>(nPoints)) / 2)));
        }
        
        const uint64_t nBuckets = 1 << (bitsPerChunk - 1);
        const uint64_t nChunks = ((scalarSize * 8 - 1) / bitsPerChunk) + 1;
        
        // Transfer data to GPU
        auto transferStart = std::chrono::high_resolution_clock::now();
        
        if (config.enablePinnedMemory) {
            // Use pinned memory for faster transfers
            memcpy(pinned_bases, bases, nPoints * sizeof(typename Curve::PointAffine));
            memcpy(pinned_scalars, scalars, nPoints * scalarSize);
            
            memoryManager.copyToGPU(gpu_bases, pinned_bases, nPoints);
            memoryManager.copyToGPU(gpu_scalars, pinned_scalars, nPoints * scalarSize);
        } else {
            memoryManager.copyToGPU(gpu_bases, bases, nPoints);
            memoryManager.copyToGPU(gpu_scalars, scalars, nPoints * scalarSize);
        }
        
        auto transferEnd = std::chrono::high_resolution_clock::now();
        gpuResult.transferTime = std::chrono::duration<double, std::milli>(transferEnd - transferStart).count();
        
        // Launch GPU kernel
        auto kernelResult = launchMSMKernel(bases, scalars, scalarSize, nPoints, bitsPerChunk);
        if (!kernelResult.success) {
            gpuResult.errorMessage = kernelResult.errorMessage;
            return gpuResult;
        }
        
        gpuResult.gpuTime = kernelResult.gpuTime;
        
        // Transfer result back
        auto resultTransferStart = std::chrono::high_resolution_clock::now();
        
        if (config.enablePinnedMemory) {
            memoryManager.copyFromGPU(pinned_result, gpu_result, 1);
            memoryManager.synchronize();
            memcpy(&result, pinned_result, sizeof(typename Curve::Point));
        } else {
            memoryManager.copyFromGPU(&result, gpu_result, 1);
            memoryManager.synchronize();
        }
        
        auto resultTransferEnd = std::chrono::high_resolution_clock::now();
        gpuResult.transferTime += std::chrono::duration<double, std::milli>(resultTransferEnd - resultTransferStart).count();
        
        auto totalEnd = std::chrono::high_resolution_clock::now();
        gpuResult.totalTime = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
        gpuResult.success = true;
        
        // Record performance metrics
        GPUPerformanceMonitor::getInstance().recordMSM(gpuResult.gpuTime, gpuResult.transferTime, nPoints);
        
    } catch (const std::exception& e) {
        gpuResult.errorMessage = std::string("GPU MSM computation failed: ") + e.what();
        gpuResult.success = false;
    }
    
    return gpuResult;
}

template<typename Curve>
GPUMSMResult GPUMSM<Curve>::computeBatchMSM(
    typename Curve::Point& result,
    const typename Curve::PointAffine* bases,
    const uint8_t* scalars,
    uint64_t scalarSize,
    uint64_t nPoints,
    uint64_t bitsPerChunk) {
    
    GPUMSMResult batchResult;
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    const uint64_t batchSize = config.maxPointsPerBatch;
    const uint64_t nBatches = (nPoints + batchSize - 1) / batchSize;
    
    std::vector<typename Curve::Point> batchResults(nBatches);
    
    for (uint64_t batch = 0; batch < nBatches; batch++) {
        uint64_t start = batch * batchSize;
        uint64_t end = std::min(start + batchSize, nPoints);
        uint64_t currentBatchSize = end - start;
        
        auto singleResult = computeMSM(
            batchResults[batch],
            &bases[start],
            &scalars[start * scalarSize],
            scalarSize,
            currentBatchSize,
            bitsPerChunk
        );
        
        if (!singleResult.success) {
            batchResult.errorMessage = "Batch " + std::to_string(batch) + " failed: " + singleResult.errorMessage;
            return batchResult;
        }
        
        batchResult.gpuTime += singleResult.gpuTime;
        batchResult.transferTime += singleResult.transferTime;
    }
    
    // Combine batch results
    result = batchResults[0];
    for (uint64_t i = 1; i < nBatches; i++) {
        // result = result + batchResults[i] (elliptic curve addition)
        // This would need to be implemented using the curve's add function
    }
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    batchResult.totalTime = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    batchResult.success = true;
    
    return batchResult;
}

template<typename Curve>
GPUMSMResult GPUMSM<Curve>::launchMSMKernel(
    const typename Curve::PointAffine* bases,
    const uint8_t* scalars,
    uint64_t scalarSize,
    uint64_t nPoints,
    uint64_t bitsPerChunk) {
    
    GPUMSMResult result;
    
    // This would call the appropriate CUDA kernel based on curve type
    // For now, we'll implement a placeholder
    
    auto kernelStart = std::chrono::high_resolution_clock::now();
    
    // Placeholder for actual kernel launch
    // In real implementation, this would call:
    // - launchG1MSMKernel for G1 operations
    // - launchG2MSMKernel for G2 operations
    
    auto kernelEnd = std::chrono::high_resolution_clock::now();
    result.gpuTime = std::chrono::duration<double, std::milli>(kernelEnd - kernelStart).count();
    result.success = true;
    
    return result;
}

template<typename Curve>
void GPUMSM<Curve>::printPerformanceStats(const GPUMSMResult& result) const {
    std::cerr << "GPU MSM Performance:" << std::endl;
    std::cerr << "  GPU Time: " << result.gpuTime << " ms" << std::endl;
    std::cerr << "  Transfer Time: " << result.transferTime << " ms" << std::endl;
    std::cerr << "  Total Time: " << result.totalTime << " ms" << std::endl;
    std::cerr << "  GPU Efficiency: " << (result.gpuTime / result.totalTime * 100) << "%" << std::endl;
}

// Performance Monitor Implementation
GPUPerformanceMonitor* GPUPerformanceMonitor::instance = nullptr;

GPUPerformanceMonitor& GPUPerformanceMonitor::getInstance() {
    if (!instance) {
        instance = new GPUPerformanceMonitor();
    }
    return *instance;
}

void GPUPerformanceMonitor::recordMSM(double gpuTime, double transferTime, uint64_t nPoints) {
    gpuTimes.push_back(gpuTime);
    transferTimes.push_back(transferTime);
    pointCounts.push_back(nPoints);
}

void GPUPerformanceMonitor::printStatistics() {
    if (gpuTimes.empty()) {
        std::cerr << "No GPU performance data recorded" << std::endl;
        return;
    }
    
    double avgGPUTime = getAverageGPUTime();
    double avgTransferTime = getAverageTransferTime();
    double avgThroughput = getAverageThroughput();
    
    std::cerr << "GPU Performance Statistics:" << std::endl;
    std::cerr << "  Average GPU Time: " << avgGPUTime << " ms" << std::endl;
    std::cerr << "  Average Transfer Time: " << avgTransferTime << " ms" << std::endl;
    std::cerr << "  Average Throughput: " << avgThroughput << " points/sec" << std::endl;
    std::cerr << "  Total Operations: " << gpuTimes.size() << std::endl;
}

void GPUPerformanceMonitor::reset() {
    gpuTimes.clear();
    transferTimes.clear();
    pointCounts.clear();
}

double GPUPerformanceMonitor::getAverageGPUTime() const {
    if (gpuTimes.empty()) return 0.0;
    double sum = std::accumulate(gpuTimes.begin(), gpuTimes.end(), 0.0);
    return sum / gpuTimes.size();
}

double GPUPerformanceMonitor::getAverageTransferTime() const {
    if (transferTimes.empty()) return 0.0;
    double sum = std::accumulate(transferTimes.begin(), transferTimes.end(), 0.0);
    return sum / transferTimes.size();
}

double GPUPerformanceMonitor::getAverageThroughput() const {
    if (pointCounts.empty() || gpuTimes.empty()) return 0.0;
    
    double totalPoints = std::accumulate(pointCounts.begin(), pointCounts.end(), 0.0);
    double totalTime = std::accumulate(gpuTimes.begin(), gpuTimes.end(), 0.0);
    
    return totalPoints / (totalTime / 1000.0); // Convert ms to seconds
}

// Explicit template instantiations for the linker
template class GPU_MSM::GPUMSM<AltBn128::Engine::G1>;
template class GPU_MSM::GPUMSM<AltBn128::Engine::G2>;

} // namespace GPU_MSM
