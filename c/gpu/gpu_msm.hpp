#ifndef GPU_MSM_HPP
#define GPU_MSM_HPP

#include <cuda_runtime.h>
#include <cuda.h>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <functional>

// Forward declarations
namespace AltBn128 {
    class Engine;
}

namespace GPU_MSM {

// GPU memory management class
class GPUMemoryManager {
private:
    static GPUMemoryManager* instance;
    bool initialized;
    cudaStream_t stream;
    
    GPUMemoryManager();
    
public:
    static GPUMemoryManager& getInstance();
    ~GPUMemoryManager();
    
    bool isInitialized() const { return initialized; }
    cudaStream_t getStream() const { return stream; }
    
    // Memory allocation helpers
    template<typename T>
    T* allocate(size_t count) {
        T* ptr;
        cudaMalloc(&ptr, count * sizeof(T));
        return ptr;
    }
    
    template<typename T>
    void deallocate(T* ptr) {
        if (ptr) cudaFree(ptr);
    }
    
    // Memory transfer helpers
    template<typename T>
    void copyToGPU(T* gpu_ptr, const T* cpu_ptr, size_t count) {
        cudaMemcpyAsync(gpu_ptr, cpu_ptr, count * sizeof(T), 
                       cudaMemcpyHostToDevice, stream);
    }
    
    template<typename T>
    void copyFromGPU(T* cpu_ptr, const T* gpu_ptr, size_t count) {
        cudaMemcpyAsync(cpu_ptr, gpu_ptr, count * sizeof(T), 
                       cudaMemcpyDeviceToHost, stream);
    }
    
    void synchronize() {
        cudaStreamSynchronize(stream);
    }
};

// GPU MSM configuration
struct GPUMSMConfig {
    uint64_t minPointsForGPU;      // Minimum points to use GPU
    uint64_t maxPointsPerBatch;    // Maximum points per GPU batch
    uint64_t threadsPerBlock;      // CUDA threads per block
    uint64_t maxBlocks;            // Maximum CUDA blocks
    bool enablePinnedMemory;       // Use pinned memory for transfers
    bool enableStreams;            // Use CUDA streams for overlap
    
    GPUMSMConfig() : 
        minPointsForGPU(100000),   // Use GPU for circuits with 100k+ points (lowered threshold)
        maxPointsPerBatch(5000000), // Process up to 5M points per batch (increased for large circuits)
        threadsPerBlock(256),
        maxBlocks(65535),
        enablePinnedMemory(true),
        enableStreams(true) {}
};

// GPU MSM result structure
struct GPUMSMResult {
    bool success;
    std::string errorMessage;
    double gpuTime;
    double transferTime;
    double totalTime;
    
    GPUMSMResult() : success(false), gpuTime(0), transferTime(0), totalTime(0) {}
};

// Main GPU MSM class
template<typename Curve>
class GPUMSM {
private:
    GPUMSMConfig config;
    GPUMemoryManager& memoryManager;
    
    // GPU memory pointers
    typename Curve::PointAffine* gpu_bases;
    uint8_t* gpu_scalars;
    typename Curve::Point* gpu_buckets;
    typename Curve::Point* gpu_result;
    
    // Memory sizes
    size_t bases_size;
    size_t scalars_size;
    size_t buckets_size;
    size_t result_size;
    
    // Pinned memory for faster transfers
    typename Curve::PointAffine* pinned_bases;
    uint8_t* pinned_scalars;
    typename Curve::Point* pinned_result;
    
    bool memory_allocated;
    
public:
    GPUMSM(const GPUMSMConfig& cfg = GPUMSMConfig());
    ~GPUMSM();
    
    // Initialize GPU resources
    bool initialize();
    void cleanup();
    
    // Main MSM computation
    GPUMSMResult computeMSM(
        typename Curve::Point& result,
        const typename Curve::PointAffine* bases,
        const uint8_t* scalars,
        uint64_t scalarSize,
        uint64_t nPoints,
        uint64_t bitsPerChunk = 0);
    
    // Batch processing for large circuits
    GPUMSMResult computeBatchMSM(
        typename Curve::Point& result,
        const typename Curve::PointAffine* bases,
        const uint8_t* scalars,
        uint64_t scalarSize,
        uint64_t nPoints,
        uint64_t bitsPerChunk = 0);
    
    // Utility functions
    bool shouldUseGPU(uint64_t nPoints) const;
    void printPerformanceStats(const GPUMSMResult& result) const;
    
private:
    // Internal helper functions
    bool allocateMemory(uint64_t maxPoints, uint64_t maxBuckets);
    void deallocateMemory();
    bool allocatePinnedMemory(uint64_t maxPoints);
    void deallocatePinnedMemory();
    
    // GPU kernel launchers
    GPUMSMResult launchMSMKernel(
        const typename Curve::PointAffine* bases,
        const uint8_t* scalars,
        uint64_t scalarSize,
        uint64_t nPoints,
        uint64_t bitsPerChunk);
    
    // Performance measurement
    double measureGPUTime(std::function<void()> operation);
    double measureTransferTime(std::function<void()> operation);
};

// Specialized implementations for different curves
// Note: These will be implemented in the .cpp file where AltBn128 is fully available
// Template specializations will be declared in the implementation file

// CUDA kernel declarations (implemented in .cu files)
extern "C" {
    // G1 MSM kernels - using void pointers to avoid type dependencies
    void launchG1MSMKernel(
        const void* bases,
        const uint8_t* scalars,
        void* buckets,
        void* result,
        uint64_t nPoints,
        uint64_t scalarSize,
        uint64_t bitsPerChunk,
        uint64_t nBuckets,
        cudaStream_t stream);
    
    // G2 MSM kernels - using void pointers to avoid type dependencies
    void launchG2MSMKernel(
        const void* bases,
        const uint8_t* scalars,
        void* buckets,
        void* result,
        uint64_t nPoints,
        uint64_t scalarSize,
        uint64_t bitsPerChunk,
        uint64_t nBuckets,
        cudaStream_t stream);
    
    // Utility kernels
    void launchScalarSlicingKernel(
        const uint8_t* scalars,
        int32_t* slicedScalars,
        uint64_t nPoints,
        uint64_t scalarSize,
        uint64_t nChunks,
        uint64_t bitsPerChunk,
        uint64_t nBuckets,
        cudaStream_t stream);
    
    void launchBucketAccumulationKernel(
        const void* bases,
        const int32_t* slicedScalars,
        void* buckets,
        uint64_t nPoints,
        uint64_t nChunks,
        uint64_t nBuckets,
        uint64_t chunkIndex,
        cudaStream_t stream);
}

// Performance monitoring
class GPUPerformanceMonitor {
private:
    static GPUPerformanceMonitor* instance;
    std::vector<double> gpuTimes;
    std::vector<double> transferTimes;
    std::vector<uint64_t> pointCounts;
    
public:
    static GPUPerformanceMonitor& getInstance();
    
    void recordMSM(double gpuTime, double transferTime, uint64_t nPoints);
    void printStatistics();
    void reset();
    
    double getAverageGPUTime() const;
    double getAverageTransferTime() const;
    double getAverageThroughput() const; // Points per second
};

} // namespace GPU_MSM

#endif // GPU_MSM_HPP
