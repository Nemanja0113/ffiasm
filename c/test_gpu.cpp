#include <iostream>
#include <chrono>
#include <vector>
#include "msm.hpp"
#include "msm_gpu.hpp"

// Simple test to demonstrate GPU acceleration
int main() {
    std::cout << "=== GPU-Accelerated MSM Test ===" << std::endl;
    
    try {
        // Create MSM instance
        // Note: This is a placeholder - you'll need to adapt this to your actual curve types
        std::cout << "Testing GPU acceleration..." << std::endl;
        
        // Test GPU initialization
        MSM_GPU::GPUMSMContext gpuContext;
        if (gpuContext.initialize()) {
            std::cout << "✓ GPU context initialized successfully" << std::endl;
            
            auto deviceInfo = gpuContext.getDeviceInfo();
            std::cout << "  Device: " << deviceInfo.deviceId << std::endl;
            std::cout << "  Compute Capability: " << deviceInfo.computeCapability << std::endl;
            std::cout << "  Total Memory: " << (deviceInfo.totalMemory / (1024*1024*1024)) << " GB" << std::endl;
            std::cout << "  Free Memory: " << (deviceInfo.freeMemory / (1024*1024*1024)) << " GB" << std::endl;
            
            // Test optimal configuration
            int threadsPerBlock, blocksPerGrid;
            gpuContext.getOptimalConfig(1000000, threadsPerBlock, blocksPerGrid);
            std::cout << "  Optimal config for 1M points: " << threadsPerBlock << " threads/block, " 
                      << blocksPerGrid << " blocks" << std::endl;
            
        } else {
            std::cout << "✗ Failed to initialize GPU context" << std::endl;
            std::cout << "  This is expected if CUDA is not available or no GPU is found" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "✗ Exception: " << e.what() << std::endl;
    }
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}
