cmake_minimum_required(VERSION 3.18)

# Enable CUDA
find_package(CUDA REQUIRED)

# Define ENABLE_CUDA flag
add_definitions(-DENABLE_CUDA)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-arch=sm_60;-std=c++11)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# CUDA source files
cuda_add_library(msm_gpu SHARED
    msm_gpu.cu
    msm_gpu.cpp
)

# Set properties for CUDA library
set_target_properties(msm_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link against CUDA libraries
target_link_libraries(msm_gpu ${CUDA_LIBRARIES})

# C++ source files (non-CUDA)
add_library(msm_cpu STATIC
    msm.cpp
)

# Set C++ standard to match main project
target_compile_features(msm_cpu PRIVATE cxx_std_11)

# Install targets
install(TARGETS msm_gpu msm_cpu
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers
install(FILES 
    msm.hpp
    msm_gpu.hpp
    DESTINATION include
)
