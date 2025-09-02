cmake_minimum_required(VERSION 3.18)

# Try to find CUDA (optional) - modern approach
find_package(CUDAToolkit QUIET)

# Check if CUDA was found
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found, enabling GPU acceleration")
    # Define ENABLE_CUDA flag
    add_definitions(-DENABLE_CUDA)
    # Enable CUDA language
    enable_language(CUDA)
else()
    message(STATUS "CUDA not found, building CPU-only version")
endif()

if(CUDAToolkit_FOUND)
    # Set CUDA flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_60 -std=c++11")

    # CUDA source files
    add_library(msm_gpu SHARED
        msm_gpu.cu
        msm_gpu.cpp
    )
    
    # Set CUDA properties
    set_target_properties(msm_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    # Link against CUDA libraries
    target_link_libraries(msm_gpu CUDA::cudart)
endif()

# C++ source files (non-CUDA)
add_library(msm_cpu STATIC
    msm.cpp
)

# Set C++ standard to match main project
target_compile_features(msm_cpu PRIVATE cxx_std_11)

# Install targets
if(CUDAToolkit_FOUND)
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
else()
    install(TARGETS msm_cpu
        ARCHIVE DESTINATION lib
    )
    
    # Install headers
    install(FILES 
        msm.hpp
        DESTINATION include
    )
endif()
