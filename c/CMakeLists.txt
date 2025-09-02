cmake_minimum_required(VERSION 3.18)

# Try to find CUDA (optional) - multiple approaches
find_package(CUDAToolkit QUIET)
if(NOT CUDAToolkit_FOUND)
    find_package(CUDA QUIET)
endif()

# Check if CUDA was found
if(CUDAToolkit_FOUND OR CUDA_FOUND)
    message(STATUS "CUDA found, enabling GPU acceleration")
    # Define ENABLE_CUDA flag
    add_definitions(-DENABLE_CUDA)
    # Enable CUDA language if using CUDAToolkit
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
    endif()
else()
    message(STATUS "CUDA not found, building CPU-only version")
endif()

if(CUDAToolkit_FOUND OR CUDA_FOUND)
    # Set CUDA flags
    if(CUDAToolkit_FOUND)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_60 -std=c++11")
    else()
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-arch=sm_60;-std=c++11)
    endif()

    # CUDA source files
    add_library(msm_gpu SHARED
        msm_gpu.cu
        msm_gpu.cpp
    )
    
    # Set CUDA properties
    if(CUDAToolkit_FOUND)
        set_target_properties(msm_gpu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
        )
        # Link against CUDA libraries
        target_link_libraries(msm_gpu CUDA::cudart)
    else()
        set_target_properties(msm_gpu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
        )
        # Include directories
        include_directories(${CUDA_INCLUDE_DIRS})
        # Link against CUDA libraries
        target_link_libraries(msm_gpu ${CUDA_LIBRARIES})
    endif()
endif()

# C++ source files (non-CUDA)
add_library(msm_cpu STATIC
    msm.cpp
)

# Set C++ standard to match main project
target_compile_features(msm_cpu PRIVATE cxx_std_11)

# Install targets
if(CUDAToolkit_FOUND OR CUDA_FOUND)
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
