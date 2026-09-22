cmake_minimum_required(VERSION 3.16)
project(AeroSculpt_Native_Core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /openmp)
else()
    add_compile_options(-O3 -mavx2 -fopenmp -fPIC)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

add_library(aerosculpt_core SHARED
    src/fast_keyframe_filter.cpp
    src/umeyama_sim3.cpp
    src/pointcloud_densifier.cpp
)

target_link_libraries(aerosculpt_core
    PRIVATE
    ${OpenCV_LIBS}
)
