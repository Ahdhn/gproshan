cmake_minimum_required(VERSION 3.12)

project(gproshan VERSION 2.0 LANGUAGES CXX)

find_package(CUDA)
if(CUDA_FOUND)
	enable_language(CUDA)
endif(CUDA_FOUND)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-fopenmp -Wall")
set(CMAKE_CUDA_FLAGS "-Xcompiler -fopenmp")

# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

if(CUDA_FOUND)
	add_definitions(-DGPROSHAN_CUDA)
	include_directories(${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)

include_directories(${gproshan_SOURCE_DIR}/include)
include_directories(${gproshan_SOURCE_DIR}/include/cuda)

FILE(GLOB_RECURSE sources ${gproshan_SOURCE_DIR}/src/*.* ${gproshan_SOURCE_DIR}/include/*.*)
add_executable(test_geodesics test/test_geodesics.cpp ${sources})
set_target_properties(test_geodesics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

file(MAKE_DIRECTORY tmp)

