cmake_minimum_required(VERSION 3.13)

project(gproshan VERSION 2.0 LANGUAGES CXX CUDA)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA 10.1 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(Armadillo REQUIRED)
find_package(CGAL REQUIRED)
find_package(X11 REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-fopenmp -Wall")
set(CMAKE_CUDA_FLAGS "-Xcompiler -fopenmp")

set(THREADS_PREFER_PTHREAD_FLAG ON)

add_definitions(${EIGEN3_DEFINITIONS})

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${AMADILLO_INCLUDE_DIR})
include_directories(${CGAL_INCLUDE_DIRS})
include_directories(${X11_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${SuiteSparse_INCLUDE_DIRS})

include_directories(${gproshan_SOURCE_DIR}/include)
include_directories(${gproshan_SOURCE_DIR}/include/viewer)
include_directories(${gproshan_SOURCE_DIR}/include/mdict)
include_directories(${gproshan_SOURCE_DIR}/include/cuda)

FILE(GLOB_RECURSE cpp_sources ${gproshan_SOURCE_DIR}/src/*.cpp)
FILE(GLOB_RECURSE cu_sources ${gproshan_SOURCE_DIR}/src/*.cu)

add_library(gproshan_cpp STATIC ${cpp_sources})
add_library(gproshan_cu STATIC ${cu_sources})

add_executable(gproshan gproshan.cpp)
add_executable(test_geodesics test_geodesics.cpp)
add_executable(test_image_denoising test_image_denoising.cpp)

set_target_properties(gproshan_cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(gproshan_cu gproshan_cpp)

target_link_libraries(gproshan gproshan_cpp gproshan_cu)
target_link_libraries(gproshan ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY})
target_link_libraries(gproshan ${OpenMP_CXX_LIBRARIES})
target_link_libraries(gproshan ${OPENGL_LIBRARIES})
target_link_libraries(gproshan ${GLUT_LIBRARIES})
target_link_libraries(gproshan ${ARMADILLO_LIBRARIES})
target_link_libraries(gproshan CGAL::CGAL)
target_link_libraries(gproshan ${X11_X11_LIB})
target_link_libraries(gproshan ${SuiteSparse_LIBRARIES})

target_link_libraries(test_geodesics gproshan_cpp gproshan_cu)
target_link_libraries(test_geodesics ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY})
target_link_libraries(test_geodesics ${OpenMP_CXX_LIBRARIES})
target_link_libraries(test_geodesics ${ARMADILLO_LIBRARIES})
target_link_libraries(test_geodesics ${SuiteSparse_LIBRARIES})

target_link_libraries(test_image_denoising gproshan_cpp gproshan_cu)
target_link_libraries(test_image_denoising ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY})
target_link_libraries(test_image_denoising ${OpenMP_CXX_LIBRARIES})
target_link_libraries(test_image_denoising ${ARMADILLO_LIBRARIES})
target_link_libraries(test_image_denoising ${X11_X11_LIB})
target_link_libraries(test_image_denoising ${SuiteSparse_LIBRARIES})

file(MAKE_DIRECTORY tmp)
