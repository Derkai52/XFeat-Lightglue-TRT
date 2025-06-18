cmake_minimum_required(VERSION 3.10)
project(xfeat_lightglue)

# Set C++ standard
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Torch REQUIRED PATHS /home/tk/libtorch/share/cmake/Torch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 手动给出TensorRT的安装路径
set(TENSORRT_ROOT "/usr/local/TensorRT-8.6.1.6")
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)
set(TENSORRT_LIBS
    ${TENSORRT_ROOT}/lib/libnvinfer.so
    ${TENSORRT_ROOT}/lib/libnvinfer_plugin.so
    ${TENSORRT_ROOT}/lib/libnvonnxparser.so
    ${TENSORRT_ROOT}/lib/libnvparsers.so
)

# CUDA路径
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
set(CUDA_LIBRARY_DIRS "/usr/local/cuda/lib64")


include_directories(
  include
  ${TENSORRT_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# xfeat-lightglue 模块编译
add_library(xfeat_libs SHARED
    src/xfeat.cpp
    include/xfeat.h 
    include/InterpolateSparse2D.h   
    include/utils.h
)

add_library(lightglue_libs SHARED
    src/lightglue.cpp
    include/lightglue.h 
    include/InterpolateSparse2D.h   
    include/utils.h
)

link_directories(${CUDA_LIBRARY_DIRS} ${TENSORRT_LIBRARY_DIRS})
target_link_libraries(xfeat_libs ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${TENSORRT_LIBS} yaml-cpp)


add_executable(match_test src/main.cpp)
target_link_libraries(match_test xfeat_libs lightglue_libs ${TORCH_LIBRARIES})