# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 2.8.7)
project(pointpillar)
set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g")

set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++14 -O0 -g")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -gencode arch=compute_$ENV{CUDASM},code=compute_$ENV{CUDASM}
)

include_directories(
    $ENV{CUDA_Inc}
    $ENV{TensorRT_Inc}
	../libraries/cuOSD/src
    src/common
	src/pointpillar
)

link_directories(
    $ENV{CUDA_Lib}
    $ENV{TensorRT_Lib}
    build
)

file(GLOB_RECURSE CORE_FILES 
    src/pointpillar/*.cu 
    src/pointpillar/*.cpp
    src/common/tensor.cu
    src/common/tensorrt.cpp
)

file(GLOB_RECURSE VISUALIZE_FILES 
  ../libraries/cuOSD/src/*.cpp
  ../libraries/cuOSD/src/*.cu
  src/common/visualize.cu
)

####################### libpointpillar_core.so ##########################
cuda_add_library(pointpillar_core SHARED 
  	${CORE_FILES}
)

target_link_libraries(pointpillar_core
	libnvinfer.so
	libnvonnxparser.so
)
#######################################################################


####################### pointpillar program #############################
cuda_add_executable(pointpillar 
	src/main.cpp
	${VISUALIZE_FILES}
)

target_link_libraries(pointpillar
	pointpillar_core
	libnvinfer.so
	libnvonnxparser.so
)
#######################################################################
