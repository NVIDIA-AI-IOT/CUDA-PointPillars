cmake_minimum_required(VERSION 3.8.11)
set(CMAKE_C_COMPILER "/usr/local/gcc-12.2.0/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/local/gcc-12.2.0/bin/g++")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
PROJECT(Template)

set(WITH_PTCS_USE OFF CACHE BOOL "Use Ptcs")
# set(FIND_CUDA ON CACHE BOOL "Find CUDA")

if(FIND_CUDA)
    find_package(CUDA )
if(${CUDA_FOUND})
  	project(CommonModules LANGUAGES CXX CUDA)
	set(CUDA_SOURCE_PROPERTY_FORMAT OBJ)
	set(CUDA_SEPARABLE_COMPILATION ON)
	include_directories(${CUDA_INCLUDE_DIRS})
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	set(CUDA_NVCC_FLAGS -arch=sm_61;-O3;-G;-g;-std=c++14)#鏍规嵁鍏蜂綋GPU鎬ц兘鏇存敼绠楀姏鍙傛暟
	link_directories($ENV{CUDA_PATH}/lib/x64)
else(${CUDA_FOUND})
	MESSAGE(STATUS "cuda not found!")
endif(${CUDA_FOUND})
endif()





cmake_policy(SET CMP0053 NEW)
set(CMAKE_BUILD_TYPE ON)
set(CMAKE_BUILD_TYPE Release)
add_definitions(-O3)
add_compile_options(-Wall)

add_subdirectory(libhesai)

add_library(hesai_sdk_lib
	driver/hesai_lidar_sdk.cc
	)
target_link_libraries(hesai_sdk_lib PUBLIC 
					lidar_lib
					${OTHER_LIB})
set_target_properties(hesai_sdk_lib 
	PROPERTIES CXX_STANDARD_REQUIRED TRUE CXX_STANDARD 14)
target_include_directories(hesai_sdk_lib PUBLIC   
    ${OTHER_INCLUDE_DIRS}
	driver
	./
) 
if(FIND_CUDA)
if(${CUDA_FOUND})
	target_compile_definitions(hesai_sdk_lib PUBLIC USE_CUDA=1)
endif(${CUDA_FOUND})
endif()

#set(Python_ROOT_DIR "/usr/bin/python3.8") 
#find_package(Python 3.8 REQUIRED COMPONENTS Development)
set(pybind11_DIR "/usr/local/lib/python3.8/dist-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# 脤铆录脫脛茫碌脛脛拢驴茅
pybind11_add_module(hsai_sdk_wrapper_python sdk/hsai_sdk_wrapper_python.cc) 
target_link_libraries(hsai_sdk_wrapper_python PRIVATE hesai_sdk_lib
/usr/lib/python3.8/config-3.8-aarch64-linux-gnu/libpython3.8.so
)
target_include_directories(hsai_sdk_wrapper_python PRIVATE   
    /usr/include/python3.8
)

# C++ wrapper library (shared)
add_library(hesai_sdk_wrapper SHARED
    sdk/hsai_sdk_wrapper.cc
)
target_link_libraries(hesai_sdk_wrapper PRIVATE hesai_sdk_lib)
set_target_properties(hesai_sdk_wrapper 
    PROPERTIES CXX_STANDARD_REQUIRED TRUE CXX_STANDARD 14)
target_include_directories(hesai_sdk_wrapper PUBLIC   
    ${OTHER_INCLUDE_DIRS}
    driver
    ./
) 

if(DISENABLE_TEST_CC)

else()
	add_executable(sample
		test/test.cc
		)

	target_link_libraries(sample
		hesai_sdk_lib
		)

	add_executable(multi_test
		test/multi_test.cc
		)

	target_link_libraries(multi_test
		hesai_sdk_lib
		)
endif()

# Example program using C++ wrapper
add_executable(example_usage
    sdk/example_usage.cpp
)
target_link_libraries(example_usage PRIVATE hesai_sdk_wrapper)
