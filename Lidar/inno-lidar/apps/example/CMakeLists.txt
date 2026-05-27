project(demo)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../../build_script/")
include(${CMAKE_MODULE_PATH}inno_lidar_base.cmake)

add_executable(demo ${PROJECT_SOURCE_DIR}/demo.cpp)

add_executable(sphere2xyz ${PROJECT_SOURCE_DIR}/sphere2xyz.cpp)

# Add inno_sdk_wrapper library (shared/dynamic library)
add_library(inno_sdk_wrapper SHARED ${PROJECT_SOURCE_DIR}/inno_sdk_wrapper.cpp)

#add_library(inno_sdk_lib SHARED  ${PROJECT_SOURCE_DIR}/inno_lidar_sdk.cpp)


if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(demo PRIVATE innolidarsdkclient innolidarsdkcommon innolidarutils ws2_32 wsock32)
    target_link_libraries(sphere2xyz PRIVATE innolidarsdkclient innolidarsdkcommon innolidarutils ws2_32 wsock32)
    target_link_libraries(inno_sdk_lib PRIVATE innolidarsdkclient innolidarsdkcommon innolidarutils ws2_32 wsock32)
    target_link_libraries(inno_sdk_wrapper PRIVATE innolidarsdkclient innolidarsdkcommon innolidarutils ws2_32 wsock32)
else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    find_library(LIBPCAP_LIBRARY NAMES pcap)
    if(${LIBPCAP_LIBRARY} STREQUAL "LIBPCAP_LIBRARY-NOTFOUND")
        message(WARNING "the pcap library was not found")
        target_link_libraries(demo PRIVATE libinnolidarsdkclient.a libinnolidarsdkcommon.a libinnolidarutils.a pthread)
        target_link_libraries(sphere2xyz PRIVATE libinnolidarsdkclient.a libinnolidarsdkcommon.a libinnolidarutils.a pthread)
        target_link_libraries(inno_sdk_wrapper PRIVATE libinnolidarsdkclient.a libinnolidarsdkcommon.a libinnolidarutils.a pthread)
    else()
        target_link_libraries(demo PRIVATE libinnolidarsdkclient.a libinnolidarsdkcommon.a libinnolidarutils.a pthread libpcap.a)
        target_link_libraries(sphere2xyz PRIVATE libinnolidarsdkclient.a libinnolidarsdkcommon.a libinnolidarutils.a pthread libpcap.a)
        target_link_libraries(inno_sdk_wrapper PRIVATE libinnolidarsdkclient.a libinnolidarsdkcommon.a libinnolidarutils.a pthread libpcap.a)
    endif()
endif()



find_package(pybind11 REQUIRED) 
#target_link_libraries(inno_sdk_lib PRIVATE pybind11::module)


pybind11_add_module(inno_sdk_wrapper_python ${PROJECT_SOURCE_DIR}/inno_lidar_sdk.cpp) 
target_link_libraries(inno_sdk_wrapper_python PRIVATE libinnolidarsdkclient.a libinnolidarsdkcommon.a libinnolidarutils.a pthread
/usr/lib/python3.8/config-3.8-aarch64-linux-gnu/libpython3.8.so
pybind11::module
)
target_include_directories(inno_sdk_wrapper_python PRIVATE   
    /usr/include/python3.8
) 


set(targets demo sphere2xyz inno_sdk_wrapper)

foreach(target ${targets})
    if(${target} STREQUAL "inno_sdk_wrapper")
        INSTALL(TARGETS ${target}
            LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/
            ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/
        )
    else()
    INSTALL(TARGETS ${target}
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/
    )
    endif()
endforeach()
