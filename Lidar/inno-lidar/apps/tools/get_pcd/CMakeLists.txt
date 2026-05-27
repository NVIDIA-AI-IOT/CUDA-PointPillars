project(get_pcd)
cmake_minimum_required( VERSION 3.0 )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../../../build_script/")
include(${CMAKE_MODULE_PATH}inno_lidar_base.cmake)

set(SOURCES_GET_PCD get_pcd.cpp)
if (CMAKE_SYSTEM_NAME MATCHES "Windows" AND NOT MINGW)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/version_info.rc.in")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/version_info.rc.in ${PROJECT_SOURCE_DIR}/get_pcd_ver.rc @ONLY)
    endif()
    if (WIN32)
        find_program(RC_COMPILER NAMES rc)
    endif()

    if (RC_COMPILER)
        message(STATUS "Found rc compiler: ${RC_COMPILER}")
    else()
        message(FATAL_ERROR "@@Could not find rc compiler!@@")
    endif()
    set(res_file ${PROJECT_SOURCE_DIR}/get_pcd_ver.res)
    set(rc_file ${PROJECT_SOURCE_DIR}/get_pcd_ver.rc)
    # Add custom commands and targets
    add_custom_command(
        OUTPUT ${res_file}  # Specify the resource file name for output
        COMMAND ${RC_COMPILER} ${rc_file}  # Executed commands and parameters
        DEPENDS ${rc_file}  # Specify dependent input files
    )
    add_custom_target(VERSION_RES DEPENDS ${res_file})
endif()

add_executable(${PROJECT_NAME} ${SOURCES_GET_PCD})

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    if (NOT MINGW)
        target_sources(${PROJECT_NAME} PRIVATE ${SOURCES_GET_PCD} ${res_file})
        add_dependencies(${PROJECT_NAME} VERSION_RES)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${res_file})
    endif()
    # include MSVC and MingW
    target_link_libraries(${PROJECT_NAME} PRIVATE innolidarsdkclient innolidarsdkcommon innolidarutils ws2_32 wsock32)
else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    find_library(LIBPCAP_LIBRARY NAMES pcap)
    if(${LIBPCAP_LIBRARY} STREQUAL "LIBPCAP_LIBRARY-NOTFOUND")
        message(WARNING "the pcap library was not found")
        target_link_libraries(${PROJECT_NAME} PRIVATE innolidarsdkclient.a innolidarsdkcommon.a innolidarutils.a pthread)
    else()
        target_link_libraries(${PROJECT_NAME} PRIVATE innolidarsdkclient.a innolidarsdkcommon.a innolidarutils.a pthread libpcap.a)
    endif()
endif()

########################
INSTALL(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/
)
