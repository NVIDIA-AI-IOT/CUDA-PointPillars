cmake_minimum_required(VERSION 3.0)
project(inno_clientsdk)

add_subdirectory(src)

if(NOT MAKE_SHARED)
    if(EXISTS "${CMAKE_SOURCE_DIR}/apps/example")
        add_subdirectory(apps/example)
    endif()

    if(EXISTS "${CMAKE_SOURCE_DIR}/apps/tools/get_pcd/")
        add_subdirectory(apps/tools/get_pcd)
    endif()
endif()
