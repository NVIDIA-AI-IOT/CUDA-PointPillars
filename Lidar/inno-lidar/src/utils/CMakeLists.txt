project(innolidarutils)
cmake_minimum_required(VERSION 3.0)

include(${PROJECT_SOURCE_DIR}/../../build_script/inno_lidar_base.cmake)

if(NOT "${ARCH_TAG}" STREQUAL "-arm" OR NOT DEFINED ARCH_TAG)
    add_definitions(-DHAVE_SSE42)
endif()
    message(STATUS "The Arch is ${ARCH_TAG}")
aux_source_directory(. DIR_SRCS)

# UNIX system but not MACOS
if(CMAKE_HOST_UNIX AND (NOT CMAKE_SYSTEM_NAME MATCHES "Darwin"))
    set(IS_ARM_ARCH OFF)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm)|(aarch64)")
        set(IS_ARM_ARCH ON)
    else()
        if (ARCH_TAG)
            if (${ARCH_TAG} STREQUAL "-arm")
                set(IS_ARM_ARCH ON)
            endif()
        endif()
    endif()
    if(IS_ARM_ARCH)
        if(NOT DEFINED MARCH_VALUE)
            set(MARCH_VALUE "armv8-a+crc")
        endif()
        if(NOT DEFINED MTUNE_VALUE)
            set(MTUNE_VALUE "cortex-a53")
        endif()

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${MARCH_VALUE} -mtune=${MTUNE_VALUE} -DARCH_ARM64")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND MAKE_SHARED)
    add_library(${PROJECT_NAME} SHARED ${DIR_SRCS})
    set_target_properties(${PROJECT_NAME} PROPERTIES VERSION
    ${INNO_SDK_VERSION_MAJOR}.${INNO_SDK_VERSION_MINOR}.${INNO_SDK_VERSION_PATCH} SOVERSION ${INNO_SDK_VERSION_MAJOR})
    target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND MAKE_SHARED)
    add_library(${PROJECT_NAME} SHARED ${DIR_SRCS})
    target_link_libraries(${PROJECT_NAME} PRIVATE pthread)
else()
    add_library(${PROJECT_NAME} ${DIR_SRCS})
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../../lib)
########################
# add_custom_command(TARGET ${PROJECT_NAME}
#     POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${PROJECT_SOURCE_DIR}/../../lib
#     COMMENT "Copying library to current working directory")
# #add_custom_target(copy_files ALL DEPENDS ${PROJECT_NAME})

# INSTALL(TARGETS ${PROJECT_NAME}
#     RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/../../bin
#     LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/../../lib
#     ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/../../lib
# )
