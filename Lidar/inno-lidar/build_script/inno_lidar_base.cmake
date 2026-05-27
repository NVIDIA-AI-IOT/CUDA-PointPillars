if((CMAKE_SYSTEM_NAME NOT STREQUAL "Windows") OR (DEFINED MINGW))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mno-ms-bitfields -w -mcrc32")
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -fPIC")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
    endif()
endif()

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Check if GCC supports -fmacro-prefix-map option
    check_cxx_compiler_flag(-fmacro-prefix-map HAS_MACRO_PREFIX_MAP_GCCXX)

    if(HAS_MACRO_PREFIX_MAP_GCCXX)
        message(STATUS "GCC C++ supports -fmacro-prefix-map")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmacro-prefix-map=${PROJECT_SOURCE_DIR}/=")
        add_definitions(-D_MACRO_PREFIX_MAP_)
    else()
        message(STATUS "GCC C++ does not support -fmacro-prefix-map")
    endif()
endif()


if (ARCH_TAG)
    message("## ARCH_TAG: ${ARCH_TAG}")
endif()

message("The CMake system name is: ${CMAKE_SYSTEM_NAME}")

if(CMAKE_SYSTEM_NAME MATCHES "Windows" AND NOT DEFINED MINGW)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -DINNO_EXPORTS -DWINDOWS_IGNORE_PACKING_MISMATCH /W1 /wd4319)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows" AND MINGW MATCHES "1")
    add_definitions(-DINNO_CLIENT)
    add_definitions(-D__MINGW64__ -D_WIN32=0 -D_GLIBCXX_USE_CXX11_ABI=0)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    add_definitions(-D__APPLE__)
else()
    # Use new C++11 ABI (default) to match libhv-1.3.3
    # Removed: add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

set(INNO_SDK_VERSION "")

# find_package(Git REQUIRED)
# execute_process(
#   COMMAND ${GIT_EXECUTABLE} describe --tags
#   WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/../../"
#   OUTPUT_VARIABLE INNO_SDK_VERSION
#   OUTPUT_STRIP_TRAILING_WHITESPACE
# )

#message("git tag:" ${INNO_SDK_VERSION})
if(INNO_SDK_VERSION STREQUAL "")
    if (EXISTS ${PROJECT_SOURCE_DIR}/../../src/sdk_common/version_gen.mk)
        file(STRINGS "${PROJECT_SOURCE_DIR}/../../src/sdk_common/version_gen.mk" INNO_SDK_VERSION)
    else()
        file(STRINGS "${PROJECT_SOURCE_DIR}/../../../src/sdk_common/version_gen.mk" INNO_SDK_VERSION)
    endif()
    string(REGEX MATCH "DYNA_LIB_MAJ=([0-9]+)" INNO_SDK_VERSION_MAJOR ${INNO_SDK_VERSION})
    string(REGEX REPLACE "DYNA_LIB_MAJ=([0-9]+)" "\\1" INNO_SDK_VERSION_MAJOR ${INNO_SDK_VERSION_MAJOR})
    string(REGEX MATCH "DYNA_LIB_MIN=([0-9]+)" INNO_SDK_VERSION_MINOR ${INNO_SDK_VERSION})
    string(REGEX REPLACE "DYNA_LIB_MIN=([0-9]+)" "\\1" INNO_SDK_VERSION_MINOR ${INNO_SDK_VERSION_MINOR})
    string(REGEX MATCH "DYNA_LIB_BUILD=([0-9]+)" INNO_SDK_VERSION_PATCH ${INNO_SDK_VERSION})
    string(REGEX REPLACE "DYNA_LIB_BUILD=([0-9]+)" "\\1" INNO_SDK_VERSION_PATCH ${INNO_SDK_VERSION_PATCH})
else()
    string(REGEX  REPLACE "^release-([0-9]+)\\..*" "\\1" INNO_SDK_VERSION_MAJOR "${INNO_SDK_VERSION}")
    string(REGEX REPLACE "^release-[0-9]+\\.([0-9]+).*" "\\1" INNO_SDK_VERSION_MINOR "${INNO_SDK_VERSION}")
    string(REGEX REPLACE "^release-[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" INNO_SDK_VERSION_PATCH "${INNO_SDK_VERSION}")
endif()
message("version:" ${INNO_SDK_VERSION})
include_directories(${PROJECT_SOURCE_DIR}/../..)
include_directories(${PROJECT_SOURCE_DIR}/../../..)
include_directories(${PROJECT_SOURCE_DIR}/../../src)
include_directories(${PROJECT_SOURCE_DIR}/../../../src)
link_directories(${PROJECT_SOURCE_DIR}/../../lib)
link_directories(${PROJECT_SOURCE_DIR}/../../../lib)

########################
if(NOT DEFINED CPPLINT_PATH)
    set(CPPLINT_PATH "${PROJECT_SOURCE_DIR}/../../build/cpplint.py")
endif()
if(EXISTS ${CPPLINT_PATH})
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../../build")
    include(cpplint)
    add_style_check_target(${PROJECT_NAME}_cpplint "${SOURCES_GET_PCD} ${SOURCES_SIMPLE_GET_PCD}" ${CPPLINT_PATH})
endif()
