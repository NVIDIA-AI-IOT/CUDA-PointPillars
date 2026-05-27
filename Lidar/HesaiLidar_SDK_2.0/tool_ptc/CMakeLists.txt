# ----------------------------------------#
# Hesai Tech <http://www.hesaitech.com>  #
# ----------------------------------------#

cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)
project(CommonModules)


set(CMAKE_BUILD_TYPE ON)

# Add each module's entry
set(WITH_PTCCLIENT ON CACHE BOOL "Build ptcClient code")
# set(WITH_PTCS_USE ON CACHE BOOL "Use Ptcs")

# 设置生成平台为 x64
if(MSVC)
add_compile_options("/source-charset:utf-8")
set(CMAKE_GENERATOR_PLATFORM x64)
if(WITH_PTCCLIENT)
    if(WITH_PTCS_USE)
        set(OPENSSL_ROOT_DIR "C:/Program Files/OpenSSL-Win64")
        set(OPENSSL_INCLUDE_DIR ${OPENSSL_ROOT_DIR}/include)
    endif(WITH_PTCS_USE)
endif(WITH_PTCCLIENT)
endif()
include_directories("${PROJECT_BINARY_DIR}")

if(WITH_PTCCLIENT)
    if(WITH_PTCS_USE)
        find_package(OpenSSL REQUIRED)
    endif(WITH_PTCS_USE)
endif(WITH_PTCCLIENT)

find_package(Threads REQUIRED)  

include_directories(
    ./
)

if(WITH_PTCCLIENT)
if(WITH_PTCS_USE)
    set(SSL_SOURCE ../libhesai/PtcClient/src/tcp_ssl_client.cc)
else()
    set(SSL_SOURCE "")
endif()
    add_library(ptcClient_lib 
        ../libhesai/PtcClient/src/ptc_client.cc
        ../libhesai/PtcClient/src/tcp_client.cc
        ../libhesai/PtcParser/src/general_ptc_parser.cc
        ../libhesai/PtcParser/src/ptc_1_0_parser.cc
        ../libhesai/PtcParser/src/ptc_2_0_parser.cc
        ../libhesai/PtcParser/ptc_parser.cc
        ${SSL_SOURCE}
    )
    target_link_libraries(ptcClient_lib
                        platutils_lib)
    if(WITH_PTCS_USE)
        target_link_libraries(ptcClient_lib
            ${OPENSSL_LIBRARIES})
    endif(WITH_PTCS_USE)
    set_target_properties(ptcClient_lib 
        PROPERTIES CXX_STANDARD_REQUIRED TRUE CXX_STANDARD 14)
    target_include_directories(ptcClient_lib PUBLIC   
        ../libhesai/PtcClient/include
        ../libhesai/PtcParser/include
        ../libhesai/PtcParser
        ../libhesai/) 
    if(WITH_PTCS_USE)
        target_include_directories(ptcClient_lib PUBLIC   
            ${OPENSSL_INCLUDE_DIR})
        target_compile_definitions(ptcClient_lib PUBLIC WITH_PTCS_USE=1)
    endif(WITH_PTCS_USE)
endif(WITH_PTCCLIENT)



# platutils lib
add_library(platutils_lib 
    ../libhesai/Common/src/plat_utils.cc
    ../libhesai/Common/src/auto_tick_count.cc)
target_link_libraries(platutils_lib
    log_lib)
set_target_properties(platutils_lib
    PROPERTIES CXX_STANDARD_REQUIRED TRUE CXX_STANDARD 14)
target_include_directories(platutils_lib PUBLIC   
    ../libhesai/Common/include
)
# log lib
add_library(log_lib 
    ../libhesai/Logger/src/logger.cc)
target_link_libraries(log_lib
    ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(log_lib
    PROPERTIES CXX_STANDARD_REQUIRED TRUE CXX_STANDARD 14)
target_include_directories(log_lib PUBLIC   
    ../libhesai/Logger/include
)


add_executable(ptc_upgrade
    ptc_upgrade.cc
)
target_link_libraries(ptc_upgrade
    ptcClient_lib
)

add_executable(ptc_tool
    ptc_tool.cc
)
target_link_libraries(ptc_tool
    ptcClient_lib
)