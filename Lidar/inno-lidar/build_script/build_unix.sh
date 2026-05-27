#!/bin/bash
# Get the current platform
platform=`uname -s`
# Check if the platform is Linux
if [ "$platform" == "Linux" ]; then
    # Check if the cross-compiler is for ARM
    if [[ "$(echo $CC | grep -o 'aarch64-linux-gnu')" ]]; then
        export ARCH_TAG=-arm
    else
        export ARCH_TAG=-x86
    fi
# Check if the platform is MacOS
elif [[ "$platform" == "Darwin" ]]; then
    export ARCH_TAG=-macos
# Check if the platform is Windows (MinGW)
elif [[ "$platform" == *"MINGW64_NT"* ]]; then
    export ARCH_TAG=-mingw64
elif command -v qcc >/dev/null 2>&1; then
    export ARCH_TAG=-qnx
fi

cd ../
echo "====== build clientsdk begin... "
build_dir="build_unix"
if [[ ! -d ${build_dir} ]]; then
    mkdir ${build_dir}
else
    rm -rf ${build_dir}/*
fi
cd ${build_dir}

if [[ "${ARCH_TAG}" ]]; then
    cmake_param="-DARCH_TAG=${ARCH_TAG} ${cmake_param}"
fi

cmake_param="-DCMAKE_BUILD_TYPE=Release ${cmake_param}"
if [[ "$shared" == "1" ]]; then
    cmake_param="${cmake_param} -DMAKE_SHARED=1"
fi
cmake ${cmake_param} ..
make clean && make -j1
if [[ "$shared" != "1" ]]; then
    make install
fi
cd -
if [[ -f lib/libinnolidarsdkclient.a || -f lib/libinnolidarsdkclient.so ]]; then
    echo "====== build inno_clientsdk success ^_^"
else
    echo "====== build inno_clientsdk failed!!!"
fi
cd build
