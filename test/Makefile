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

#Get default CUDA version installed by dep package
CUDAVERSION ?= cuda-$(shell dpkg -l | grep cuda-core | sed -e "s/ \{1,\}/ /g" | cut -d ' ' -f 3 | cut -d '.' -f 1,2 | sed -e "s/-.*//g" | sort -n | tail -n 1)

CHECK_CUDA := 0
CHECK_CUDA := $(shell if [ -x "/usr/local/$(CUDAVERSION)" ]; then echo 1; fi;)

CUDNN_PATH ?=

ifneq ($(CHECK_CUDA), 1)
    #no version info, use cuda default path
    CUDAVERSION := cuda
    CHECK_CUDA := $(shell if [ -x "/usr/local/$(CUDAVERSION)" ]; then echo 1; fi;)
    $(info USE Default CUDA DIR: /usr/local/$(CUDAVERSION))
    ifneq ($(CHECK_CUDA), 1)
        $(error $("Please install cuda package"))
    endif
endif

LIBDIR := lib64

TARGET_ARCH ?= $(shell uname -m)

$(info TARGET_ARCH: $(TARGET_ARCH))

ifeq ($(TARGET_ARCH), aarch64)
    ifeq ($(shell uname -m), aarch64)
        CC = g++
    else
        CC = aarch64-linux-gnu-g++
    endif
    NVCC = /usr/local/$(CUDAVERSION)/bin/nvcc -m64 -ccbin $(CC)
else ifeq ($(TARGET_ARCH), x86_64)
    CC = g++
    NVCC = /usr/local/$(CUDAVERSION)/bin/nvcc -m64
else
    $(error Auto-detection of platform failed. Please specify one of the following arguments to make: TARGET_ARCH=[aarch64|x86_64])
endif

CXXFLAGS        += -std=c++11
CCFLAGS         += -D_REENTRANT
LDFLAGS         += -Wl,--allow-shlib-undefined -pthread
#CCFLAGS         += -D_GLIBCXX_USE_CXX11_ABI=0

dbg ?= 0
# show libraries used by linker in debug mode
ifeq ($(dbg),1)
    $(info dbg: $(dbg))
	CCFLAGS     += -g
	NVCCFLAGS   += -G --ptxas-options=-v
	LDFLAGS += -Wl,--trace
endif

ifeq ($(TARGET_ARCH), x86_64)
CUDA_VERSION := $(shell cat /usr/local/$(CUDAVERSION)/targets/x86_64-linux/include/cuda.h |grep "define CUDA_VERSION" |awk '{print $$3}') 
endif
ifeq ($(TARGET_ARCH), ppc64le)
CUDA_VERSION := $(shell cat /usr/local/$(CUDAVERSION)/targets/ppc64le-linux/include/cuda.h |grep "define CUDA_VERSION" |awk '{print $$3}') 
endif
ifeq ($(TARGET_ARCH), aarch64)
CUDA_VERSION := $(shell cat /usr/local/$(CUDAVERSION)/targets/aarch64-linux/include/cuda.h |grep "define CUDA_VERSION" |awk '{print $$3}') 
endif

CUDA_VERSION := $(strip $(CUDA_VERSION))
$(info CUDA_VERSION: $(CUDA_VERSION))

ifeq ($(CUDA_VERSION),8000)
  SMS_VOLTA = 
else
  ifneq ($(TARGET_ARCH),ppc64le)
    ifeq ($(CUDA_VERSION),9000)
      SMS_VOLTA ?= 70 
    else
      SMS_VOLTA ?= 70 72
    endif
  else
    SMS_VOLTA ?= 70 
  endif
endif

ifeq ($(TARGET_ARCH), aarch64)
    ifeq ($(CUDA_VERSION), 9000)
      SMS_VOLTA := 62 70
    endif
endif

ifeq ($(CUDA_VERSION),10010)
SMS_TURING ?= 75
endif

# Gencode arguments
SMS ?= 30 35 50 53 60 61 $(SMS_VOLTA) $(SMS_TURING)
$(info SMS: $(SMS))

ifeq ($(GENCODE_FLAGS),)
    # Generate SASS code for each SM architecture listed in $(SMS)
    $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

    ifeq ($(SMS),)
        # Generate PTX code from SM 20
        GENCODE_FLAGS += -gencode arch=compute_53,code=sm_53
    endif
    # Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
    HIGHEST_SM := $(lastword $(sort $(SMS)))
    ifneq ($(HIGHEST_SM),)
        GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
    endif
endif

CUDA_CFLAGS := -I/usr/local/$(CUDAVERSION)/include
CUDA_LIBS   := -L/usr/local/$(CUDAVERSION)/$(LIBDIR) -lcudart_static -lrt -ldl -lpthread -lcudart

INCLUDE     :=
INCLUDE     += $(CUDA_CFLAGS)
INCLUDE     += -I/usr/include/

LIBRARIES   :=
LIBRARIES   += -L/usr/lib
LIBRARIES   += -L/usr/local/lib
LIBRARIES   += $(CUDA_LIBS)
LIBRARIES   += -lpthread
LIBRARIES   += -L/usr/lib/aarch64-linux-gnu/ -lnvinfer -lnvonnxparser -lz

OBJ_DIR         := obj
SRC_DIR         := ../src
OUTPUT          := output
TARGET          := $(OUTPUT)/demo

CPP_FILES       := $(wildcard $(SRC_DIR)/*.cpp)
CU_FILES        := $(wildcard $(SRC_DIR)/*.cu)

CPP_FILES       := $(notdir $(CPP_FILES))
CU_FILES        := $(notdir $(CU_FILES))

OBJ_FILES_CPP   := $(CPP_FILES:%.cpp=$(OBJ_DIR)/%.o)
OBJ_FILES_CU    := $(CU_FILES:%.cu=$(OBJ_DIR)/%.o)

all: $(OBJ_DIR) $(TARGET)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(OUTPUT)

$(OBJ_FILES_CPP): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(INCLUDE) $(CCFLAGS) $(CXXFLAGS) -O2 -fPIC -o $@ -c $<

$(OBJ_FILES_CU): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(INCLUDE) $(CXXFLAGS) $(CCFLAGS) $(NVCCFLAGS) -O2 -lineinfo $(GENCODE_FLAGS) -Xcompiler -fPIC -c $< -o $@

$(OBJ_DIR)/main.o: main.cpp
	$(CC) $(INCLUDE) $(CCFLAGS) $(CXXFLAGS) -O2 -fPIC -o $@ -c $<

$(TARGET): $(OBJ_FILES_CU) $(OBJ_FILES_CPP) $(OBJ_DIR)/main.o
	$(CC) $(CCFLAGS) $(CXXFLAGS) -O2 -o $@ $^ $(LIBRARIES) $(LIBRARY_FILES)
	@echo

clean:
	@rm -rf $(OBJ_DIR) $(OUTPUT) $(TARGET)

