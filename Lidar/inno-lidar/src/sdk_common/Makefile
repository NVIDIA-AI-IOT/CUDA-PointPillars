###
#  Copyright (C) 2023 - Innovusion Inc.
#
#  All Rights Reserved.
#
#  $Id$
##

PROJECT = innolidarsdkcommon
CPPLINT = ../../build/cpplint.py

include version_gen.mk


all: lint_checked $(PROJECT)

.PHONY: clean
clean:
	rm -rf build
	rm -f libinnolidarsdkcommon.*

$(PROJECT):
	@$(eval JOBS := $(shell echo $(MAKEFLAGS) | grep -o '\-j[[:digit:]]\+' | sed 's/-j//'))
	@$(if $(JOBS), , $(eval JOBS := 4))
	@$(eval MAKEFLAGS = "")
	mkdir -p build

ifeq ($(ARCH_TAG), -mingw64)
	cd build && cmake -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=../../build/toolchain_mingw.cmake $(CMAKE_OPTIONS) .. && make -j$(JOBS)
else
	cd build && cmake -G "Unix Makefiles" .. && make -j$(JOBS)
endif

ifeq (,$(filter $(ARCH_TAG), -mingw64 -qnx))
	cd build && cmake -G "Unix Makefiles" -DMAKE_SHARED=1 .. && make -j$(JOBS)
endif
	rm -rf build

lint_checked: $(wildcard *.h) $(wildcard *.cpp)
	$(CPPLINT) --linelength=120 --counting=detailed \
	--filter=-runtime/references,-build/include_subdir,-build/c++11,-build/header_guard,+build/include_what_you_use \
	--headers=hpp,h \
	--extensions=hpp,cpp,cc,h \
	--root=. $?

# modify the default action
.DEFAULT_GOAL := all
