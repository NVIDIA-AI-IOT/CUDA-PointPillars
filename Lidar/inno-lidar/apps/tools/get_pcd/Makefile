###
#  Copyright (C) 2023 - Innovusion Inc.
#
#  All Rights Reserved.
#
#  $Id$
##

.PHONY: all clean


all:
	@$(eval JOBS := $(shell echo $(MAKEFLAGS) | grep -o '\-j[[:digit:]]\+' | sed 's/-j//'))
	@$(if $(JOBS), , $(eval JOBS := 4))
	@$(eval MAKEFLAGS = "")
	mkdir -p build
	cd build && cmake -G "Unix Makefiles" .. && make -j$(JOBS) && make install
	rm -rf build


clean:
	rm -rf build
	rm -f get_pcd
