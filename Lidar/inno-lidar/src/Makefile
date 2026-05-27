###
#  Copyright (C) 2023 - Innovusion Inc.
#
#  All Rights Reserved.
#
#  $Id$
##
.PHONY: all build utils sdk_common sdk_client clean

all: build

build: utils sdk_common sdk_client

utils:
	@$(MAKE) -C $@

sdk_common: | utils
	@$(MAKE) -C $@

sdk_client: | sdk_common
	@$(MAKE) -C $@

clean:
	@$(MAKE) -C utils clean
	@$(MAKE) -C sdk_common clean
	@$(MAKE) -C sdk_client clean
