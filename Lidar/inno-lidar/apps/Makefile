###
#  Copyright (C) 2023 - Innovusion Inc.
#
#  All Rights Reserved.
#
#  $Id$
##

.PHONY: build all example tools clean

build: tools example

all: build

example:
ifneq ($(wildcard example),)
	@$(MAKE) -C $@ all
endif


tools:
ifneq ($(wildcard tools),)
	@$(MAKE) -C $@ all
endif


clean:
ifneq ($(wildcard example),)
	@$(MAKE) -C example clean
endif

ifneq ($(wildcard tools),)
	@$(MAKE) -C tools clean
endif
