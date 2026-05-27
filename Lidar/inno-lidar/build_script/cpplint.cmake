find_package(PythonInterp)

set(STYLE_FILTER)

# disable unwanted filters and enable wanted filters
set(STYLE_FILTER ${STYLE_FILTER}-runtime/references,)
set(STYLE_FILTER ${STYLE_FILTER}-build/include_subdir,)
set(STYLE_FILTER ${STYLE_FILTER}-build/c++11,)
# Disable header_guard warning, Consider using #pragma once instead
set(STYLE_FILTER ${STYLE_FILTER}-build/header_guard,)
set(STYLE_FILTER ${STYLE_FILTER}+build/include_what_you_use)

# Add a target that runs cpplint.py
#
# Parameters:
# - TARGET_NAME the name of the target to add
# - SOURCES_LIST a complete list of source and include files to check
function(add_style_check_target TARGET_NAME SOURCES_LIST CPPLINT_PATH)

  if(NOT PYTHONINTERP_FOUND)
    return()
  endif()

  list(REMOVE_DUPLICATES SOURCES_LIST)
  list(SORT SOURCES_LIST)
  # message(${SOURCES_LIST})

  add_custom_target(${TARGET_NAME}
    COMMAND "${CMAKE_COMMAND}" -E chdir
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${PYTHON_EXECUTABLE}"
            "${CPPLINT_PATH}"
            "--filter=${STYLE_FILTER}"
            "--counting=detailed"
            "--extensions=hpp,cpp,cc,h"
            "--linelength=120"
            "--exclude=src/getopt.h"
            ${SOURCES_LIST}
    DEPENDS ${SOURCES_LIST}
    COMMENT "Linting ${TARGET_NAME}"
    VERBATIM)

endfunction()
