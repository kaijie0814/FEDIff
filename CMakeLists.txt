##
#  CMake script for the step-28 tutorial program:
##

# Set the name of the project and target:
set(TARGET "FEDiff")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#  file(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#  file(GLOB_RECURSE TARGET_INC  "include/*.h")
#  set(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
FILE(GLOB_RECURSE TARGET_SRC  "*.cc")
FILE(GLOB_RECURSE TARGET_INC  "*.h")
SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})

# Usually, you will not need to modify anything beyond this point...

cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 8.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  COMPONENTS grid_reordering PETSc
  )
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()

#
# Are all dependencies fullfilled?
#
IF(NOT DEAL_II_WITH_PETSC OR NOT DEAL_II_WITH_SLEPC)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_PETSC = ON
    DEAL_II_WITH_SLEPC = ON
which is required for this tutorial step."
    )
ENDIF()

deal_ii_initialize_cached_variables()
project(${TARGET})
deal_ii_invoke_autopilot()
