# get source files
file(
  GLOB 
  sources 
  *.F90
  *.c
  *.cxx
  *.hxx
  )

# get kernel source file
file(
  GLOB
  kernel_sources 
  kernels/*.F90
  kernels/*.c
  kernels/*.cxx
  kernels/*.hxx
  )

file(
  GLOB 
  to_remove
  debug.cxx)

# Source files to ignore for compilation
list(REMOVE_ITEM sources ${to_remove})
list(REMOVE_ITEM kernel_sources ${to_remove})

# message(STATUS "Kernel sources: ${kernel_sources}")
set(sources ${sources} ${kernel_sources})

if(${RUNTIME} MATCHES "StarPU")

  file(
    GLOB 
    starpu_sources 
    StarPU/*.F90
    StarPU/*.c
    StarPU/*.cxx
    StarPU/*.hxx)

  message(STATUS "StarPU sources: ${starpu_sources}")

  set(sources ${sources} ${starpu_sources})

endif()

message(STATUS "Sources: ${sources}")

add_library(spldlt ${sources})
