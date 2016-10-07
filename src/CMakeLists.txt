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

# message(STATUS "Kernel sources: ${kernel_sources}")
set(sources ${sources} ${kernel_sources})

add_library(spldlt ${sources})
