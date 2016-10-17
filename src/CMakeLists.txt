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
message(STATUS "Sources: ${sources}")
set(sources ${sources} ${kernel_sources})

if(${RUNTIME} MATCHES "StarPU")

  file(
    GLOB 
    starpu_sources 
    StarPU/*.F90
    StarPU/*.c)

  set(sources ${sources} ${starpu_sources})

endif()

add_library(spldlt ${sources})
