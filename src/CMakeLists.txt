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
  sources 
  kernels/*.F90
  kernels/*.c
  kernels/*.cxx
  kernels/*.hxx
  )

add_library(spldlt ${sources})
