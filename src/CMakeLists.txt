# get source files
file(
  GLOB 
  sources 
  *.F90
  *.c
  *.cxx)

add_library(spldlt ${sources})
