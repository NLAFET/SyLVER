# get source files
file(
  GLOB 
  sources 
  *.F90
  *.c
  *.cxx
  *.hxx)

add_library(spldlt ${sources})
