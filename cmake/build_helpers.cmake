function(sylver_add_example example)
  get_filename_component(example_name ${example} NAME_WE)
  add_executable(${example_name} ${example})

  # Fortran mod files 
  target_include_directories(${example_name}
    PRIVATE
    ${CMAKE_BINARY_DIR}/src)

  target_link_libraries(${example_name} PRIVATE sylver)
  target_link_libraries(${example_name} PRIVATE ${LIBS})  
endfunction()
