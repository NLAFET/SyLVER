# List tests drivers
set(examples)

if (${EXAMPLES} MATCHES "ON")
  list(APPEND examples "spldlt_simple_example.F90")
endif()

  
foreach(example ${examples})

    MESSAGE( STATUS "Example: " ${example} )

    get_filename_component(example_name ${example} NAME_WE)

    add_executable(${example_name} ${example})
    target_link_libraries(${example_name} sylver)
    target_include_directories(${example_name} PUBLIC ${CMAKE_BINARY_DIR}/src)
    target_link_libraries(${example_name} ${SPRAL_LIBRARIES})
    target_link_libraries(${example_name} ${LIBS})
    
endforeach()
