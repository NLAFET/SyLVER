########################################
# Setup list of test drivers
# MESSAGE( STATUS "Libs: " ${LIBS} )
set(test_drivers)

list(APPEND test_drivers "spldlt_test.F90")
list(APPEND test_drivers "splu_test.F90")

foreach(test_driver ${test_drivers})
    
  get_filename_component(test_driver_name ${test_driver} NAME_WE)

  add_executable(${test_driver_name} ${test_driver})
  set_property(TARGET ${test_driver_name} PROPERTY LINKER_LANGUAGE Fortran)
  target_include_directories(${test_driver_name} PUBLIC ${CMAKE_BINARY_DIR}/src)
  target_link_libraries(${test_driver_name} sylver)
  target_include_directories(${test_driver_name} PUBLIC ${SPRAL_DIR})
  target_link_libraries(${test_driver_name} ${SPRAL_LIBRARIES})
  target_link_libraries(${test_driver_name} ${LIBS})

endforeach()
