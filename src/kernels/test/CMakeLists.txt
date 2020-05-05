include(${CMAKE_SOURCE_DIR}/cmake/create_test.cmake)

sylver_create_unit_test("chol_kernels.cxx")
sylver_create_unit_test("copy_backup.cxx")
sylver_create_unit_test("find_col_abs_max.cxx")
