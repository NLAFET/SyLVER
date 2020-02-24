set(SYLVER_INSTALL_INCLUDE_DIR "include")
set(SYLVER_INSTALL_LIBRARY_DIR "lib")
set(SYLVER_INSTALL_PKGCONFIG_DIR "lib/pkgconfig")
set(SYLVER_INSTALL_CONFIG_DIR "lib/cmake/SyLVER")
set(SYLVER_INSTALL_MODULE_DIR "lib/cmake/SyLVER/Modules")

function(sylver_install_library name)
    # install .so and .a files
    install(TARGETS "${name}"
        EXPORT SyLVER
        LIBRARY DESTINATION ${SYLVER_INSTALL_LIBRARY_DIR}
        ARCHIVE DESTINATION ${SYLVER_INSTALL_LIBRARY_DIR}
        )
endfunction()

function(sylver_install)

  # install the public header files
  install(DIRECTORY "${SyLVER_SOURCE_DIR}/include/"
    DESTINATION "${SYLVER_INSTALL_INCLUDE_DIR}"
    FILES_MATCHING PATTERN "*.hpp"
    )
  install(DIRECTORY "${SyLVER_BINARY_DIR}/include/"
    DESTINATION "${SYLVER_INSTALL_INCLUDE_DIR}"
    FILES_MATCHING PATTERN "*.hpp"
    )
  install(DIRECTORY "${SyLVER_BINARY_DIR}/src/"
    DESTINATION "${SYLVER_INSTALL_INCLUDE_DIR}"
    FILES_MATCHING PATTERN "*.mod"
    )
  
endfunction()
