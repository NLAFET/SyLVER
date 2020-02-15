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

include(CheckCXXCompilerFlag)

#
# Tests whether the cxx compiler understands a flag.
# If so, add it to 'variable'.
#
# Usage:
#     enable_cxx_flag_if_supported(variable flag)
#

MACRO(enable_cxx_compiler_flag_if_supported _variable _flag)
  #
  # Clang is too conservative when reporting unsupported compiler flags.
  # Therefore, we promote all warnings for an unsupported compiler flag to
  # actual errors with the -Werror switch:
  #
  IF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    SET(_werror_string "-Werror ")
  ELSE()
    SET(_werror_string "")
  ENDIF()

  STRING(STRIP "${_flag}" _flag_stripped)
  SET(_flag_stripped_orig "${_flag_stripped}")

  #
  # Gcc does not emit a warning if testing -Wno-... flags which leads to
  # false positive detection. Unfortunately it later warns that an unknown
  # warning option is used if another warning is emitted in the same
  # compilation unit.
  # Therefore we invert the test for -Wno-... flags:
  #
  IF(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    STRING(REPLACE "-Wno-" "-W" _flag_stripped "${_flag_stripped}")
  ENDIF()

  # unset(CXX_COMPILER_FLAG_SUPPORTED)
  IF(NOT "${_flag_stripped}" STREQUAL "")
    STRING(REGEX REPLACE "^-" "" _flag_name "${_flag_stripped}")
    STRING(REPLACE "," "" _flag_name "${_flag_name}")
    STRING(REPLACE "-" "_" _flag_name "${_flag_name}")
    STRING(REPLACE "+" "_" _flag_name "${_flag_name}")
    CHECK_CXX_COMPILER_FLAG(
      "${_werror_string}${_flag_stripped}"
      SYLVER_HAVE_FLAG_${_flag_name}
      # CXX_COMPILER_FLAG_SUPPORTED
      )

    message("SYLVER_HAVE_FLAG_${_flag_name}")
    if (SYLVER_HAVE_FLAG_${_flag_name})
      message("Flag ${_flag_name} is supported")
    else()
      message("Flag ${_flag_name} is NOT supported")
    endif()

    # message("CXX_COMPILER_FLAG_SUPPORTED: ${CXX_COMPILER_FLAG_SUPPORTED}")    
    # set(SYLVER_HAVE_FLAG_${_flag_name} ${CXX_COMPILER_FLAG_SUPPORTED})
    
    IF(SYLVER_HAVE_FLAG_${_flag_name})
      SET(${_variable} "${${_variable}} ${_flag_stripped_orig}")
      STRING(STRIP "${${_variable}}" ${_variable})
    ENDIF()
  ENDIF()
ENDMACRO()
