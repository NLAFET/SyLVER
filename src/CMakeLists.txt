set(sources
  AppendAlloc.hxx
  assemble.hxx
  BlockUnsym.hxx
  BuddyAllocator.hxx
  factor_failed.hxx
  factor.hxx
  factor_indef.hxx
  factor_unsym.hxx
  NumericFront.hxx
  NumericFrontUnsym.hxx
  NumericTree.cxx
  NumericTree.hxx
  NumericTreePosdef.cxx
  NumericTreePosdef.hxx
  NumericTreeUnsym.cxx
  NumericTreeUnsym.hxx
  spldlt_analyse_mod.F90
  spldlt_factorize_mod.F90
  spldlt_mod.F90
  spldlt_utils_mod.F90
  splu_analyse_mod.F90
  splu_factorize_mod.F90
  sylver_ciface.cxx
  sylver_ciface.hxx
  sylver_ciface_mod.F90
  sylver_datatypes_mod.F90
  sylver_inform_mod.F90
  sylver_mod.F90
  sylver_topology_mod.F90
  SymbolicTree.cxx
  topology.cxx
  topology.hxx)

if (SYLVER_BUILD_UNIT_TESTS)
  add_subdirectory(test)
endif()


set(tasks_sources
  tasks/assemble.hxx
  tasks/assemble_block.hxx
  tasks/factor_failed.hxx
  tasks/factor_indef.hxx
  tasks/form_contrib.hxx
  tasks/indef.hxx
  tasks/tasks.hxx
  tasks/unsym.hxx)
  
set(kernel_sources
  kernels/assemble.cxx
  kernels/assemble.hxx
  kernels/assemble_unsym.hxx
  kernels/factor.cxx
  kernels/factor.hxx
  kernels/factor_indef.hxx
  kernels/factor_unsym.hxx
  kernels/ldlt_app.cxx
  kernels/ldlt_app.hxx
  kernels/llt.hxx
  kernels/lu_app.hxx
  kernels/lu_nopiv.hxx
  kernels/lu_pp.hxx
  kernels/lu_tpp.hxx
  kernels/wrappers.cxx
  kernels/wrappers.hxx)

if (SYLVER_BUILD_UNIT_TESTS)
  add_subdirectory(kernels/test)
endif()


file(
  GLOB 
  to_remove
  debug.cxx)

# Source files to ignore for compilation
# list(REMOVE_ITEM sources ${to_remove})
list(REMOVE_ITEM kernel_sources ${to_remove})

# Interfaces
set(iface_sources
  interfaces/C/sylver_ciface.F90)

# message(STATUS "Kernel sources: ${kernel_sources}")
set(sources
  ${sources}
  ${iface_sources}
  ${tasks_sources}
  ${kernel_sources})

if(${SYLVER_RUNTIME} MATCHES "StarPU")

  set(starpu_sources
    StarPU/assemble.cxx
    StarPU/assemble.hxx
    StarPU/codelets.hxx
    StarPU/codelets_posdef.hxx
    StarPU/common.cxx
    StarPU/common.hxx
    StarPU/factor_failed.cxx
    StarPU/factor_failed.hxx
    StarPU/factor_indef.cxx
    StarPU/factor_indef.hxx
    StarPU/hlws.cxx
    StarPU/kernels.cxx
    StarPU/kernels.hxx
    StarPU/kernels_indef.cxx
    StarPU/kernels_indef.hxx
    StarPU/kernels_unsym.cxx
    StarPU/kernels_unsym.hxx
    StarPU/scheduler.c
    StarPU/scheduler.h
    StarPU/subtree.cxx
    StarPU/subtree.hxx
    StarPU/starpu.cxx
    StarPU/starpu_f_mod.F90
    StarPU/starpu_f_wrappers.c)
  
  # message(STATUS "StarPU sources: ${starpu_sources}")

  set(sources ${sources} ${starpu_sources})

  if(SYLVER_ENABLE_CUDA)

    set(
      starpu_cuda_sources
      StarPU/cuda/kernels.hxx
      StarPU/cuda/kernels_indef.hxx
      )

  endif()
  
endif()

if(SYLVER_ENABLE_CUDA)

  file(
    GLOB
    gpu_sources
    gpu/*.cxx
    gpu/*.hxx
    kernels/gpu/*.cxx
    kernels/gpu/*.hxx
    )  
  
  set(sources ${sources} ${gpu_sources})

  file(
    GLOB
    cuda_sources 
    kernels/gpu/*.cu
    kernels/cuda/block_ldlt.cu
    kernels/cuda/ldlt.cu
    )

  if (${HAVE_CUTLASS} MATCHES "OFF")
    file(
      GLOB 
      to_remove
      kernels/gpu/chol.cu)
  endif()

  # set(sources ${sources} ${cuda_sources})
  # set(sources ${cuda_sources})
  
  # MESSAGE( STATUS "CUDA source files: " ${cuda_sources} )
endif()

if(SYLVER_ENABLE_CUDA)

  if(SYLVER_NATIVE_CUDA)
    
    add_library(sylver
      ${sources}
      ${cuda_sources}
      ${starpu_cuda_sources})

    sylver_default_includes(sylver)

  else()
    
    add_library(sylver ${sources})

    cuda_add_library(
      sylver_cuda
      ${cuda_sources}
      ${starpu_cuda_sources})

    sylver_default_includes(sylver)
    sylver_default_includes(sylver_cuda)

    sylver_install_library(sylver_cuda)

  endif()

else()

  add_library(sylver ${sources})

  sylver_default_includes(sylver)

endif()

sylver_install_library(sylver)

# message(STATUS "Sources: ${sources}")

