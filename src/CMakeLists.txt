set(sources
  AppendAlloc.hxx
  assemble.hxx
  BlockBase.hxx
  BlockUnsym.hxx
  BuddyAllocator.hxx
  factor_failed.hxx
  factor.hxx
  factor_indef.hxx
  factor_unsym.hxx
  NumericFrontBase.hxx
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
  SymbolicFront.hxx
  SymbolicTree.cxx
  SymbolicTree.hxx
  tasks_factor_indef.hxx
  tasks.hxx
  Tile.hxx
  topology.cxx
  topology.hxx)

set(tasks_sources
  tasks/assemble.hxx
  tasks/factor_failed.hxx
  tasks/form_contrib.hxx
  tasks/indef.hxx
  tasks/unsym.hxx)
  
# get kernel source file
file(
  GLOB
  kernel_sources 
  kernels/*.F90
  kernels/*.c
  kernels/*.cxx
  kernels/*.hxx
  )

file(
  GLOB 
  to_remove
  debug.cxx)

# Source files to ignore for compilation
list(REMOVE_ITEM sources ${to_remove})
list(REMOVE_ITEM kernel_sources ${to_remove})

# message(STATUS "Kernel sources: ${kernel_sources}")
set(sources ${sources} ${tasks_sources} ${kernel_sources})

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
    StarPU/kernels.cxx
    StarPU/kernels.hxx
    StarPU/kernels_indef.cxx
    StarPU/kernels_indef.hxx
    StarPU/kernels_unsym.cxx
    StarPU/kernels_unsym.hxx
    StarPU/scheduler.c
    StarPU/scheduler.h
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
    kernels/gpu/*.cu)

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
    
  else()
    
    add_library(sylver ${sources})

    cuda_add_library(
      sylver_cuda
      ${cuda_sources}
      ${starpu_cuda_sources})

    sylver_install_library(sylver_cuda)

  endif()

else()

  add_library(sylver ${sources})
endif()

sylver_install_library(sylver)

# message(STATUS "Sources: ${sources}")

