module sylver_mod
  use sylver_datatypes_mod
  use spldlt_mod

contains

  !> @brief Initialize SyLVER and starts runtime system.
  subroutine sylver_init(ncpu, ngpu)
#if defined(SPLDLT_USE_STARPU)
    use iso_c_binding
    use starpu_f_mod
#endif
    implicit none
    
    integer, intent(in) :: ncpu ! Number of CPU workers
    integer, intent(in) :: ngpu ! Number of GPU workers
#if defined(SPLDLT_USE_STARPU)
    integer(c_int) :: ret
#endif

#if defined(SPLDLT_USE_STARPU)
    ! initialize starpu
    ret = starpu_f_init(ncpu, ngpu)
#endif

  end subroutine sylver_init

  !> @brief Shutdown SpLDLT solver, shutdown runtime system.
  subroutine sylver_finalize()
#if defined(SPLDLT_USE_STARPU)
    use iso_c_binding
    use starpu_f_mod
    !$ use omp_lib
#endif
    implicit none

#if defined(SPLDLT_USE_STARPU)
#if defined(SPLDLT_USE_GPU)
    call starpu_f_cublas_shutdown();
#endif

    call starpu_f_shutdown()
#endif

  end subroutine sylver_finalize

end module sylver_mod
