module spldlt_mod

contains

  ! Initialize SpLDLT solver, initialize runtime system.
  subroutine spldlt_init(ncpu)
#if defined(SPLDLT_USE_STARPU)
    use iso_c_binding
    use starpu_f_mod
#endif
    implicit none
    
    integer, intent(in) :: ncpu
#if defined(SPLDLT_USE_STARPU)
    integer(c_int) :: ret
#endif

#if defined(SPLDLT_USE_STARPU)
    ! initialize starpu
    ret = starpu_f_init(ncpu)
#endif

  end subroutine spldlt_init

  ! Shutdown SpLDLT solver, shutdown runtime system.
  subroutine spldlt_finalize()
#if defined(SPLDLT_USE_STARPU)
    use iso_c_binding
    use starpu_f_mod
    !$ use omp_lib
#endif
    implicit none

#if defined(SPLDLT_USE_STARPU)
    call starpu_f_shutdown()
#endif

  end subroutine spldlt_finalize

end module spldlt_mod
