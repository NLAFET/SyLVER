!> @file
!> @copyright 2016- The Science and Technology Facilities Council (STFC)
!> @author    Florent Lopez
module spldlt_mod
  use spldlt_analyse_mod
  use spldlt_factorize_mod
  
contains

  !> @brief Initialize SpLDLT solver, initialize runtime system.
  ! TODO: Use topology structure to represent the machine
  subroutine spldlt_init(ncpu, ngpu)
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

  end subroutine spldlt_init

  !> @brief Shutdown SpLDLT solver, shutdown runtime system.
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

  !> @brief Release memory and cleanup data structure
  subroutine spldlt_free(spldlt_akeep, spldlt_fkeep)
    use spldlt_analyse_mod
    use spldlt_factorize_mod
    implicit none
    
    type(spldlt_akeep_type), intent(inout) :: spldlt_akeep
    type(spldlt_fkeep_type), intent(inout) :: spldlt_fkeep

    call spldlt_akeep%free()
    call spldlt_fkeep%free()

  end subroutine spldlt_free

end module spldlt_mod
