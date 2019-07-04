!> @file
!> @copyright 2016- The Science and Technology Facilities Council (STFC)
!> @author Florent Lopez
module spldlt_mod
  use spldlt_analyse_mod
  use spldlt_factorize_mod
  
contains

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
