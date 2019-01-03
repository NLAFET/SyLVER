!> \file
!> \copyright 2016- The Science and Technology Facilities Council (STFC)
!> \author    Florent Lopez
module splu_factorize_mod
  use, intrinsic :: iso_c_binding
  implicit none

  !> @brief structure containing Data generated during the
  !> factorization phase
  type splu_fkeep_type
     type(c_ptr) :: c_numeric_tree ! C pointer to numeric tree 
  end type splu_fkeep_type

contains

  !> @brief Perfom sparse LU factorization of given matrix held in a
  !> CSC format.
  subroutine splu_factor(spldlt_akeep, splu_fkeep, ptr, row, val, options, inform)
    use spral_ssids_datatypes
    use spldlt_datatypes_mod
    use spldlt_analyse_mod, only: spldlt_akeep_type
    use spral_ssids_inform, only : ssids_inform
    implicit none
    
    type(spldlt_akeep_type), intent(in) :: spldlt_akeep
    type(splu_fkeep_type), intent(inout) :: splu_fkeep
    integer(long), intent(in) :: ptr(:) ! Col pointers (whole triangle)
    integer, intent(in) :: row(:) ! Row indices (whole triangle)
    real(wp), dimension(*), intent(in) :: val ! A values (whole matrix)
    type(splu_options), intent(in) :: options
    type(ssids_inform), intent(inout) :: inform

    ! Error management
    character(50)  :: context      ! Procedure name (used when printing).

    context = 'splu_factor'

    print *, context 
    
  end subroutine splu_factor

end module splu_factorize_mod
