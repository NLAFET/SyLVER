!> \file
!> \copyright 2016- The Science and Technology Facilities Council (STFC)
!> \author    Florent Lopez
module sylver_ciface_mod
  use, intrinsic :: iso_c_binding
  implicit none
 
  !> @brief Interoperable subset of sylver_options
  type , bind(C) :: sylver_options_c
     integer(C_INT) :: print_level
     real(C_DOUBLE) :: small
     real(C_DOUBLE) :: u
     real(C_DOUBLE) :: multiplier
     integer(C_INT) :: nb
     integer(C_INT) :: pivot_method
     integer(C_INT) :: failed_pivot_method
  end type sylver_options_c

contains

  !> @brief Copy Fortran options structure into interoperable one
  subroutine copy_options_f2c(foptions, coptions)
    use spldlt_datatypes_mod, only: sylver_options 
    implicit none

    type(sylver_options), intent(in) :: foptions
    type(sylver_options_c), intent(inout) :: coptions

    coptions%print_level    = foptions%print_level
    coptions%small          = foptions%small
    coptions%u              = foptions%u
    coptions%multiplier     = foptions%multiplier
    coptions%nb             = foptions%nb
    coptions%pivot_method   = min(3, max(1, foptions%pivot_method))
    coptions%failed_pivot_method = min(2, max(1, foptions%failed_pivot_method))

  end subroutine copy_options_f2c

end module sylver_ciface_mod
