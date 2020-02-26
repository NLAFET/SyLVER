!> @file
!> @copyright 2016- The Science and Technology Facilities Council (STFC)
!> @author    Florent Lopez
module sylver_ciface_mod
  use, intrinsic :: iso_c_binding
  implicit none
 
  !> @brief Interoperable subset of sylver_options structure
  type , bind(c) :: options_c
     integer(C_INT) :: print_level
     logical(C_BOOL) :: action
     real(C_DOUBLE) :: small
     real(C_DOUBLE) :: u
     real(C_DOUBLE) :: multiplier
     integer(C_LONG) :: small_subtree_threshold
     integer(C_INT) :: nb
     integer(C_INT) :: pivot_method
     integer(C_INT) :: failed_pivot_method
     integer(C_INT) :: cpu_topology
  end type options_c

  !> @brief Interoperable subset of sylver_inform structure
  type, bind(c) :: inform_c
     integer(C_INT) :: flag
     integer(C_INT) :: num_delay
     integer(C_INT) :: num_neg
     integer(C_INT) :: num_two
     integer(C_INT) :: num_zero
     integer(C_INT) :: maxfront
     integer(C_INT) :: not_first_pass
     integer(C_INT) :: not_second_pass
  end type inform_c
     
contains

  !> @brief Copy Fortran options structure into interoperable one
  subroutine copy_options_f2c(foptions, coptions)
    use sylver_datatypes_mod, only: sylver_options 
    implicit none

    type(sylver_options), intent(in) :: foptions
    type(options_c), intent(inout) :: coptions

    coptions%print_level    = foptions%print_level
    coptions%action          = foptions%action
    coptions%small          = foptions%small
    coptions%u              = foptions%u
    coptions%small_subtree_threshold     = foptions%small_subtree_threshold
    coptions%multiplier     = foptions%multiplier
    coptions%nb             = foptions%nb
    coptions%pivot_method   = min(3, max(1, foptions%pivot_method))
    coptions%failed_pivot_method = min(2, max(1, foptions%failed_pivot_method))
    coptions%cpu_topology = foptions%cpu_topology
    
  end subroutine copy_options_f2c

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> @brief Copy subset of ssids_inform from interoperable type
  subroutine copy_inform_c2f(cinform, finform)
    use sylver_inform_mod, only: sylver_inform 
    implicit none
    type(inform_c), intent(in) :: cinform
    type(sylver_inform), intent(inout) :: finform

    ! Combine stats
    if(cinform%flag < 0) then
       finform%flag = min(finform%flag, cinform%flag) ! error
    else
       finform%flag = max(finform%flag, cinform%flag) ! success or warning
    endif
    finform%num_delay    = finform%num_delay + cinform%num_delay
    finform%num_neg      = finform%num_neg + cinform%num_neg
    finform%num_two      = finform%num_two + cinform%num_two
    finform%maxfront     = max(finform%maxfront, cinform%maxfront)
    finform%not_first_pass = finform%not_first_pass + cinform%not_first_pass
    finform%not_second_pass = finform%not_second_pass + cinform%not_second_pass
    finform%matrix_rank  = finform%matrix_rank - cinform%num_zero
  end subroutine copy_inform_c2f

end module sylver_ciface_mod
