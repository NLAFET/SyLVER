module spldlt_factorize_mod
  use, intrinsic :: iso_c_binding
  implicit none

  type spldlt_fkeep_type
     ! Facored elimination tree
     type(c_ptr) :: numeric_tree_c
  end type spldlt_fkeep_type

  interface spldlt_create_symbolic_tree_c
     type(c_ptr) function spldlt_create_symbolic_tree(symbolic_tree, aval) &
          bind(C, name="spldlt_create_numeric_tree_dbl")
       use, intrinsic :: iso_c_binding
       type(c_ptr), value :: symbolic_tree
       real(c_double), dimension(*), intent(in) :: aval
     end function spldlt_create_symbolic_tree
  end interface spldlt_create_symbolic_tree_c

contains

  subroutine spldlt_factorize(val, spldlt_akeep, spldlt_fkeep, options)
    use spral_ssids_akeep
    use spldlt_analyse_mod
    implicit none
    
    real(wp), dimension(*), target, intent(in) :: val ! A values (lwr triangle)
    type(spldlt_akeep_type), target :: spldlt_akeep
    type(spldlt_fkeep_type) :: spldlt_fkeep
    type(ssids_options), intent(in) :: options

    type(ssids_akeep), pointer :: akeep

    akeep => spldlt_akeep%akeep

    spldlt_fkeep%numeric_tree_c = spldlt_create_symbolic_tree_c( &
         spldlt_akeep%symbolic_tree_c, val)

  end subroutine spldlt_factorize
  
end module spldlt_factorize_mod
