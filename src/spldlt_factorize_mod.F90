module spldlt_factorize_mod
  use, intrinsic :: iso_c_binding
  ! use spral_ssids_datatypes
  use spral_ssids_cpu_iface ! fixme only
  implicit none

  type numeric_tree_type
     type(c_ptr) :: ptr_c ! pointer to the C structure
   contains
     procedure :: solve_fwd
     procedure :: solve_bwd
     procedure :: solve_diag_bwd
  end type numeric_tree_type
  
  type spldlt_fkeep_type
     ! Facored elimination tree
     type(numeric_tree_type) :: numeric_tree ! structure representing the numeric tree
   contains
     procedure :: solve
  end type spldlt_fkeep_type

  ! Numeric tree routines

  ! routine to create a numeric subtree from the symbolic one
  ! return a C ptr on the tree structure
  interface spldlt_create_numeric_tree_c
     type(c_ptr) function spldlt_create_numeric_tree_dlb(symbolic_tree, aval, options) &
          bind(C, name="spldlt_create_numeric_tree_dbl")
       use, intrinsic :: iso_c_binding
       import :: cpu_factor_options
       implicit none
       type(c_ptr), value :: symbolic_tree
       real(c_double), dimension(*), intent(in) :: aval
       type(cpu_factor_options), intent(in) :: options ! SSIDS options
     end function spldlt_create_numeric_tree_dlb
  end interface spldlt_create_numeric_tree_c

  ! destroy the C ptr on numeric tree strucutre
  interface spldlt_destroy_numeric_tree_c
     subroutine spldlt_destroy_numeric_tree_dlb(numeric_tree) &
          bind(C, name="spldlt_destroy_numeric_tree_dbl")          
       use, intrinsic :: iso_c_binding
       type(c_ptr), value :: numeric_tree
     end subroutine spldlt_destroy_numeric_tree_dlb
  end interface spldlt_destroy_numeric_tree_c

  ! Solve routines

  ! forward solve
  interface spldlt_tree_solve_fwd_c
     integer(c_int) function spldlt_tree_solve_fwd_dbl(numeric_tree, nrhs, x, ldx) &
          bind(C, name="spldlt_tree_solve_fwd_dbl")
       use, intrinsic :: iso_c_binding
       type(c_ptr), value :: numeric_tree
       integer(c_int), value :: nrhs
       real(c_double), dimension(*), intent(inout) :: x
       integer(c_int), value :: ldx
     end function spldlt_tree_solve_fwd_dbl
  end interface spldlt_tree_solve_fwd_c

  ! backward solve
  interface spldlt_tree_solve_bwd_c
     integer(c_int) function spldlt_tree_solve_bwd_dbl(numeric_tree, nrhs, x, ldx) &
          bind(C, name="spldlt_tree_solve_bwd_dbl")
       use, intrinsic :: iso_c_binding
       type(c_ptr), value :: numeric_tree
       integer(c_int), value :: nrhs
       real(c_double), dimension(*), intent(inout) :: x
       integer(c_int), value :: ldx       
     end function spldlt_tree_solve_bwd_dbl
  end interface spldlt_tree_solve_bwd_c

  ! backward solve with diagonal solve (indefinite case) 
  interface spldlt_tree_solve_diag_bwd_c
     integer(c_int) function spldlt_tree_solve_diag_bwd_dbl(numeric_tree, nrhs, x, ldx) &
          bind(C, name="spldlt_tree_solve_diag_bwd_dbl")
       use, intrinsic :: iso_c_binding
       type(c_ptr), value :: numeric_tree
       integer(c_int), value :: nrhs
       real(c_double), dimension(*), intent(inout) :: x
       integer(c_int), value :: ldx       
     end function spldlt_tree_solve_diag_bwd_dbl
  end interface spldlt_tree_solve_diag_bwd_c
  
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
    type(cpu_factor_options) :: coptions

    akeep => spldlt_akeep%akeep

    call cpu_copy_options_in(options, coptions)
    spldlt_fkeep%numeric_tree%ptr_c = spldlt_create_numeric_tree_c( &
         spldlt_akeep%symbolic_tree_c, val, coptions)

  end subroutine spldlt_factorize

  ! Solve phase
  subroutine solve(spldlt_fkeep, nrhs, x, ldx)
    use spral_ssids_datatypes
    implicit none

    class(spldlt_fkeep_type) :: spldlt_fkeep
    integer, intent(in) :: nrhs
    integer, intent(in) :: ldx
    real(wp), dimension(ldx,nrhs), intent(inout) :: x

    print *, "solve"

    ! permute and scale
    ! TODO
    
    ! Perform solve
    ! Fwd solve
    call spldlt_fkeep%numeric_tree%solve_fwd(nrhs, x, ldx)

    ! Bwd solve
    ! call spldlt_fkeep%numeric_tree%solve_bwd(nrhs, x, ldx)

    ! Diag and bwd solve
    call spldlt_fkeep%numeric_tree%solve_diag_bwd(nrhs, x, ldx)

    ! un-permute and un-scale
    ! TODO
    
  end subroutine solve

  ! Fwd solve on numeric tree
  subroutine solve_fwd(numeric_tree, nrhs, x, ldx)
    use, intrinsic :: iso_c_binding
    use spral_ssids_datatypes
    implicit none
    
    class(numeric_tree_type), intent(in) :: numeric_tree 
    integer, intent(in) :: nrhs
    integer, intent(in) :: ldx
    real(wp), dimension(*), intent(inout) :: x

    integer(c_int) :: flag ! return value

    flag = spldlt_tree_solve_fwd_c(numeric_tree%ptr_c, nrhs, x, ldx)
    ! TODO error managment
    ! if(flag.ne.SSIDS_SUCCESS) inform%flag = flag
  end subroutine solve_fwd

  ! Bwd solve on numeric tree
 subroutine solve_bwd(numeric_tree, nrhs, x, ldx)
    use, intrinsic :: iso_c_binding
    use spral_ssids_datatypes
    implicit none
    
    class(numeric_tree_type), intent(in) :: numeric_tree 
    integer, intent(in) :: nrhs
    integer, intent(in) :: ldx
    real(wp), dimension(*), intent(inout) :: x

    integer(c_int) :: flag ! return value
    
    flag = spldlt_tree_solve_bwd_c(numeric_tree%ptr_c, nrhs, x, ldx)
    ! TODO error managment
    ! if(flag.ne.SSIDS_SUCCESS) inform%flag = flag
  end subroutine solve_bwd

  ! Bwd and diag solve on numeric tree
  subroutine solve_diag_bwd(numeric_tree, nrhs, x, ldx)
    use, intrinsic :: iso_c_binding
    use spral_ssids_datatypes
    implicit none

    class(numeric_tree_type), intent(in) :: numeric_tree 
    integer, intent(in) :: nrhs
    integer, intent(in) :: ldx
    real(wp), dimension(*), intent(inout) :: x

    integer(c_int) :: flag ! return value

    flag = spldlt_tree_solve_diag_bwd_c(numeric_tree%ptr_c, nrhs, x, ldx)
    ! TODO error managment
    ! if(flag.ne.SSIDS_SUCCESS) inform%flag = flag
  end subroutine solve_diag_bwd

end module spldlt_factorize_mod
