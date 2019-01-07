!> \file
!> \copyright 2016- The Science and Technology Facilities Council (STFC)
!> \author    Florent Lopez
module splu_factorize_mod
  use, intrinsic :: iso_c_binding
  implicit none

  !
  ! Derived types
  !

  !> @brief structure containing Data generated during the
  !> factorization phase
  type splu_fkeep_type
     type(c_ptr) :: c_numeric_tree ! C pointer to numeric tree 
  end type splu_fkeep_type

  !
  ! Interfaces
  !

  ! routine to create a numeric subtree from the symbolic one
  ! return a C ptr on the tree structure
  interface splu_create_numeric_tree_c
     type(c_ptr) function splu_create_numeric_tree_dlb( &
          symbolic_tree, lval) &
          bind(C, name="splu_create_numeric_tree_dbl")
       use, intrinsic :: iso_c_binding
       ! use spral_ssids_cpu_iface, only : cpu_factor_options, cpu_factor_stats
       implicit none
       type(c_ptr), value :: symbolic_tree
       real(c_double), dimension(*), intent(in) :: lval
     end function splu_create_numeric_tree_dlb
  end interface splu_create_numeric_tree_c

contains

  subroutine factor_core(splu_akeep, splu_fkeep, lval)
    use spral_ssids_datatypes
    use splu_analyse_mod, only: splu_akeep_type
    implicit none

    type(splu_akeep_type), target, intent(in) :: splu_akeep ! Analysis data
    type(splu_fkeep_type), intent(inout) :: splu_fkeep ! Factorization data
    real(wp), dimension(*), intent(in) :: lval ! A values (whole matrix)

    ! Instanciate numeric tree using C interface
    splu_fkeep%c_numeric_tree = splu_create_numeric_tree_c( &
         splu_akeep%symbolic_tree_c, lval)

  end subroutine factor_core

  !> @brief Perfom sparse LU factorization of given matrix held in a
  !> CSC format.
  subroutine splu_factor(splu_akeep, splu_fkeep, ptr, row, val, options, inform)
    use spral_ssids_datatypes
    use spldlt_datatypes_mod
    use splu_analyse_mod, only: splu_akeep_type
    use spral_ssids_inform, only : ssids_inform
    implicit none
    
    type(splu_akeep_type), intent(in) :: splu_akeep
    type(splu_fkeep_type), intent(inout) :: splu_fkeep
    integer(long), intent(in) :: ptr(:) ! Col pointers (whole triangle)
    integer, intent(in) :: row(:) ! Row indices (whole triangle)
    real(wp), dimension(*), intent(in) :: val ! A values (whole matrix)
    type(sylver_options), intent(in) :: options
    type(ssids_inform), intent(inout) :: inform

    ! integer :: n
    ! integer(long) :: nz ! ptr(n+1)-1
    ! real(wp), allocatable :: lval(:) ! A value in lower triagle using CSC format (includes diag) 
    ! real(wp), allocatable :: uval(:) ! A value in upper triagle using CSR format (includes diag)

    ! Error management
    character(50)  :: context      ! Procedure name (used when printing).

    context = 'splu_factor'

    ! n = splu_akeep%akeep%n
    ! nz = ptr(n+1)-1
    
    ! FIXME size (nz/2 + n) would probably be enough for lval
    ! allocate(lval(nz), uval(nz))

    ! ! Fill lval array
    ! call get_l_val(n, nz, ptr, row, val, lval)

    ! ! Fill uval array
    ! call get_u_val(n, nz, ptr, row, val, uval)

    ! Compute factors
    call factor_core(splu_akeep, splu_fkeep, val)
    
  end subroutine splu_factor

  !> @brief Extract values in the lower triangle of input matrix and
  !> put them in lval using a CSC format.
  subroutine get_l_val(n, nz, ptr, row, val, lval)
    use spral_ssids_datatypes
    implicit none

    integer, intent(in) :: n ! Order of system
    integer(long), intent(in) :: nz ! Number of non-zero entries
    integer(long), intent(in) :: ptr(n+1) ! Col pointers (whole triangle)
    integer, intent(in) :: row(nz) ! Row indices (whole triangle)
    real(wp), intent(in) :: val(nz) ! A values (whole matrix)
    ! FIXME size (nz/2 + n) would probably be enough for lval
    real(wp), intent(out) :: lval(nz) ! A values (lower triangle)

    integer :: i,j
    integer(long) :: kk
    integer :: idx

    idx = 1
    do j = 1, n
       do kk = ptr(j), ptr(j+1) - 1
          i = row(kk)
          if (i .ge. j) then ! Lower triangle
             lval(idx) = val(kk)
             idx = idx + 1
          end if
       end do
    end do
    
    print *, 'lval = ', lval(1:idx-1)

  end subroutine get_l_val

  !> @brief Extract values in the upper triangle of a structurally
  !> symetric matrix and put them in lval using a CSR format.
  subroutine get_u_val(n, nz, ptr, row, val, uval)
    use spral_ssids_datatypes
    implicit none

    integer, intent(in) :: n ! Order of system
    integer(long), intent(in) :: nz ! Number of non-zero entries
    integer(long), intent(in) :: ptr(n+1) ! Col pointers (whole triangle)
    integer, intent(in) :: row(nz) ! Row indices (whole triangle)
    real(wp), intent(in) :: val(nz) ! A values (whole matrix)
    ! FIXME size (nz/2 + n) would probably be enough for lval
    real(wp), intent(out) :: uval(nz) ! A values (lower triangle)

    integer :: i,j
    integer(long) :: kk
    integer, allocatable :: next(:)
    integer :: sa
    
    allocate(next(n+1))

    ! next = ptr
    ! Figure out starting position in each row 
    sa = 1
    ! Here we exploit the fact that the matrix is structurally
    ! symmetric so we look into the columns
    do j = 1, n
       next(j) = sa
       do kk = ptr(j), ptr(j+1) - 1
          i = row(kk)
          if (i .ge. j) then
             sa = sa + 1
          end if
       end do
    end do

    ! Fill uval
    do j = 1, n
       do kk = ptr(j), ptr(j+1) - 1
          i = row(kk)
          if (i .le. j) then ! Upper triangle
             uval(next(i)) = val(kk)
             next(i) = next(i) + 1
          end if
       end do
    end do

    ! print *, 'uval = ', uval(1:sa-1)

  end subroutine get_u_val
  
end module splu_factorize_mod
