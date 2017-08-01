module spldlt_factorize_mod
  use, intrinsic :: iso_c_binding
  ! use spral_ssids_datatypes
  use spral_ssids_cpu_iface ! fixme only
  use spral_ssids_fkeep, only : ssids_fkeep
  implicit none

  type numeric_tree_type
     type(c_ptr) :: ptr_c ! pointer to the C structure
   contains
     procedure :: solve_fwd
     procedure :: solve_bwd
     procedure :: solve_diag_bwd
  end type numeric_tree_type
  
  type spldlt_fkeep_type
     type(ssids_fkeep), pointer :: fkeep => null()
     ! Facored elimination tree
     type(numeric_tree_type) :: numeric_tree ! structure representing the numeric tree
   contains
     procedure :: solve
  end type spldlt_fkeep_type

  ! Numeric tree routines

  ! routine to create a numeric subtree from the symbolic one
  ! return a C ptr on the tree structure
  interface spldlt_create_numeric_tree_c
     type(c_ptr) function spldlt_create_numeric_tree_dlb(fkeep, symbolic_tree, aval, &
          child_contrib, exec_loc_aux, options) &
          bind(C, name="spldlt_create_numeric_tree_dbl")
       use, intrinsic :: iso_c_binding
       import :: cpu_factor_options
       implicit none
       type(c_ptr), value :: fkeep
       type(c_ptr), value :: symbolic_tree
       real(c_double), dimension(*), intent(in) :: aval
       type(C_PTR), dimension(*), intent(inout) :: child_contrib
       integer(C_INT), dimension(*), intent(in) :: exec_loc_aux
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

  ! SSIDS interfaces
  interface
     type(C_PTR) function c_create_numeric_subtree(posdef, symbolic_subtree, &
          aval, scaling, child_contrib, options, stats) &
          bind(C, name="spral_ssids_cpu_create_num_subtree_dbl")
       use, intrinsic :: iso_c_binding
       import :: cpu_factor_options, cpu_factor_stats
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: symbolic_subtree
       real(C_DOUBLE), dimension(*), intent(in) :: aval
       type(C_PTR), value :: scaling
       type(C_PTR), dimension(*), intent(inout) :: child_contrib
       type(cpu_factor_options), intent(in) :: options
       type(cpu_factor_stats), intent(out) :: stats
     end function c_create_numeric_subtree
     
     subroutine c_destroy_numeric_subtree(posdef, subtree) &
          bind(C, name="spral_ssids_cpu_destroy_num_subtree_dbl")
       use, intrinsic :: iso_c_binding
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
     end subroutine c_destroy_numeric_subtree
  end interface

contains

  subroutine spldlt_factor_subtree_c(posdef, val, cakeep, cfkeep, p, child_contrib_c, coptions) &
       bind(C, name="spldlt_factor_subtree_c")
    use spral_ssids_datatypes
    use spral_ssids_akeep, only : ssids_akeep
    use spral_ssids_fkeep, only : ssids_fkeep
    use spral_ssids_inform, only : ssids_inform
    use spral_ssids_subtree, only : symbolic_subtree_base
    use spral_ssids_cpu_subtree, only : cpu_numeric_subtree, cpu_symbolic_subtree
    use spral_ssids_contrib, only : contrib_type
    use, intrinsic :: iso_c_binding
    implicit none

    logical(C_BOOL), value :: posdef
    real(c_double), dimension(*), intent(in) :: val
    type(c_ptr), value :: cakeep
    type(c_ptr), value :: cfkeep
    integer(c_int), value :: p ! Partition number, C-indexed
    type(C_PTR), dimension(*), intent(inout) :: child_contrib_c
    ! type(c_ptr), dimension(:), allocatable, intent(inout) :: child_contrib_c
    type(cpu_factor_options), intent(in) :: coptions ! SSIDS options

    integer :: st ! Error management
    type(ssids_akeep), pointer :: akeep
    type(ssids_fkeep), pointer :: fkeep
    type(ssids_inform) :: inform
    integer :: part
    type(cpu_numeric_subtree), pointer :: cpu_factor => null()    
    type(cpu_factor_stats) :: cstats
    type(C_PTR) :: cscaling
    class(symbolic_subtree_base), pointer :: subtree_ptr => null()
    type(contrib_type), pointer :: contrib => null()
    ! type(C_PTR) :: csubtree
    
    call c_f_pointer(cakeep, akeep)
    call c_f_pointer(cfkeep, fkeep)

    part = p+1 ! p is C-indexed
    
    ! Retrieve contrib structure associated with subtree
    call c_f_pointer(child_contrib_c(akeep%contrib_idx(part)), contrib)

    select type(subtree_ptr => akeep%subtree(part)%ptr)
    class is(cpu_symbolic_subtree) ! factorize subtree on CPU

       nullify(fkeep%subtree(part)%ptr)

       ! Allocate cpu_factor for output
       allocate(cpu_factor, stat=st)
       if (st .ne. 0) goto 10
       cpu_factor%symbolic => subtree_ptr

       ! Call C++ factor routine
       cpu_factor%posdef = posdef
       cscaling = C_NULL_PTR ! TODO(Florent) Set scaling
       cpu_factor%csubtree = c_null_ptr 

       cpu_factor%csubtree = &
            c_create_numeric_subtree(posdef, cpu_factor%symbolic%csubtree, &
            val, cscaling, child_contrib_c, coptions, cstats)
       if (cstats%flag .lt. 0) then
          call c_destroy_numeric_subtree(cpu_factor%posdef, cpu_factor%csubtree)
          deallocate(cpu_factor, stat=st)
          inform%flag = cstats%flag
          return
       end if

       ! Extract to Fortran data structures
       call cpu_copy_stats_out(cstats, inform)

       print *, "     flag = ", inform%flag
       print *, " maxfront = ", inform%maxfront
       ! cpu_factor%csubtree = csubtree
       ! print *, "cpu_factor%posdef = ", cpu_factor%posdef
       print *, "cpu_factor%csubtree = ", cpu_factor%csubtree

       ! Success, set result and return
       fkeep%subtree(part)%ptr => cpu_factor

    end select

    if (akeep%contrib_idx(part) .le. akeep%nparts) then 
       contrib = fkeep%subtree(part)%ptr%get_contrib()
       contrib%ready = .true.
    end if

    return

10  continue
    
    print *, "[spldlt_factor_subtree_c] Error"
    deallocate(cpu_factor, stat=st)
    return
  end subroutine spldlt_factor_subtree_c

  subroutine spldlt_factor(posdef, val, spldlt_akeep, spldlt_fkeep, fkeep, options, inform)
    use spral_ssids_datatypes
    use spral_ssids_inform, only : ssids_inform
    use spral_ssids_akeep
    use spral_ssids_fkeep, only : ssids_fkeep
    use spral_ssids_contrib, only : contrib_type
    use spral_ssids_subtree, only : numeric_subtree_base
    use spral_ssids_cpu_subtree, only : cpu_numeric_subtree
    use spldlt_analyse_mod
    use, intrinsic :: iso_c_binding
    implicit none
    
    logical, intent(in) :: posdef 
    real(wp), dimension(*), target, intent(in) :: val ! A values (lwr triangle)
    type(spldlt_akeep_type), target :: spldlt_akeep
    type(spldlt_fkeep_type) :: spldlt_fkeep
    type(ssids_fkeep), target :: fkeep
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(inout) :: inform

    type(ssids_akeep), pointer :: akeep
    type(cpu_factor_options) :: coptions

    integer :: exec_loc
    integer, dimension(:), allocatable :: exec_loc_aux
    integer :: i
    type(contrib_type), dimension(:), allocatable, target :: child_contrib
    type(C_PTR), dimension(:), allocatable :: child_contrib_c

    ! Error management
    integer :: st

    spldlt_fkeep%fkeep => fkeep
    fkeep%pos_def = posdef

    print *, "[spldlt_factorize] posdef = ", fkeep%pos_def

    akeep => spldlt_akeep%akeep

    ! Setup data storage
    if (allocated(fkeep%subtree)) then
       do i = 1, size(fkeep%subtree)
          if (associated(fkeep%subtree(i)%ptr)) then
             call fkeep%subtree(i)%ptr%cleanup()
             deallocate(fkeep%subtree(i)%ptr)
          end if
       end do
       deallocate(fkeep%subtree)
    end if

    ! Allocate space for subtrees
    allocate(fkeep%subtree(akeep%nparts), stat=st)
    if (st .ne. 0) goto 100

    allocate(child_contrib(akeep%nparts), stat=st)
    allocate(exec_loc_aux(akeep%nparts), stat=st)
    if (st .ne. 0) goto 100
    exec_loc_aux = 0
    
    ! ! Factor subtrees
    ! do i = 1, akeep%nparts

    !    exec_loc = akeep%subtree(i)%exec_loc
    !    if (akeep%contrib_idx(i) .le. akeep%nparts) exec_loc_aux(akeep%contrib_idx(i)) = exec_loc
    !    if (exec_loc .eq. -1) cycle

    !    print *, "[spldlt_factorize] part = ", i

    !    ! TODO Use scaling if required
    !    fkeep%subtree(i)%ptr => akeep%subtree(i)%ptr%factor( &
    !         fkeep%pos_def, val, &
    !         child_contrib(akeep%contrib_ptr(i):akeep%contrib_ptr(i+1)-1), &
    !         options, inform &
    !         )
       
    !    if (akeep%contrib_idx(i) .gt. akeep%nparts) cycle ! part is a root
    !    child_contrib(akeep%contrib_idx(i)) = &
    !         fkeep%subtree(i)%ptr%get_contrib()
    !    child_contrib(akeep%contrib_idx(i))%ready = .true.

    ! end do

    ! Convert child_contrib to contrib_ptr
    allocate(child_contrib_c(size(child_contrib)), stat=st)
    if (st .ne. 0) goto 100
    do i = 1, size(child_contrib)
       child_contrib_c(i) = C_LOC(child_contrib(i))
    end do

    call cpu_copy_options_in(options, coptions)
    spldlt_fkeep%numeric_tree%ptr_c = spldlt_create_numeric_tree_c(c_loc(fkeep), &
         spldlt_akeep%symbolic_tree_c, val, child_contrib_c, exec_loc_aux, & 
         coptions)

    ! Free space
    deallocate(child_contrib_c)
    deallocate(exec_loc_aux)
    deallocate(child_contrib)

    return

100 continue
    
    print *, "[Error][spldlt_factorize] st: ", st

    return
  end subroutine spldlt_factor

  ! Solve phase
  subroutine solve(spldlt_fkeep, nrhs, x, ldx, spldlt_akeep, inform)
    use spral_ssids_datatypes
    use spral_ssids_inform, only : ssids_inform
    use spral_ssids_fkeep, only : ssids_fkeep
    use spldlt_analyse_mod
    implicit none

    class(spldlt_fkeep_type) :: spldlt_fkeep
    integer, intent(in) :: nrhs
    integer, intent(in) :: ldx
    real(wp), dimension(ldx,nrhs), intent(inout) :: x
    type(spldlt_akeep_type), intent(in) :: spldlt_akeep
    type(ssids_inform), intent(inout) :: inform

    real(wp), dimension(:,:), allocatable :: x2
    type(ssids_akeep) :: akeep
    type(ssids_fkeep), pointer :: fkeep

    integer :: r
    integer :: n
    integer :: part
    integer :: exec_loc
    
    akeep = spldlt_akeep%akeep
    n = akeep%n

    allocate(x2(n, nrhs), stat=inform%stat)
    if(inform%stat.ne.0) goto 100

    ! print *, "solve, nrhs: ", nrhs, ", ldx: ", ldx
    ! print *, "solve, x: ", x
    ! print *, "solve, n: ", n

    ! permute and scale
    ! TODO
    ! Just copy
    do r = 1, nrhs
       x2(1:n, r) = x(akeep%invp(1:n), r)
    end do

    fkeep => spldlt_fkeep%fkeep

    ! Subtree solves
    ! Fwd solve    
    do part = 1, akeep%nparts
       if (akeep%subtree(part)%exec_loc .eq. -1) cycle
       call fkeep%subtree(part)%ptr%solve_fwd(nrhs, x2, n, inform)
       if (inform%stat .ne. 0) goto 100
    end do

    ! Perform solve
    ! Fwd solve
    call spldlt_fkeep%numeric_tree%solve_fwd(nrhs, x2, ldx)
    
    ! Bwd solve
    ! call spldlt_fkeep%numeric_tree%solve_bwd(nrhs, x, ldx)

    ! Diag and bwd solve
    call spldlt_fkeep%numeric_tree%solve_diag_bwd(nrhs, x2, ldx)

    ! Bwd solve
    do part = akeep%nparts, 1, -1
       if (akeep%subtree(part)%exec_loc .eq. -1) cycle
       call fkeep%subtree(part)%ptr%solve_diag_bwd(nrhs, x2, n, inform)
       if (inform%stat .ne. 0) goto 100
    end do

    ! un-permute and un-scale
    ! TODO
    ! Just copy
    do r = 1, nrhs
       x(akeep%invp(1:n), r) = x2(1:n, r)
    end do

   100 continue
   inform%flag = SSIDS_ERROR_ALLOCATION
   return    
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

    ! print *, "solve fwd, x: ", x(1:ldx, 1)

    flag = spldlt_tree_solve_fwd_c(numeric_tree%ptr_c, nrhs, x, ldx)
    ! TODO error managment
    ! if(flag.ne.SSIDS_SUCCESS) inform%flag = flag

    ! print *, "solve fwd, x: ", x(1:ldx)

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
