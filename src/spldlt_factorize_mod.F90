!> @file
!> @copyright 2016- The Science and Technology Facilities Council (STFC)
!> @author    Florent Lopez
module spldlt_factorize_mod
  use, intrinsic :: iso_c_binding
  ! use spral_ssids_datatypes
  use spral_ssids_cpu_iface ! fixme only
  use spral_ssids_fkeep, only : ssids_fkeep
  implicit none

  type numeric_tree_type
     logical(C_BOOL) :: posdef
     type(c_ptr) :: ctree ! pointer to the C structure
   contains
     procedure :: solve_fwd
     procedure :: solve_bwd
     procedure :: solve_diag_bwd
     procedure :: solve_diag
  end type numeric_tree_type

  ! Data generated during factorization phase 
  type spldlt_fkeep_type
     type(ssids_fkeep) :: fkeep
     ! Assemb;y tree
     type(numeric_tree_type) :: numeric_tree ! structure representing the numeric tree
   contains
     procedure :: solve
  end type spldlt_fkeep_type

  ! Numeric tree routines

  ! routine to create a numeric subtree from the symbolic one
  ! return a C ptr on the tree structure
  interface spldlt_create_numeric_tree_c
     type(c_ptr) function spldlt_create_numeric_tree_dlb( &
          posdef, fkeep, symbolic_tree, aval, child_contrib, options, stats) &
          bind(C, name="spldlt_create_numeric_tree_dbl")
       use, intrinsic :: iso_c_binding
       use spral_ssids_cpu_iface, only : cpu_factor_options, cpu_factor_stats
       implicit none
       logical(c_bool), value :: posdef
       type(c_ptr), value :: fkeep
       type(c_ptr), value :: symbolic_tree
       real(c_double), dimension(*), intent(in) :: aval
       type(c_ptr), dimension(*), intent(inout) :: child_contrib
       type(cpu_factor_options), intent(in) :: options ! SSIDS options
       type(cpu_factor_stats), intent(out) :: stats
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
     integer(c_int) function spldlt_tree_solve_fwd_dbl( &
          posdef, numeric_tree, nrhs, x, ldx) &
          bind(C, name="spldlt_tree_solve_fwd_dbl")
       use, intrinsic :: iso_c_binding
       logical(c_bool), value :: posdef
       type(c_ptr), value :: numeric_tree
       integer(c_int), value :: nrhs
       real(c_double), dimension(*), intent(inout) :: x
       integer(c_int), value :: ldx
     end function spldlt_tree_solve_fwd_dbl
  end interface spldlt_tree_solve_fwd_c

  ! backward solve
  interface spldlt_tree_solve_bwd_c
     integer(c_int) function spldlt_tree_solve_bwd_dbl( &
          posdef, numeric_tree, nrhs, x, ldx) &
          bind(C, name="spldlt_tree_solve_bwd_dbl")
       use, intrinsic :: iso_c_binding
       logical(c_bool), value :: posdef
       type(c_ptr), value :: numeric_tree
       integer(c_int), value :: nrhs
       real(c_double), dimension(*), intent(inout) :: x
       integer(c_int), value :: ldx       
     end function spldlt_tree_solve_bwd_dbl
  end interface spldlt_tree_solve_bwd_c

  ! backward solve with diagonal solve (indefinite case) 
  interface spldlt_tree_solve_diag_bwd_c
     integer(c_int) function spldlt_tree_solve_diag_bwd_dbl( &
          posdef, numeric_tree, nrhs, x, ldx) &
          bind(C, name="spldlt_tree_solve_diag_bwd_dbl")
       use, intrinsic :: iso_c_binding
       logical(c_bool), value :: posdef
       type(c_ptr), value :: numeric_tree
       integer(c_int), value :: nrhs
       real(c_double), dimension(*), intent(inout) :: x
       integer(c_int), value :: ldx       
     end function spldlt_tree_solve_diag_bwd_dbl
  end interface spldlt_tree_solve_diag_bwd_c

  ! Diagonal solve (indefinite case) 
  interface spldlt_tree_solve_diag_c
     integer(c_int) function spldlt_tree_solve_diag_dbl( &
          posdef, numeric_tree, nrhs, x, ldx) &
          bind(C, name="spldlt_tree_solve_diag_dbl")
       use, intrinsic :: iso_c_binding
       logical(c_bool), value :: posdef
       type(c_ptr), value :: numeric_tree
       integer(c_int), value :: nrhs
       real(c_double), dimension(*), intent(inout) :: x
       integer(c_int), value :: ldx       
     end function spldlt_tree_solve_diag_dbl
  end interface spldlt_tree_solve_diag_c

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

     subroutine c_get_contrib(posdef, subtree, n, val, ldval, rlist, ndelay, &
          delay_perm, delay_val, lddelay) &
          bind(C, name="spral_ssids_cpu_subtree_get_contrib_dbl")
       use, intrinsic :: iso_c_binding
       implicit none
       logical(C_BOOL), value :: posdef
       type(C_PTR), value :: subtree
       integer(C_INT) :: n
       type(C_PTR) :: val
       integer(C_INT) :: ldval
       type(C_PTR) :: rlist
       integer(C_INT) :: ndelay
       type(C_PTR) :: delay_perm
       type(C_PTR) :: delay_val
       integer(C_INT) :: lddelay
     end subroutine c_get_contrib

     type(c_ptr) function test_malloc(p, ptr_in) bind(C, name="test_malloc")
       use, intrinsic :: iso_c_binding
       implicit none
       integer(c_int), value :: p
       type(c_ptr), value :: ptr_in
     end function test_malloc
     
  end interface
  
contains

  ! ! Print useful information for debugging
  ! subroutine spldlt_print_debuginfo_c(cakeep, cfkeep, p) bind(C)
  !   use, intrinsic :: iso_c_binding
  !   use spral_ssids_akeep, only : ssids_akeep
  !   use spral_ssids_fkeep, only : ssids_fkeep
  !   use spral_ssids_subtree, only : numeric_subtree_base
  !   use spral_ssids_cpu_subtree, only : cpu_numeric_subtree
  !   implicit none

  !   type(c_ptr), value :: cakeep
  !   type(c_ptr), value :: cfkeep
  !   integer(c_int), value :: p ! Partition number, C-indexed
    
  !   type(ssids_akeep), pointer :: akeep => null()
  !   type(ssids_fkeep), pointer :: fkeep => null()
  !   integer :: part ! Partition number, Fortran-indexed
  !   class(numeric_subtree_base), pointer :: numeric_subtree_ptr => null()

  !   call c_f_pointer(cakeep, akeep)
  !   call c_f_pointer(cfkeep, fkeep)

  !   part = p+1

  !   ! print *, "[spldlt_print_debuginfo_c] part = ",  part

  !   select type(subtree_ptr => fkeep%subtree(part)%ptr)
  !   class is(cpu_numeric_subtree) ! factorize subtree on CPU

  !      ! print *, "[spldlt_print_debuginfo_c] size(fkeep%subtree) = ", size(fkeep%subtree) 
  !      ! write(*, '("[spldlt_print_debuginfo_c] c_loc(csubtree) = ", z16)') c_loc(subtree_ptr%csubtree)
  !      write(*, '("[spldlt_print_debuginfo_c] part = ", i5, " csubtree = ", z16)') part, subtree_ptr%csubtree

  !   end select
    
  ! end subroutine spldlt_print_debuginfo_c

  subroutine spldlt_get_contrib_c(cakeep, cfkeep, p, child_contrib_c) bind(C)
    use, intrinsic :: iso_c_binding
    use spral_ssids_akeep, only : ssids_akeep
    use spral_ssids_fkeep, only : ssids_fkeep
    use spral_ssids_contrib, only : contrib_type
    use spral_ssids_subtree, only : numeric_subtree_base ! Debug
    use spral_ssids_cpu_subtree, only : cpu_numeric_subtree ! Debug
    implicit none

    type(c_ptr), value :: cakeep
    type(c_ptr), value :: cfkeep
    integer(c_int), value :: p ! Partition number, C-indexed
    type(C_PTR), dimension(*), intent(in) :: child_contrib_c

    type(ssids_akeep), pointer :: akeep => null()
    type(ssids_fkeep), pointer :: fkeep => null()
    integer :: part ! Partition number, Fortran-indexed
    type(contrib_type), pointer :: contrib => null()

    ! Debug
    class(numeric_subtree_base), pointer :: subtree_ptr => null()
    type(C_PTR) :: cval, crlist, delay_perm, delay_val
    
    call c_f_pointer(cakeep, akeep)
    call c_f_pointer(cfkeep, fkeep)

    part = p+1

    ! print *, "[spldlt_get_contrib_c] contrib_idx = ", akeep%contrib_idx
    ! print *, "contrib_idx(part) = ", akeep%contrib_idx(part)

    call c_f_pointer(child_contrib_c(akeep%contrib_idx(part)), contrib)
    ! allocate(contrib)

    ! print *, "[spldlt_get_contrib_c] part = ", part
    ! print *, "[spldlt_get_contrib_c] child_contrib_c = ", child_contrib_c(akeep%contrib_idx(part))

    ! print *, "[spldlt_get_contrib_c] ready = ", contrib%ready

    ! select type(subtree_ptr => fkeep%subtree(part)%ptr)
    ! class is(cpu_numeric_subtree) ! factorize subtree on CPU

    ! print *, "[spldlt_get_contrib_c] part = ", part, " csubtree = ", subtree_ptr%csubtree 
       
    ! call c_get_contrib(subtree_ptr%posdef, subtree_ptr%csubtree, contrib%n, cval,        &
    !      contrib%ldval, crlist, contrib%ndelay, delay_perm, delay_val, &
    !      contrib%lddelay)
       
    ! end select

    ! contrib = fkeep%subtree(part)%ptr%get_contrib()
    ! contrib%ready = .true.
    
  end subroutine spldlt_get_contrib_c

  function spldlt_factor_subtree_cpu(&
       cpu_symb, posdef, val, child_contrib_c, coptions, cstats)
    use, intrinsic :: iso_c_binding
    use spral_ssids_subtree, only: numeric_subtree_base
    use spral_ssids_cpu_subtree, only : cpu_numeric_subtree, cpu_symbolic_subtree
    implicit none
    class(numeric_subtree_base), pointer :: spldlt_factor_subtree_cpu
    class(cpu_symbolic_subtree), target, intent(inout) :: cpu_symb
    logical(C_BOOL), intent(in) :: posdef
    real(c_double), dimension(*), intent(in) :: val
    type(c_ptr), dimension(*), intent(inout) :: child_contrib_c
    type(cpu_factor_options), intent(in) :: coptions ! SSIDS options
    type(cpu_factor_stats), intent(out) :: cstats

    type(cpu_numeric_subtree), pointer :: cpu_factor
    type(C_PTR) :: cscaling
    integer :: st

    ! Leave output as null until successful exit
    nullify(spldlt_factor_subtree_cpu)

    ! Allocate cpu_factor for output
    ! allocate(cpu_factor, stat=st)
    allocate(cpu_factor)

    cscaling = c_null_ptr
    ! if (present(scaling)) cscaling = C_LOC(scaling) ! TODO(Florent) Set scaling if needed

    cpu_factor%posdef = posdef 
    cpu_factor%symbolic => cpu_symb
    ! ptr = c_null_ptr
    ! cpu_factor%csubtree = test_malloc(p, ptr)

    cpu_factor%csubtree = &
         c_create_numeric_subtree(posdef, cpu_factor%symbolic%csubtree, &
         val, cscaling, child_contrib_c, coptions, cstats)
    if (cstats%flag .lt. 0) then
       call c_destroy_numeric_subtree(cpu_factor%posdef, cpu_factor%csubtree)
       deallocate(cpu_factor, stat=st)
       return
    end if
    ! Success, set result and return
    spldlt_factor_subtree_cpu => cpu_factor

    return

  end function spldlt_factor_subtree_cpu
  
  subroutine spldlt_factor_subtree_c( &
       cakeep, cfkeep, p, val, child_contrib_c, coptions, cstats) &
       bind(C, name="spldlt_factor_subtree_c")
    use spral_ssids_akeep, only : ssids_akeep
    use spral_ssids_fkeep, only : ssids_fkeep
    use spral_ssids_cpu_subtree, only : cpu_numeric_subtree, cpu_symbolic_subtree
    use spral_ssids_cpu_iface, only : cpu_factor_options
    ! use starpu_f_mod ! debug
    use spral_ssids_contrib, only : contrib_type
    implicit none

    type(c_ptr), value :: cakeep
    type(c_ptr), value :: cfkeep
    integer(c_int), value :: p ! Partition number, C-indexed
    real(c_double), dimension(*), intent(in) :: val
    type(c_ptr), dimension(*), intent(inout) :: child_contrib_c
    type(cpu_factor_options), intent(in) :: coptions ! SSIDS options
    type(cpu_factor_stats), intent(out) :: cstats ! Worker stats

    type(ssids_akeep), pointer :: akeep => null()
    type(ssids_fkeep), pointer :: fkeep => null()
    integer :: part
    type(cpu_numeric_subtree), pointer :: cpu_factor => null()
    type(c_ptr) :: ptr
    type(contrib_type), pointer :: contrib
    logical(C_BOOL) :: posdef

    call c_f_pointer(cakeep, akeep)
    call c_f_pointer(cfkeep, fkeep)

    part = p+1 ! p is C-indexed
    posdef = fkeep%pos_def
    ! posdef = .true.

    ! call c_f_pointer(child_contrib_c(akeep%contrib_idx(part)), contrib)
    call c_f_pointer(child_contrib_c(part), contrib)

    select type(subtree_ptr => akeep%subtree(part)%ptr)
    type is (cpu_symbolic_subtree)
       fkeep%subtree(part)%ptr => spldlt_factor_subtree_cpu( &
            subtree_ptr, posdef, val, child_contrib_c, coptions, cstats)
    end select

    !if (akeep%contrib_idx(part) .le. akeep%nparts) then
    ! There is a parent subtree to contribute to
    ! print *, "[spldlt_factor_subtree_c] part = ", part
    contrib = fkeep%subtree(part)%ptr%get_contrib()
    contrib%ready = .true.
    !end if

  end subroutine spldlt_factor_subtree_c
  
!   subroutine spldlt_factor_subtree_c(posdef, val, cakeep, cfkeep, p, child_contrib_c, coptions) &
!        bind(C, name="spldlt_factor_subtree_c")
!     use spral_ssids_datatypes
!     use spral_ssids_akeep, only : ssids_akeep
!     use spral_ssids_fkeep, only : ssids_fkeep
!     use spral_ssids_inform, only : ssids_inform
!     use spral_ssids_subtree, only : symbolic_subtree_base
!     use spral_ssids_cpu_subtree, only : cpu_numeric_subtree, cpu_symbolic_subtree
!     use spral_ssids_contrib, only : contrib_type
!     use, intrinsic :: iso_c_binding
!     implicit none

!     logical(C_BOOL), value :: posdef
!     real(c_double), dimension(*), intent(in) :: val
!     type(c_ptr), value :: cakeep
!     type(c_ptr), value :: cfkeep
!     integer(c_int), value :: p ! Partition number, C-indexed
!     type(C_PTR), dimension(*) :: child_contrib_c
!     ! type(c_ptr), dimension(:), allocatable, intent(inout) :: child_contrib_c
!     type(cpu_factor_options), intent(in) :: coptions ! SSIDS options

!     integer :: st ! Error management
!     type(ssids_akeep), pointer :: akeep => null()
!     type(ssids_fkeep), pointer :: fkeep => null()
!     type(ssids_inform) :: inform
!     integer :: part
!     type(cpu_numeric_subtree), pointer :: cpu_factor => null()
!     type(cpu_factor_stats) :: cstats
!     type(C_PTR) :: cscaling
!     class(symbolic_subtree_base), pointer :: subtree_ptr => null()
!     type(contrib_type), pointer :: contrib => null()
!     ! type(contrib_type) :: contrib
!     type(C_PTR) :: cval, crlist, delay_perm, delay_val
!     ! type(C_PTR) :: csubtree

!     ! print *, "[spldlt_factor_subtree_c] cakeep = ", cakeep, ", cfkeep = ", cfkeep
    
!     call c_f_pointer(cakeep, akeep)
!     call c_f_pointer(cfkeep, fkeep)

!     part = p+1 ! p is C-indexed
!     ! print *, "[spldlt_factor_subtree_c] npart", akeep%nparts, ", part = ", part
!     ! print *, "[spldlt_factor_subtree_c] contrib_idx = ", akeep%contrib_idx
!     ! Retrieve contrib structure associated with subtree
!     call c_f_pointer(child_contrib_c(akeep%contrib_idx(part)), contrib)

!     select type(subtree_ptr => akeep%subtree(part)%ptr)
!     class is(cpu_symbolic_subtree) ! factorize subtree on CPU

!        nullify(fkeep%subtree(part)%ptr)

!        ! Allocate cpu_factor for output
!        allocate(cpu_factor, stat=st)
!        if (st .ne. 0) goto 10
!        ! cpu_factor%symbolic => subtree_ptr

!        ! ! Call C++ factor routine
!        ! cpu_factor%posdef = posdef
!        ! cscaling = C_NULL_PTR ! TODO(Florent) Set scaling
!        ! cpu_factor%csubtree = c_null_ptr

!        ! cpu_factor%csubtree = &
!        !      c_create_numeric_subtree(posdef, cpu_factor%symbolic%csubtree, &
!        !      val, cscaling, child_contrib_c, coptions, cstats)

!        ! cpu_factor%csubtree = test_malloc()
!        call test_malloc(c_loc(cpu_factor%csubtree))
!        ! print *, "     flag = ", cstats%flag  
!        ! if (cstats%flag .lt. 0) then
!        !    call c_destroy_numeric_subtree(cpu_factor%posdef, cpu_factor%csubtree)
!        !    deallocate(cpu_factor, stat=st)
!        !    inform%flag = cstats%flag
!        !    return
!        ! end if

!        ! Extract to Fortran data structures
!        ! call cpu_copy_stats_out(cstats, inform)

!        ! print *, "     flag = ", inform%flag
!        ! print *, " maxfront = ", inform%maxfront
!        ! cpu_factor%csubtree = csubtree
!        ! print *, "cpu_factor%posdef = ", cpu_factor%posdef
!        ! print *, "[spldlt_factor_subtree_c] part = ", part, " cpu_factor%csubtree = ", cpu_factor%csubtree

!        ! Success, set result and return
!        fkeep%subtree(part)%ptr => cpu_factor

!        ! if (akeep%contrib_idx(part) .le. akeep%nparts) then

!        ! allocate(contrib)
          
!        ! contrib = fkeep%subtree(part)%ptr%get_contrib()
              
!        ! call c_get_contrib(cpu_factor%posdef, cpu_factor%csubtree, contrib%n, cval, &
!        !      contrib%ldval, crlist, contrib%ndelay, delay_perm, delay_val, &
!        !      contrib%lddelay)

!        ! contrib%ready = .true.
       
!        ! deallocate(contrib)
!        ! nullify(contrib)

!        ! end if

!     end select

!     ! if (akeep%contrib_idx(part) .le. akeep%nparts) then
!     !    contrib = fkeep%subtree(part)%ptr%get_contrib()
!     !    contrib%ready = .true.
!     ! end if

!     return

! 10  continue
    
!     print *, "[spldlt_factor_subtree_c] Error"
!     deallocate(cpu_factor, stat=st)
!     return
!   end subroutine spldlt_factor_subtree_c

  subroutine factor_core(spldlt_akeep, spldlt_fkeep, val, options, inform)
    use spral_ssids_datatypes
    use spral_ssids_cpu_iface 
    use spral_ssids_akeep, only : ssids_akeep
    use spral_ssids_fkeep, only : ssids_fkeep
    use spral_ssids_contrib, only : contrib_type
    use spral_ssids_inform, only : ssids_inform
    use spldlt_analyse_mod, only : spldlt_akeep_type
    implicit none

    type(spldlt_akeep_type), target, intent(in) :: spldlt_akeep
    type(spldlt_fkeep_type), target, intent(inout) :: spldlt_fkeep
    real(wp), dimension(*), target, intent(in) :: val ! A values (lwr triangle)
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(inout) :: inform

    type(ssids_akeep), pointer :: akeep => null()
    type(ssids_fkeep), pointer :: fkeep => null()
    ! integer, dimension(:), allocatable :: exec_loc_aux
    type(contrib_type), dimension(:), allocatable, target :: child_contrib
    type(C_PTR), dimension(:), allocatable :: child_contrib_c
    integer :: i
    type(c_ptr) :: cfkeep
    type(cpu_factor_options) :: coptions
    logical(c_bool) :: posdef
    type(cpu_factor_stats) :: cstats
    ! Error management
    character(50)  :: context      ! Procedure name (used when printing).
    integer :: st

    context = 'factor_core'

    akeep => spldlt_akeep%akeep
    fkeep => spldlt_fkeep%fkeep

    ! Allocate space for subtrees
    !allocate(fkeep%subtree(akeep%nparts), stat=st)
    allocate(fkeep%subtree(spldlt_akeep%nsubtrees), stat=st)
    if (st .ne. 0) goto 100

    ! allocate(child_contrib(akeep%nparts), stat=st)
    allocate(child_contrib(spldlt_akeep%nsubtrees), stat=st)
    ! allocate(exec_loc_aux(akeep%nparts), stat=st)
    if (st .ne. 0) goto 100
    ! exec_loc_aux = 0

    posdef = fkeep%pos_def

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

    cfkeep = c_loc(fkeep)
 
    spldlt_fkeep%numeric_tree%posdef = posdef
    call cpu_copy_options_in(options, coptions)
    spldlt_fkeep%numeric_tree%ctree = spldlt_create_numeric_tree_c( &
         posdef, cfkeep, spldlt_akeep%symbolic_tree_c, val, &
         child_contrib_c, coptions, cstats)

    ! Extract to Fortran data structures
    call cpu_copy_stats_out(cstats, inform)

    ! Cleanup Memory
    deallocate(child_contrib_c)
    ! deallocate(exec_loc_aux)
    deallocate(child_contrib)

    return
100 continue
    
    print *, "Error in ", context 
    
    return
  end subroutine factor_core

  subroutine spldlt_factorize(spldlt_akeep, spldlt_fkeep, posdef, val, &
       options, inform)
    use spral_ssids_datatypes
    use spral_ssids_inform, only : ssids_inform
    use spral_ssids_akeep, only : ssids_akeep
    use spral_ssids_fkeep, only : ssids_fkeep
    use spral_ssids_contrib, only : contrib_type
    use spral_ssids_subtree, only : numeric_subtree_base
    use spral_ssids_cpu_subtree, only : cpu_numeric_subtree
    use spldlt_analyse_mod
    use spldlt_datatypes_mod, only: spldlt_options
    implicit none
    
    type(spldlt_akeep_type), intent(in) :: spldlt_akeep
    type(spldlt_fkeep_type), target, intent(inout) :: spldlt_fkeep
    logical, intent(in) :: posdef 
    real(wp), dimension(*), target, intent(in) :: val ! A values (lwr triangle)
    type(spldlt_options), target, intent(in) :: options
    type(ssids_inform), intent(inout) :: inform

    type(ssids_options), pointer :: ssids_opts ! SSIDS options 
    ! type(ssids_akeep), pointer :: akeep
    type(ssids_fkeep), pointer :: fkeep => null()
    integer :: i

    ! Error management
    character(50)  :: context      ! Procedure name (used when printing).
    integer :: st

    context = 'spldlt_factor'

    fkeep => spldlt_fkeep%fkeep
    fkeep%pos_def = posdef
    ssids_opts => options%super

    !
    ! Perform scaling if required
    !
    ! if ((options%scaling .gt. 0) .or. present(scale)) then
    !    if (allocated(fkeep%scaling)) then
    !       if (size(fkeep%scaling) .lt. n) then
    !          deallocate(fkeep%scaling, stat=st)
    !          allocate(fkeep%scaling(n), stat=st)
    !       end if
    !    else
    !       allocate(fkeep%scaling(n), stat=st)
    !    end if
    !    if (st .ne. 0) go to 10
    ! else
    !    deallocate(fkeep%scaling, stat=st)
    ! end if

    ! TODO
    ! ...

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

    ! Call main factorization routine
    call factor_core(spldlt_akeep, spldlt_fkeep, val, ssids_opts, inform)

    ! if ((options%print_level .ge. 1) .and. (options%unit_diagnostics .ge. 0)) then
    !    write (options%unit_diagnostics,'(/a)') &
    !         ' Completed factorisation with:'
    !    write (options%unit_diagnostics, &
    !         '(a,2(/a,i12),2(/a,es12.4),5(/a,i12))') &
    !         ' information parameters (inform%) :', &
    !         ' flag                   Error flag                               = ',&
    !         inform%flag, &
    !         ' maxfront               Maximum frontsize                        = ',&
    !         inform%maxfront, &
    !         ' num_factor             Number of entries in L                   = ',&
    !         real(inform%num_factor), &
    !         ' num_flops              Number of flops performed                = ',&
    !         real(inform%num_flops), &
    !         ' num_two                Number of 2x2 pivots used                = ',&
    !         inform%num_two, &
    !         ' num_delay              Number of delayed eliminations           = ',&
    !         inform%num_delay, &
    !         ' rank                   Computed rank                            = ',&
    !         inform%matrix_rank, &
    !         ' num_neg                Computed number of negative eigenvalues  = ',&
    !         inform%num_neg
    ! end if

    return
100 continue
    
    print *, "[Error][spldlt_factorize] st: ", st

    return
  end subroutine spldlt_factorize

  ! Solve phase
  subroutine solve(spldlt_fkeep, spldlt_akeep, nrhs, x, ldx, inform)
    use spral_ssids_datatypes
    use spral_ssids_inform, only : ssids_inform
    use spral_ssids_fkeep, only : ssids_fkeep
    use spldlt_analyse_mod
    implicit none

    class(spldlt_fkeep_type), target :: spldlt_fkeep
    type(spldlt_akeep_type), intent(in) :: spldlt_akeep
    integer, intent(in) :: nrhs
    integer, intent(in) :: ldx
    real(wp), dimension(ldx,nrhs), intent(inout) :: x
    type(ssids_inform), intent(inout) :: inform

    real(wp), dimension(:,:), allocatable :: x2
    type(ssids_akeep) :: akeep
    type(ssids_fkeep), pointer :: fkeep

    integer :: r
    integer :: n
    integer :: part
    integer :: exec_loc
    
    logical(c_bool) :: posdef
    
    akeep = spldlt_akeep%akeep
    n = akeep%n

    posdef = spldlt_fkeep%fkeep%pos_def
    ! posdef = .true.

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
    ! do part = 1, akeep%nparts
    do part = 1, spldlt_akeep%nsubtrees
       if (akeep%subtree(part)%exec_loc .eq. -1) cycle
       call fkeep%subtree(part)%ptr%solve_fwd(nrhs, x2, n, inform)
       if (inform%stat .ne. 0) goto 100
    end do

    ! Perform solve
    ! Fwd solve
    call spldlt_fkeep%numeric_tree%solve_fwd(posdef, nrhs, x2, ldx)
    
    ! Bwd solve
    ! call spldlt_fkeep%numeric_tree%solve_bwd(nrhs, x, ldx)

    ! ! Diag solve
    ! do part = akeep%nparts, 1, -1
    !    if (akeep%subtree(part)%exec_loc .eq. -1) cycle
    !    call fkeep%subtree(part)%ptr%solve_diag(nrhs, x2, n, inform)
    !    if (inform%stat .ne. 0) goto 100
    ! end do

    ! ! Diag solve
    ! call spldlt_fkeep%numeric_tree%solve_diag(posdef, nrhs, x2, ldx)

    ! ! Bwd solve
    ! call spldlt_fkeep%numeric_tree%solve_bwd(posdef, nrhs, x2, ldx)

    ! ! Bwd solve
    ! do part = akeep%nparts, 1, -1
    !    if (akeep%subtree(part)%exec_loc .eq. -1) cycle
    !    call fkeep%subtree(part)%ptr%solve_bwd(nrhs, x2, n, inform)
    !    if (inform%stat .ne. 0) goto 100
    ! end do

    ! Diag and bwd solve
    call spldlt_fkeep%numeric_tree%solve_diag_bwd(posdef, nrhs, x2, ldx)

    ! Diag bwd solve
    !do part = akeep%nparts, 1, -1
    do part = spldlt_akeep%nsubtrees, 1, -1
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
  subroutine solve_fwd(numeric_tree, posdef, nrhs, x, ldx)
    use, intrinsic :: iso_c_binding
    use spral_ssids_datatypes
    implicit none
    
    class(numeric_tree_type), intent(in) :: numeric_tree
    logical(c_bool), intent(in) :: posdef
    integer, intent(in) :: nrhs
    integer, intent(in) :: ldx
    real(wp), dimension(*), intent(inout) :: x

    integer(c_int) :: flag ! return value

    ! print *, "solve fwd, x: ", x(1:ldx, 1)

    flag = spldlt_tree_solve_fwd_c(posdef, numeric_tree%ctree, nrhs, x, ldx)
    ! TODO error managment
    ! if(flag.ne.SSIDS_SUCCESS) inform%flag = flag

    ! print *, "solve fwd, x: ", x(1:ldx)

  end subroutine solve_fwd

  ! Bwd solve on numeric tree
  subroutine solve_bwd(numeric_tree, posdef, nrhs, x, ldx)
    use, intrinsic :: iso_c_binding
    use spral_ssids_datatypes
    implicit none

    class(numeric_tree_type), intent(in) :: numeric_tree
    logical(c_bool), intent(in) :: posdef
    integer, intent(in) :: nrhs
    integer, intent(in) :: ldx
    real(wp), dimension(*), intent(inout) :: x

    integer(c_int) :: flag ! return value

    flag = spldlt_tree_solve_bwd_c(posdef, numeric_tree%ctree, nrhs, x, ldx)
    ! TODO error managment
    ! if(flag.ne.SSIDS_SUCCESS) inform%flag = flag
  end subroutine solve_bwd
  
  ! Bwd and diag solve on numeric tree
  subroutine solve_diag_bwd(numeric_tree, posdef, nrhs, x, ldx)
    use, intrinsic :: iso_c_binding
    use spral_ssids_datatypes
    implicit none

    class(numeric_tree_type), intent(in) :: numeric_tree 
    logical(c_bool), intent(in) :: posdef
    integer, intent(in) :: nrhs
    integer, intent(in) :: ldx
    real(wp), dimension(*), intent(inout) :: x

    integer(c_int) :: flag ! return value

    flag = spldlt_tree_solve_diag_bwd_c(posdef, numeric_tree%ctree, nrhs, x, ldx)
    ! TODO error managment
    ! if(flag.ne.SSIDS_SUCCESS) inform%flag = flag
  end subroutine solve_diag_bwd

  ! Diag solve on numeric tree
  subroutine solve_diag(numeric_tree, posdef, nrhs, x, ldx)
    use, intrinsic :: iso_c_binding
    use spral_ssids_datatypes
    implicit none

    class(numeric_tree_type), intent(in) :: numeric_tree 
    logical(c_bool), intent(in) :: posdef
    integer, intent(in) :: nrhs
    integer, intent(in) :: ldx
    real(wp), dimension(*), intent(inout) :: x

    integer(c_int) :: flag ! return value

    flag = spldlt_tree_solve_diag_c(posdef, numeric_tree%ctree, nrhs, x, ldx)

  end subroutine solve_diag

  !> @Brief Solve phase
  !>
  !> @param x On entry, x must be set so that if i has been used to
  !> index a variable, x(i,j) is the corresponding component of the
  !> right-hand side for the jth system (j = 1,2,..., nrhs).  On exit,
  !> if i has been used to index a variable, x(i,j) holds solution for
  !> variable i to system j.
  !> @param ldx Leading dimension for x.
  subroutine spldlt_solve(spldlt_akeep, spldlt_fkeep, nrhs, x, ldx, options, &
       inform, job)
    use spral_ssids_datatypes
    use spral_ssids_inform, only : ssids_inform
    use spldlt_datatypes_mod, only: sylver_options
    use spldlt_analyse_mod, only: spldlt_akeep_type
    use sylver_inform_mod, only: sylver_inform
    implicit none

    type(spldlt_akeep_type), intent(in) :: spldlt_akeep ! Analyse data
    ! structures
    type(spldlt_fkeep_type), intent(inout) :: spldlt_fkeep ! Data
    ! structure holding factors and other factorization related
    ! information
    integer, intent(in) :: nrhs ! Number of rhs 
    integer, intent(in) :: ldx ! Leading dimension for x
    real(wp), dimension(ldx,nrhs), intent(inout), target :: x ! On
      ! entry, x must be set so that if i has been used to index a
      ! variable, x(i,j) is the corresponding component of the
      ! right-hand side for the jth system (j = 1,2,..., nrhs).  On
      ! exit, if i has been used to index a variable, x(i,j) holds
      ! solution for variable i to system j
    type(sylver_options), intent(in) :: options ! User options 
    type(sylver_inform), intent(out) :: inform
    integer, optional, intent(in) :: job ! used to indicate whether
      ! partial solution required
      ! job = 1 : forward eliminations only (PLX = B)
      ! job = 2 : diagonal solve (DX = B) (indefinite case only)
      ! job = 3 : backsubs only ((PL)^TX = B)
      ! job = 4 : diag and backsubs (D(PL)^TX = B) (indefinite case only)
      ! job absent: complete solve performed

    type(ssids_inform) :: ssids_info
    character(50) :: context  ! Procedure name (used when printing).

    context = 'spldlt_solve'

    ! ! Perform appropriate printing
    ! if ((options%print_level .ge. 1) .and. (options%unit_diagnostics .ge. 0)) then
    !    write (options%unit_diagnostics,'(//a)') &
    !         ' Entering ssids_solve with:'
    !    write (options%unit_diagnostics,'(a,4(/a,i12),(/a,i12))') &
    !         ' options parameters (options%) :', &
    !         ' print_level         Level of diagnostic printing        = ', &
    !         options%print_level, &
    !         ' unit_diagnostics    Unit for diagnostics                = ', &
    !         options%unit_diagnostics, &
    !         ' unit_error          Unit for errors                     = ', &
    !         options%unit_error, &
    !         ' unit_warning        Unit for warnings                   = ', &
    !         options%unit_warning, &
    !         ' nrhs                                                    = ', &
    !         nrhs
    !    if (nrhs .gt. 1) write (options%unit_diagnostics,'(/a,i12)') &
    !         ' ldx                                                     = ', &
    !         ldx
    ! end if

   call spldlt_fkeep%solve(spldlt_akeep, nrhs, x, ldx, ssids_info)

   ! Print useful info if requested
   call inform%print_flag(options, context)

  end subroutine spldlt_solve

end module spldlt_factorize_mod
