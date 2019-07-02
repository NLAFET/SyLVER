!> @file
!> @copyright 2019- The Science and Technology Facilities Council (STFC)
!> @author    Florent Lopez
program main
  use sylver_mod
  use sylver_test_mod
  implicit none
  
  integer :: we_unit = 11
  character(len=6) :: we_file = "we.out"
  integer :: dl_unit = 12
  character(len=6) :: dl_file = "dl.out"

  integer :: ncpu
  integer :: ngpu
  
  !we_unit = 11
  !dl_unit = 12

  we_unit = 6
  dl_unit = 6

  ! warning flags

  if(we_unit.gt.6) open(unit=we_unit,file=we_file,status="replace")
  if(dl_unit.gt.6) open(unit=dl_unit,file=dl_file,status="replace")

  ncpu = 1
#if defined(SPLDLT_USE_GPU)
  ngpu = 1
#else
  ngpu = 0
#endif
  
  ! Initilaize SyLVER
  call spldlt_init(ncpu, ngpu)
  
  errors = 0

  call test_warnings
  call test_errors
  
  write(*, "(/a)") "=========================="
  write(*, "(a,i4)") "Total number of errors = ", errors

  ! Shutdown SpLDLT
  call spldlt_finalize()

  if(we_unit.gt.6) close(we_unit)
  if(dl_unit.gt.6) close(dl_unit)

  if(errors.ne.0) stop 1 ! ERROR CODE for make check script

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine test_warnings
    implicit none

    type(matrix_type) :: a
    type(sylver_options) :: options
    integer(long) :: ne
    integer, dimension(:), allocatable :: order
    real(wp), dimension(:,:), allocatable :: rhs
    real(wp), dimension(:,:), allocatable :: res
    real(wp), dimension(:,:), allocatable :: x
    real(wp), dimension(:), allocatable :: x1
    logical :: posdef
    integer :: i
    type(spldlt_akeep_type) :: akeep
    type(sylver_inform) :: info
    
    write(*,"(a)")
    write(*,"(a)") "================"
    write(*,"(a)") "Testing warnings"
    write(*,"(a)") "================"

    options%unit_warning = we_unit
    options%unit_diagnostics = dl_unit
    options%print_level = 2

    options%ordering = 0 ! supply the ordering

    write(*,"(/a/)") " * Testing warnings (columns)"

    call simple_mat(a)
    ne = a%ne
    if (allocated(order)) deallocate(order)
    allocate (order(1:a%n))
    do i = 1,a%n
       order(i) = i
    end do
    
    write(*,"(a)",advance="no") " * Testing out of range above............"
    call simple_mat(a,2)
    ne = a%ptr(a%n+1)-1
    a%ptr(a%n+1) = a%ptr(a%n+1) + 1
    a%row(ne+1) = -1
    a%val(ne+1) = 1.
    a%ne = ne + 1
    posdef = .true.
    call gen_rhs(a, rhs, x1, x, res, 1)
    call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
    call print_result(info%flag, SYLVER_WARNING_IDX_OOR)
    call chk_answer(posdef, a, akeep, options, rhs, x, res, &
      SYLVER_WARNING_IDX_OOR)
    call spldlt_akeep_free(akeep)

    write(*,"(a)",advance="no") " * Testing out of range below............"
    call simple_mat_lower(a,1)
    ne = a%ne
    a%ptr(a%n+1) = a%ptr(a%n+1) + 1
    a%row(ne+1) = a%n + 1
    a%val(ne+1) = 1.
    a%ne = ne + 1
    call gen_rhs(a, rhs, x1, x, res, 1)
    if (allocated(order)) deallocate(order)
    allocate (order(1:a%n))
    do i = 1,a%n
       order(i) = i
    end do
    call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
    call print_result(info%flag, SYLVER_WARNING_IDX_OOR)
    call chk_answer(posdef, a, akeep, options, rhs, x, res, &
         SYLVER_WARNING_IDX_OOR)
    call spldlt_akeep_free(akeep)

    write(*,"(a)",advance="no") " * Testing duplicates...................."
    call simple_mat(a,2)
    ne = a%ptr(a%n+1)-1
    a%ne = ne
    a%ptr(a%n+1) = a%ptr(a%n+1) + 1
    a%row(ne+1) = a%n 
    a%val(ne+1) = 10. 
    a%ne = ne + 1
    call gen_rhs(a, rhs, x1, x, res, 1)
    if (allocated(order)) deallocate(order)
    allocate (order(1:a%n))
    do i = 1,a%n
       order(i) = i
    end do
    call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
    call print_result(info%flag,SYLVER_WARNING_DUP_IDX)
    call chk_answer(posdef, a, akeep, options, rhs, x, res, &
         SYLVER_WARNING_DUP_IDX)
    call spldlt_akeep_free(akeep)

    write(*,"(a)",advance="no") " * Testing out of range and duplicates..."
    call simple_mat_lower(a,2)
    ne = a%ptr(a%n+1)-1
    a%ne = ne
    a%ptr(a%n+1) = a%ptr(a%n+1) + 2
    a%row(ne+1) = a%n + 1
    a%val(ne+1) = 10. 
    a%row(ne+2) = a%n 
    a%val(ne+2) = 10. 
    a%ne = ne + 2
    call gen_rhs(a, rhs, x1, x, res, 1)
    if (allocated(order)) deallocate(order)
    allocate (order(1:a%n))
    do i = 1,a%n
       order(i) = i
    end do
    call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
    call print_result(info%flag,SYLVER_WARNING_DUP_AND_OOR)
    call chk_answer(posdef, a, akeep, options, rhs, x, res, &
         SYLVER_WARNING_DUP_AND_OOR, fs=.true.)
    call spldlt_akeep_free(akeep)
    
    write(*,"(a)", advance="no") " * Testing missing diagonal entry (indef)....."
    a%ptr = (/ 1, 4, 5, 6, 7 /)
    a%row(1:6) = (/ 1, 2, 4,     2,    4,    4 /)
    a%val(1:6) = (/   10.0, 2.0, 3.0, &
         10.0, &
         4.0, &
         10.0 /)
    posdef = .false.
    call gen_rhs(a, rhs, x1, x, res, 1)
    if (allocated(order)) deallocate(order)
    allocate (order(1:a%n))
    do i = 1,a%n
       order(i) = i
    end do
    a%ne = a%ptr(a%n+1) - 1
    call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
    call print_result(info%flag, SYLVER_WARNING_MISSING_DIAGONAL)
    call chk_answer(posdef, a, akeep, options, rhs, x, res, &
         SYLVER_WARNING_MISSING_DIAGONAL)
    call spldlt_akeep_free(akeep)

    write(*,"(a)", advance="no") " * Testing missing diagonal and out of range.."
    posdef = .false.
    call simple_mat_lower(a)
    a%ptr = (/ 1, 4, 5, 6, 8 /)
    a%row(1:7) = (/ 1, 2, 4,     2,    4,    4,   -1 /)
    a%val(1:7) = (/   10.0, 2.0, 3.0, &
         10.0, &
         4.0, &
         10.0, 1.0 /)
    call gen_rhs(a, rhs, x1, x, res, 1)
    if (allocated(order)) deallocate(order)
    allocate (order(1:a%n))
    do i = 1,a%n
       order(i) = i
    end do
    a%ne = a%ptr(a%n+1) - 1
    call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
    call print_result(info%flag, SYLVER_WARNING_MISS_DIAG_OORDUP)
    call chk_answer(posdef, a, akeep, options, rhs, x, res, &
         SYLVER_WARNING_MISS_DIAG_OORDUP)
    call spldlt_akeep_free(akeep)

    write(*,"(a)",advance="no") " * Testing arrays min size (zero diag)........"
    posdef = .false.
    call simple_mat_zero_diag(a)
    a%ne = a%ptr(a%n+1) - 1
    call gen_rhs(a, rhs, x1, x, res, 1)
    call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
    call print_result(info%flag, SYLVER_WARNING_MISSING_DIAGONAL)
    call chk_answer(posdef, a, akeep, options, rhs, x, res, &
         SYLVER_WARNING_MISSING_DIAGONAL)
    call spldlt_akeep_free(akeep)

    write(*,"(a)", advance="no") " * Testing missing diagonal and duplicate....."
    posdef = .false.
    call simple_mat(a)
    a%ptr = (/ 1, 5, 6, 7, 8 /)
    a%row(1:7) = (/ 1, 2, 2, 4,     2,    4,    4 /)
    a%val(1:7) = (/   10.0, 2.0, 3.0, 1.0, &
         10.0, &
         4.0, &
         10.0 /)
    call gen_rhs(a, rhs, x1, x, res, 1)
    if (allocated(order)) deallocate(order)
    allocate (order(1:a%n))
    do i = 1,a%n
       order(i) = i
    end do
    a%ne = a%ptr(a%n+1) - 1
    call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
    call print_result(info%flag, SYLVER_WARNING_MISS_DIAG_OORDUP)
    call chk_answer(posdef, a, akeep, options, rhs, x, res, &
         SYLVER_WARNING_MISS_DIAG_OORDUP, fs=.true.)
    call spldlt_akeep_free(akeep)

    write(*,"(a)", advance="no") " * Testing analyse with structurally singular."
    posdef = .false.
    options%action = .true.
    call simple_sing_mat2(a)
    call gen_rhs(a, rhs, x1, x, res, 1)
    if (allocated(order)) deallocate(order)
    allocate(order(a%n))
    do i=1,a%n
       order(i) = i
    end do
    a%ne = a%ptr(a%n+1) - 1
    call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
    call print_result(info%flag, SYLVER_WARNING_ANAL_SINGULAR)

    call chk_answer(posdef, a, akeep, options, rhs, x, res, &
         SYLVER_WARNING_FACT_SINGULAR, fs=.true.)
    call spldlt_akeep_free(akeep)
    write(*,"(a)", advance="no") " * Testing analyse with structurally singular."
    posdef = .false.
    options%action = .true.
    call simple_sing_mat2(a)
    call gen_rhs(a, rhs, x1, x, res, 1)
    if (allocated(order)) deallocate(order)
    allocate(order(a%n))
    do i=1,a%n
       order(i) = i ! not set order(3) = 0 but code should still be ok
       ! provided order holds permutation
    end do
    call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
    call print_result(info%flag, SYLVER_WARNING_ANAL_SINGULAR)
    a%ne = a%ptr(a%n+1) - 1
    call chk_answer(posdef, a, akeep, options, rhs, x, res, &
         SYLVER_WARNING_FACT_SINGULAR, fs=.true.)
    call spldlt_akeep_free(akeep)
!!!!!!!!!

    posdef = .false.
    write(*,"(a)") " * Testing factor with singular matrix......."
    call simple_sing_mat(a)
    if (allocated(order)) deallocate(order)
    allocate (order(1:a%n))
    do i = 1,a%n
       order(i) = i
    end do
    a%ne = a%ptr(a%n+1) - 1
    call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
    if(info%flag < 0) then
       write(*, "(a,i4)") &
            "Unexpected error during analyse. flag = ", info%flag
       errors = errors + 1
       return
    endif
    options%action = .true.
    call gen_rhs(a, rhs, x1, x, res, 1)
    call chk_answer(posdef, a, akeep, options, rhs, x, res, &
         SYLVER_WARNING_FACT_SINGULAR, fs=.true.)
    call spldlt_akeep_free(akeep)

    posdef = .false.
    write(*,"(a)") " * Testing factor with match ord no scale...."
    call simple_mat(a)
    options%ordering = 2
    a%ne = a%ptr(a%n+1) - 1
    call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, val=a%val, check=.true.)
    if(info%flag < 0) then
       write(*, "(a,i4)") &
            "Unexpected error during analyse. flag = ", info%flag
       errors = errors + 1
       return
    endif
    call gen_rhs(a, rhs, x1, x, res, 1)
    call chk_answer(posdef, a, akeep, options, rhs, x, res, &
         SYLVER_WARNING_MATCH_ORD_NO_SCALE, fs=.true.)
    call spldlt_akeep_free(akeep)
    options%ordering = 0 ! restore

    ! write(*,"(/a/)") " * Testing warnings (coord)"

    ! call simple_mat(a)
    ! ne = a%ne

    ! write(*,"(a)",advance="no") " * Testing out of range above............"
    ! call simple_mat(a,2)
    ! ne = a%ptr(a%n+1)-1
    ! a%ptr(a%n+1) = a%ptr(a%n+1) + 1
    ! a%row(ne+1) = -1
    ! a%col(ne+1) = 1
    ! a%val(ne+1) = 1.
    ! a%ne = ne + 1
    ! posdef = .true.
    ! call gen_rhs(a, rhs, x1, x, res, 1)
    ! options%ordering = 1
    ! call ssids_analyse_coord(a%n, a%ne, a%row, a%col, akeep, options, info)
    ! call print_result(info%flag,SSIDS_WARNING_IDX_OOR)
    ! call chk_answer(posdef, a, akeep, options, rhs, x, res, &
    !      SSIDS_WARNING_IDX_OOR)
    ! call spldlt_akeep_free(akeep)

    ! write(*,"(a)", advance="no") " * Testing analyse struct singular and MC80.."
    ! posdef = .false.
    ! call simple_sing_mat(a)
    ! call gen_rhs(a, rhs, x1, x, res, 1)
    ! options%ordering = 2
    ! options%scaling = 3 ! scaling from Matching-based ordering
    ! call ssids_analyse_coord(a%n, a%ne, a%row, a%col, akeep, options, info, &
    !      val=a%val)
    ! call print_result(info%flag, SSIDS_WARNING_ANAL_SINGULAR)
    ! options%action = .true.
    ! a%ne = a%ptr(a%n+1) - 1
    ! call chk_answer(posdef, a, akeep, options, rhs, x, res, &
    !      SSIDS_WARNING_FACT_SINGULAR, fs=.true.)
    ! call ssids_free(akeep, cuda_error)

  end subroutine test_warnings

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine test_errors
   type(matrix_type) :: a
   type(sylver_options) :: options
   type(spldlt_akeep_type) :: akeep
   type(spldlt_fkeep_type) :: fkeep
   type(sylver_inform) :: info

   integer :: i
   integer :: n
   integer(long) :: ne
   logical :: posdef
   integer :: nrhs
   integer :: temp
   integer :: cuda_error
   integer, dimension(:), allocatable :: order
   real(wp), dimension(:,:), allocatable :: d, x
   real(wp), dimension(:), allocatable :: x1, d1
   type(random_state) :: state

   options%unit_error = we_unit
   options%unit_warning = we_unit

   write(*,"(/a)") "======================"
   write(*,"(a)") "Testing errors:"
   write(*,"(a)") "======================"

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   ! tests on call to ssids_analyse (entry by columns)

   write(*,"(/a)") " * Testing bad arguments spldlt_analyse (columns)"

   options%ordering = 0

   call simple_mat_lower(a)
   if (allocated(order)) deallocate(order)
   allocate(order(a%n))
   do i = 1,a%n
      order(i) = i
   end do
   write(*,"(a)",advance="no") " * Testing n<0..............................."
   n = -1
   call spldlt_analyse(akeep, n, a%ptr, a%row, options, info, order, check=.true.)
   call print_result(info%flag, SYLVER_ERROR_A_N_OOR)
   call spldlt_akeep_free(akeep)

   call simple_mat_lower(a)
   write(*,"(a)",advance="no") " * Testing ptr with zero component..........."
   temp = a%ptr(1)
   a%ptr(1) = 0
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
   call print_result(info%flag, SYLVER_ERROR_A_PTR)
   a%ptr(1) = temp ! reset ptr(1) value
   call spldlt_akeep_free(akeep)

   write(*,"(a)",advance="no") " * Testing non-monotonic ptr................."
   temp = a%ptr(2)
   a%ptr(2) = a%ptr(3)
   a%ptr(3) = temp
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
   call print_result(info%flag, SYLVER_ERROR_A_PTR)
   a%ptr(3) = a%ptr(2) ! reset ptr(3) value
   a%ptr(2) = temp ! reset ptr(2) value
   call spldlt_akeep_free(akeep)

   write(*,"(a)",advance="no") " * Testing all A%row oor....................."
   a%row(1:a%ptr(2)-1) = 0
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
   call print_result(info%flag, SYLVER_ERROR_A_ALL_OOR)
   call spldlt_akeep_free(akeep)

   write(*,"(a)",advance="no") " * Testing nemin oor........................."
   call simple_mat_lower(a)
   if (allocated(order)) deallocate(order)
   allocate(order(a%n))
   do i = 1,a%n
      order(i) = i
   end do
   options%nemin = -1
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
   call print_result(info%flag, SYLVER_SUCCESS)
   call spldlt_akeep_free(akeep)
   options%nemin = sylver_nemin_default ! Reset nemin value

 end subroutine test_errors
   
end program main
