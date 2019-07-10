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

  ncpu = 4
#if defined(SPLDLT_USE_GPU)
  ngpu = 1
#else
  ngpu = 0
#endif
  
  ! Initilaize SyLVER
  call sylver_init(ncpu, ngpu)
  
  errors = 0

  call test_warnings
  call test_errors
  call test_special
  
  write(*, "(/a)") "=========================="
  write(*, "(a,i4)") "Total number of errors = ", errors

  ! Shutdown SpLDLT
  call sylver_finalize()

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
    info%flag = SYLVER_SUCCESS ! Reset flag

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
    info%flag = SYLVER_SUCCESS ! Reset flag

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
    info%flag = SYLVER_SUCCESS ! Reset flag

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
    info%flag = SYLVER_SUCCESS ! Reset flag
    
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
    info%flag = SYLVER_SUCCESS ! Reset flag
    
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
    info%flag = SYLVER_SUCCESS ! Reset flag

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
    info%flag = SYLVER_SUCCESS ! Reset flag

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
    info%flag = SYLVER_SUCCESS ! Reset flag

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
    info%flag = SYLVER_SUCCESS ! Reset flag

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
    info%flag = SYLVER_SUCCESS ! Reset flag
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
   options%print_level = 2

   write(*,"(/a)") "======================"
   write(*,"(a)")  "Testing errors:"
   write(*,"(a)")  "======================"

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   ! Tests on call to spldlt_analyse (entry by columns)

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
   info%flag = SYLVER_SUCCESS ! Reset flag
   
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
   info%flag = SYLVER_SUCCESS ! Reset flag

   write(*,"(a)",advance="no") " * Testing order absent......................"
   options%ordering = 0
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, check=.true.)
   call print_result(info%flag, SYLVER_ERROR_ORDER)
   call spldlt_akeep_free(akeep)

   write(*,"(a)",advance="no") " * Testing order too short..................."
   if (allocated(order)) deallocate(order)
   allocate(order(a%n-1))
   order(1:a%n-1) = 1
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
   call print_result(info%flag, SYLVER_ERROR_ORDER)
   call spldlt_akeep_free(akeep)
   deallocate(order)

   call simple_mat(a)
   write(*,"(a)",advance="no") " * Testing order out of range above.........."
   allocate(order(a%n))
   order(1) = a%n+1
   do i = 2,a%n
      order(i) = i
   end do
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
   call print_result(info%flag, SYLVER_ERROR_ORDER)
   call spldlt_akeep_free(akeep)
   deallocate(order)

   call simple_mat(a)
   write(*,"(a)",advance="no") " * Testing order out of range below.........."
   allocate(order(a%n))
   order(1) = 0
   do i = 2,a%n
      order(i) = i
   end do
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
   call print_result(info%flag, SYLVER_ERROR_ORDER)
   call spldlt_akeep_free(akeep)

   write(*,"(a)",advance="no") " * Testing options%ordering out of range....."
   options%ordering = -1
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
   call print_result(info%flag, SYLVER_ERROR_ORDER)
   call spldlt_akeep_free(akeep)
   options%ordering = 0 ! Reset ordering

!!!!!!!!!!!!!!!!!!!!
   
   call simple_mat(a)
   write(*,"(a)",advance="no") " * Testing options%ordering oor.............."
   if (allocated(order)) deallocate(order)
   allocate(order(a%n))
   options%ordering = 3
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
   call print_result(info%flag, SYLVER_ERROR_ORDER)
   deallocate(order)
   call spldlt_akeep_free(akeep)
   options%ordering = 0

!!!!!!!!!!!!!!!!!!!!
   
   call simple_mat(a)
   write(*,"(a)",advance="no") " * Testing val absent........................"
   if (allocated(order)) deallocate(order)
   allocate(order(a%n))
   options%ordering = 2
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
   call print_result(info%flag, SYLVER_ERROR_VAL)
   deallocate(order)
   call spldlt_akeep_free(akeep)
   options%ordering = 0 ! Reset ordering
   info%flag = SYLVER_SUCCESS ! Reset error flag
   
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   
   ! Tests on call to spldlt_factorize

   posdef = .false.
   write(*,"(/a)") " * Testing errors from ssids_factor"

   write(*,"(a)",advance="no") " * Testing after analyse error..............."
   options%ordering = 2
   ! Trigger error in analyse as ordering = 2 and val is not present
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info, ptr=a%ptr, row=a%row)
   call print_result(info%flag, SYLVER_ERROR_CALL_SEQUENCE)
   call spldlt_akeep_free(akeep)
   options%ordering = 0 ! Reset ordering
   info%flag = SYLVER_SUCCESS ! Reset error flag

!!!!!!!!!!!!!!!!!!!!
   
   call simple_mat(a)
   write(*,"(a)",advance="no") " * Testing not calling analyse..............."
   posdef = .false.
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info, ptr=a%ptr, row=a%row)
   call print_result(info%flag, SYLVER_ERROR_CALL_SEQUENCE)
   call spldlt_akeep_free(akeep)
   info%flag = SYLVER_SUCCESS ! Reset error flag

!!!!!!!!!!!!!!!!!!!!
   
   call simple_mat(a)
   posdef = .false.
   write(*,"(a)",advance="no") " * Testing ptr absent........................"
   if (.not. allocated(order)) allocate(order(a%n))
   do i = 1,a%n
      order(i) = i
   end do
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.false.)
   if(info%flag .ne. SYLVER_SUCCESS) then
      write(*, "(a,i4)") &
           "Unexpected error during analyse. flag = ", info%flag
      errors = errors + 1
      call spldlt_akeep_free(akeep)
      return
   endif
   ! Hungarian scaling
   options%scaling = 1 
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info, row=a%row)
   call print_result(info%flag, SYLVER_ERROR_PTR_ROW)
   ! Auction scaling
   options%scaling = 2 
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info, row=a%row)
   call print_result(info%flag, SYLVER_ERROR_PTR_ROW)
   ! Norm-equilibration scaling
   options%scaling = 4 
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info, row=a%row)
   call print_result(info%flag, SYLVER_ERROR_PTR_ROW)
   call spldlt_akeep_free(akeep)
   options%scaling = 0 ! Reset scaling
   info%flag = SYLVER_SUCCESS ! Reset error flag

!!!!!!!!!!!!!!!!!!!!

   call simple_mat(a)
   posdef = .false.
   write(*,"(a)",advance="no") " * Testing row absent........................"
   if (.not. allocated(order)) allocate(order(a%n))
   do i = 1,a%n
      order(i) = i
   end do
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.false.)
   if(info%flag .ne. SYLVER_SUCCESS) then
      write(*, "(a,i4)") &
           "Unexpected error during analyse. flag = ", info%flag
      errors = errors + 1
      call spldlt_akeep_free(akeep)
      return
   endif
   ! Hungarian scaling
   options%scaling = 1 
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info, ptr=a%ptr)
   call print_result(info%flag, SYLVER_ERROR_PTR_ROW)
   ! Auction scaling
   options%scaling = 2 
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info, ptr=a%ptr)
   call print_result(info%flag, SYLVER_ERROR_PTR_ROW)
   ! Norm-equilibration scaling
   options%scaling = 4 
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info, ptr=a%ptr)
   call print_result(info%flag, SYLVER_ERROR_PTR_ROW)
   call spldlt_akeep_free(akeep)
   options%scaling = 0 ! Reset scaling
   info%flag = SYLVER_SUCCESS ! Reset error flag

!!!!!!!!!!!!!!!!!!!!

   write(*,"(a)",advance="no") " * Testing factor with singular matrix......."
   call simple_sing_mat(a)
   !call simple_mat(a)
   options%ordering = 0
   if (allocated(order)) deallocate(order)
   allocate(order(a%n))
   do i = 1,a%n
      order(i) = i
   end do

   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
   if(info%flag .lt. SYLVER_SUCCESS) then
      write(*, "(a,i4)") &
           "Unexpected error during analyse. flag = ", info%flag
      errors = errors + 1
      return
   endif

   options%action = .false.
   posdef = .false.
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info, ptr=a%ptr, row=a%row)
   call print_result(info%flag, SYLVER_ERROR_SINGULAR)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)
   info%flag = SYLVER_SUCCESS ! Reset error flag

! !!!!!!!!!!!!!!!!!!!!

   write(*,"(a)",advance="no") &
        " * Testing factor with singular matrix (MC64 scale)."
   call simple_sing_mat(a)
   options%ordering = 1
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
   if(info%flag < 0) then
      write(*, "(a,i4)") &
           "Unexpected error during analyse. flag = ", info%flag
      errors = errors + 1
      return
   endif
   print *, "rank = ", akeep%akeep%inform%matrix_rank 

   options%action = .false.
   options%scaling = 1 ! MC64
   posdef = .false.
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info, ptr=a%ptr, row=a%row)
   call print_result(info%flag, SYLVER_ERROR_SINGULAR)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)
   info%flag = SYLVER_SUCCESS ! Reset error flag
   
!!!!!!!!!!!!!!!!!!!!

   write(*,"(a)",advance="no") " * Testing factor psdef with indef..........."
   !call simple_mat(a)
   call simple_mat_indef(a)
   ! call simple_sing_mat(a)
   ! a%val(1) = -a%val(1)
   options%ordering = 1
   posdef = .true.
   if (allocated(order)) deallocate(order)
   allocate(order(a%n))
   do i = 1,a%n
      order(i) = i
   end do
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true.)
   if(info%flag < 0) then
      write(*, "(a,i4)") &
           "Unexpected error during analyse. flag = ", info%flag
      errors = errors + 1
      return
   endif
   print *, "n = ", a%n
   print *, "rank = ", akeep%akeep%inform%matrix_rank 
   options%scaling = 0
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info, ptr=a%ptr, row=a%row)
   call print_result(info%flag, SYLVER_ERROR_NOT_POS_DEF)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)
   info%flag = SYLVER_SUCCESS ! Reset error flag

!!!!!!!!!!!!!!!!!!!!

   write(*,"(a)",advance="no") " * Testing factor psdef with indef, large..."
   call gen_bordered_block_diag(.false., (/ 15, 455, 10 /), 20, a%n, a%ptr, &
        a%row, a%val, state)
   print *, "n = ", a%n
   if (allocated(order)) deallocate(order)
   allocate(order(a%n))
   do i = 1,a%n
      order(i) = i
   end do
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true., ncpu=ncpu)
   posdef = .true.
   !posdef = .false.
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info, ptr=a%ptr, row=a%row)
   call print_result(info%flag, SYLVER_ERROR_NOT_POS_DEF)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)
   info%flag = SYLVER_SUCCESS ! Reset error flag

!!!!!!!!!!!!!!!!!!!!

   write(*,"(a)",advance="no") " * Testing u oor............................."
   call simple_mat(a)
   options%ordering = 0
   if (allocated(order)) deallocate(order)
   allocate(order(a%n))
   do i = 1,a%n
      order(i) = i
   end do
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true., ncpu=ncpu)
   if(info%flag < 0) then
      write(*, "(a,i4)") &
           "Unexpected error during analyse. flag = ", info%flag
      errors = errors + 1
      return
   endif
   posdef = .false.
   options%u = -0.1
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info)
   options%u = 0.01 ! Reset value of u
   call print_result(info%flag, SYLVER_SUCCESS)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)

!!!!!!!!!!!!!!!!!!!!

   write(*,"(a)",advance="no") " * Testing options%scaling=3 no matching....."
   call simple_mat(a)
   options%ordering = 0
   if (allocated(order)) deallocate(order)
   allocate(order(a%n))
   do i = 1,a%n
      order(i) = i
   end do
   options%scaling = 0 ! No scaling
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true., ncpu=ncpu)
   if(info%flag < 0) then
      write(*, "(a,i4)") &
           "Unexpected error during analyse. flag = ", info%flag
      errors = errors + 1
      return
   endif
   posdef = .false.
   options%scaling = 3 ! MC64 from matching-based order
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info)
   options%scaling = 0
   call print_result(info%flag, SYLVER_ERROR_NO_SAVED_SCALING)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   
   ! tests on call to spldlt_solve

   write(*,"(/a)") " * Testing bad arguments spldlt_solve"

   write(*,"(a)",advance="no") " * Testing solve after factor error.........."
   call simple_sing_mat(a)
   options%ordering = 0
   if (allocated(order)) deallocate(order)
   allocate(order(a%n))
   do i = 1,a%n
      order(i) = i
   end do

   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true., ncpu=ncpu)

   options%action = .false.
   options%scaling = 0
   posdef = .false.
   ! Trigger error in spldlt_factorize
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info)
   if(info%flag .ne. SSIDS_ERROR_SINGULAR) then
      write(*, "(a,i4)") &
     "Unexpected flag returned by ssids_factor. flag = ", info%flag
      errors = errors + 1
      return
   endif

   if (allocated(x1)) deallocate(x1)
   allocate(x1(a%n))
   x1(1:a%n) = one
   call spldlt_solve(akeep, fkeep, 1, x1, a%n, options, info)
   call print_result(info%flag, SYLVER_ERROR_CALL_SEQUENCE)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)

!!!!!!!!!!!!!!!!!!!!
   
   call simple_mat(a)
   deallocate(x1)
   allocate(x1(a%n))
   x1(1:a%n) = one
   posdef = .false.
   write(*,"(a)",advance="no") " * Testing solve out of sequence............."
   if (allocated(order)) deallocate(order)
   allocate(order(a%n))
   do i = 1,a%n
     order(i) = i
   end do
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true., ncpu=ncpu)

   call spldlt_solve(akeep, fkeep, 1, x1, a%n, options, info)
   call print_result(info%flag, SYLVER_ERROR_CALL_SEQUENCE)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)

!!!!!!!!!!!!!!!!!!!!

   call simple_mat(a)
   posdef = .false.
   write(*,"(a)",advance="no") " * Testing job out of range below............"
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true., ncpu=ncpu)
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info)
   call spldlt_solve(akeep, fkeep, 1, x1, a%n, options, info, job=-1)
   call print_result(info%flag, SYLVER_ERROR_JOB_OOR)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)

!!!!!!!!!!!!!!!!!!!!

   call simple_mat(a)
   posdef = .false.
   write(*,"(a)",advance="no") " * Testing job out of range above............"
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true., ncpu=ncpu)
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info)
   call spldlt_solve(akeep, fkeep, 1, x1, a%n, options, info, job=5)
   call print_result(info%flag, SYLVER_ERROR_JOB_OOR)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)

!!!!!!!!!!!!!!!!!!!!

   call simple_mat(a)
   posdef = .false.
   write(*,"(a)",advance="no") " * Testing error in x (one rhs).............."
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true., ncpu=ncpu)
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info)
   if (allocated(x1)) deallocate(x1)
   allocate(x1(a%n-1))
   call spldlt_solve(akeep, fkeep, 1, x1, size(x1,1), options, info)
   call print_result(info%flag, SYLVER_ERROR_X_SIZE)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)

!!!!!!!!!!!!!!!!!!!!

   call simple_mat(a)
   posdef = .false.
   write(*,"(a)",advance="no") " * Testing error in nrhs....................."
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true., ncpu=ncpu)
   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info)
   nrhs = -2
   if (allocated(x)) deallocate(x)
   allocate(x(a%n,1))
   call spldlt_solve(akeep, fkeep, nrhs, x1, a%n, options, info)
   call print_result(info%flag, SYLVER_ERROR_X_SIZE)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)

 end subroutine test_errors

 subroutine test_special
   type(matrix_type) :: a
   type(sylver_options) :: options
   type(spldlt_akeep_type) :: akeep
   type(spldlt_fkeep_type) :: fkeep
   type(sylver_inform) :: info

   integer :: i
   logical :: check
   logical :: posdef
   integer :: st, cuda_error
   integer :: test
   integer, dimension(:), allocatable :: order
   real(wp), dimension(:), allocatable :: scale
   real(wp), dimension(:), allocatable :: x1
   real(wp), dimension(:,:), allocatable :: rhs, x, res
   type(random_state) :: state

   integer :: big_test_n = int(1e5 + 5)

   options%unit_error = we_unit
   options%unit_warning = we_unit
   options%print_level = 2

   write(*,"(a)")
   write(*,"(a)") "====================="
   write(*,"(a)") "Testing special cases"
   write(*,"(a)") "====================="

   ! do test = 1,2
   !    if (test == 1) then
   write(*,"(a)",advance="no") &
        " * Testing n = 0 (CSC)..................."
   a%n = 0
   a%ne = 0
   if (allocated(a%ptr)) deallocate(a%ptr, stat=st)
   if (allocated(a%row)) deallocate(a%row,stat=st)
   if (allocated(a%val)) deallocate(a%val,stat=st)
   
   allocate(a%ptr(a%n+1),a%row(a%ne),a%val(a%ne))

   if (allocated(order)) deallocate(order)
   allocate(order(a%n))
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true., ncpu=ncpu)
   if(info%flag.ne.0) then
      write(*, "(a,i4)") &
           "Unexpected error during analyse. flag = ", info%flag
      errors = errors + 1
      call spldlt_akeep_free(akeep)
      return
   endif
   ! else
   !    write(*,"(a)",advance="no") &
   !         " * Testing n = 0 (coord)................."
   !    a%n = 0
   !    a%ne = 0
   !    deallocate(a%row,a%val)
   !    allocate(a%col(a%ne),a%row(a%ne),a%val(a%ne))

   !    if (allocated(order)) deallocate(order)
   !    allocate(order(a%n))
   !    call ssids_analyse_coord(a%n, a%ne, a%row, a%col, akeep, options, &
   !         info, order=order)
   !    if(info%flag.ne.0) then
   !       write(*, "(a,i4)") &
   !            "Unexpected error during analyse_coord. flag = ", info%flag
   !       errors = errors + 1
   !       call ssids_free(akeep, cuda_error)
   !       exit
   !    endif
   ! endif

   deallocate(scale,stat=st)
   allocate(scale(a%n))
   options%scaling = 1 ! MC64

   posdef = .true.

   call spldlt_factorize(akeep, fkeep, posdef, a%val, options, &
        info, scale=scale)
   if(info%flag.ne.0) then
      write(*, "(a,i4)") &
           "Unexpected error during factor. flag = ", info%flag
      errors = errors + 1
      call spldlt_akeep_free(akeep)
      call spldlt_fkeep_free(fkeep)
      return
   endif

   if (allocated(x1)) deallocate(x1)
   allocate(x1(a%n))
   call spldlt_solve(akeep, fkeep, 1, x1, a%n, options, info)
   if(info%flag.ne.0) then
      write(*, "(a,i4)") &
           "Unexpected error during solve. flag = ", info%flag
      errors = errors + 1
      call spldlt_akeep_free(akeep)
      call spldlt_fkeep_free(fkeep)
      return
   endif

   call print_result(info%flag, SYLVER_SUCCESS)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)
   ! enddo

   if (allocated(order)) deallocate(order)
   if (allocated(a%ptr)) deallocate(a%ptr)
   if (allocated(a%row)) deallocate(a%row)
   if (allocated(a%val)) deallocate(a%val)

   ! A matrix with entry in column 1 only, explicit zeroes on diagonal
   allocate(a%ptr(big_test_n+1), a%row(4*big_test_n), a%val(4*big_test_n))
   allocate(order(10))
   write(*,"(a)",advance="no") &
        " * Testing zero pivot code .............."
   a%n = 10
   a%ptr(1) = 1
   a%ptr(2) = a%n + 1
   do i = 1, a%n
      a%row(i) = i
      a%val(i) = i
      order(i) = i
   end do
   do i = 2, a%n
      a%ptr(i+1) = a%ptr(i) + 1
      a%row(a%ptr(i)) = i
      a%val(a%ptr(i)) = 0.0_wp
   end do
   options%ordering = 0
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true., ncpu=ncpu)
   call print_result(info%flag, SYLVER_SUCCESS)
   call spldlt_factorize(akeep, fkeep, .false., a%val, options, info)
   call print_result(info%flag, SYLVER_WARNING_FACT_SINGULAR)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)

   ! Trigger block factor-solve code with zeroes
   ! (   0.0         )
   ! ( 1e-21 1.0     )
   ! (       2.0 3.0 )
   write(*,"(a)",advance="no") &
        " * Testing zero pivot code (block)......."
   a%n = 3
   a%ptr(1:4) = (/ 1, 3, 5, 6 /)
   a%row(1:5) = (/ 1, 2, 2, 3, 3 /)
   a%val(1:5) = (/ 0d0, 1d-2*options%small, 1d0, 2d0, 3d0 /)
   order(1:3) = (/ 1, 2, 3 /)
   options%ordering = 0
   options%nemin = 16
   options%scaling = 0
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true., ncpu=ncpu)
   call print_result(info%flag,SSIDS_SUCCESS)
   call spldlt_factorize(akeep, fkeep, .false., a%val, options, info)
   call print_result(info%flag, SYLVER_WARNING_FACT_SINGULAR)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)

   ! Trigger single column factor-solve code with zeroes
   ! (   0.0         )
   ! ( 1e-21 1.0     )
   ! (       2.0 3.0 )
   write(*,"(a)",advance="no") &
        " * Testing zero pivot code (column)......"
   a%n = 3
   a%ptr(1:4) = (/ 1, 3, 5, 6 /)
   a%row(1:5) = (/ 1, 2, 2, 3, 3 /)
   a%val(1:5) = (/ 0d0, 1d-2*options%small, 1d0, 2d0, 3d0 /)
   order(1:3) = (/ 1, 2, 3 /)
   options%ordering = 0
   options%nemin = 1
   options%scaling = 0
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, order, check=.true., ncpu=ncpu)
   call print_result(info%flag,SSIDS_SUCCESS)
   call spldlt_factorize(akeep, fkeep, .false., a%val, options, info)
   call print_result(info%flag, SYLVER_WARNING_FACT_SINGULAR)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)

   options = default_options
   write(*,"(a)",advance="no") &
        " * Testing n>1e5, ne<3.0*n, order=1......"
   a%n = big_test_n
   call gen_random_indef(a, 2_long*big_test_n, state)
   options%ordering = 1
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, check=.true., ncpu=ncpu)
   call print_result(info%flag,SYLVER_SUCCESS)
   call spldlt_akeep_free(akeep)

   write(*,"(a)",advance="no") &
        " * Testing n>1e5, ne>3.0*n, order=1......"
   a%n = big_test_n
   call gen_random_indef(a, 4_long*big_test_n, state)
   options%ordering = 1
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, check=.true., ncpu=ncpu)
   call print_result(info%flag,SYLVER_SUCCESS)
   call spldlt_akeep_free(akeep)

   write(*,"(a)",advance="no") &
        " * Testing n>1e5, ne>3.0*n, order=2......"
   a%n = big_test_n
   call gen_random_indef(a, 4_long*big_test_n, state)
   options%ordering = 2
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, check=.true., ncpu=ncpu, val=a%val)
   call print_result(info%flag, SYLVER_SUCCESS)
   call spldlt_akeep_free(akeep)

   ! (     1.0 )
   ! ( 1.0     )
   write(*,"(a)",advance="no") &
        " * Testing n<1e5,oxo,m1<1.8*m2,order=1..."
   a%n = 2
   a%ptr(1:3) = (/ 1, 2, 2 /)
   a%row(1:1) = (/ 2 /)
   a%val(1:1) = (/ 1.0 /)
   options%ordering = 1
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, check=.true., ncpu=ncpu)
   call print_result(info%flag, SYLVER_WARNING_MISSING_DIAGONAL)
   call gen_rhs(a, rhs, x1, x, res, 1)
   call chk_answer(.false., a, akeep, options, rhs, x, &
        res, SYLVER_WARNING_MISSING_DIAGONAL, fs=.true.)
   ! call spldlt_factorize(akeep, fkeep, .false., a%val, options, info)
   ! call print_result(info%flag, SYLVER_WARNING_MISSING_DIAGONAL)
   call spldlt_akeep_free(akeep)
   call spldlt_fkeep_free(fkeep)

   ! (  x   x  1.0 )
   ! (  x   x  2.0 )
   ! ( 1.0 2.0  x  )
   write(*,"(a)",advance="no") &
        " * Testing n<1e5,oxo,m1>1.8*m2,order=1..."
   a%n = 3
   a%ptr(1:4) = (/ 1, 2, 3, 3 /)
   a%row(1:2) = (/ 3, 3 /)
   a%val(1:2) = (/ 1.0, 2.0 /)
   options%ordering = 1
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, check=.true., ncpu=ncpu)
   call print_result(info%flag, SYLVER_WARNING_MISSING_DIAGONAL)
   call gen_rhs(a, rhs, x1, x, res, 1)
   call chk_answer(.false., a, akeep, options, rhs, x, &
        res, SYLVER_WARNING_FACT_SINGULAR, fs=.true.)
   call spldlt_akeep_free(akeep)

   ! (  x   x  1.0 )
   ! (  x   x  2.0 )
   ! ( 1.0 2.0  x  )
   write(*,"(a)",advance="no") &
        " * Testing n<1e5,oxo,m1>1.8*m2,order=2..."
   a%n = 3
   a%ptr(1:4) = (/ 1, 2, 3, 3 /)
   a%row(1:2) = (/ 3, 3 /)
   a%val(1:2) = (/ 1.0, 2.0 /)
   options%ordering = 2
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, check=.true., ncpu=ncpu, val=a%val)
   call print_result(info%flag, SYLVER_WARNING_ANAL_SINGULAR)
   call gen_rhs(a, rhs, x1, x, res, 1)
   call chk_answer(.false., a, akeep, options, rhs, x, &
        res, SYLVER_WARNING_FACT_SINGULAR, fs=.true.)
   call spldlt_akeep_free(akeep)

   ! Test posdef with node at least 384 and multiple nodes [for coverage]
   write(*,"(a)",advance="no") &
        " * Testing n=500, posdef, BBD............"
   options = default_options
   call gen_bordered_block_diag(.true., (/ 15, 455, 10 /), 20, a%n, a%ptr, &
        a%row, a%val, state)
   call spldlt_analyse(akeep, a%n, a%ptr, a%row, options, info, check=.true., ncpu=ncpu, val=a%val)
   call print_result(info%flag, SYLVER_SUCCESS)
   call gen_rhs(a, rhs, x1, x, res, 1)
   call chk_answer(.true., a, akeep, options, rhs, x, res, SYLVER_SUCCESS)
   call spldlt_akeep_free(akeep)

 end subroutine test_special

 subroutine test_random
   type(spldlt_akeep_type)   :: akeep
   type(spldlt_fkeep_type)   :: fkeep
   type(sylver_options) :: options
   type(sylver_inform)  :: info

   !logical, parameter :: debug = .true.
   logical, parameter :: debug = .false.

   integer :: maxn = 1000
   integer :: maxnemin = 48
   integer :: maxnz =  1000000
   integer, parameter :: maxnrhs = 10
   integer, parameter :: nprob = 100
   type(random_state) :: state

   type(matrix_type) :: a
   real(wp), allocatable, dimension(:, :) :: rhs,x
   real(wp), allocatable, dimension(:) :: rhs1d,x1
   real(wp), allocatable, dimension(:, :) :: res

   logical :: posdef
   integer :: prblm, i, j, k, n1, nrhs, mt
   integer(long) :: ne, nza
   integer, dimension(:), allocatable :: order, piv_order
   integer, dimension(:), allocatable :: xindex, bindex
   logical, dimension(:), allocatable :: lflag
   real(wp), dimension(:,:), allocatable :: d
   real(wp), dimension(:), allocatable :: d1
   integer :: cuda_error
   logical :: check, coord
   real(wp) :: num_flops

   write(*, "(a)")
   write(*, "(a)") "======================="
   write(*, "(a)") "Testing random matrices"
   write(*, "(a)") "======================="

   allocate(a%ptr(maxn+1))
   allocate(a%row(2*maxnz), a%val(2*maxnz), a%col(2*maxnz))
   allocate(order(maxn))
   allocate(piv_order(maxn))
   allocate(rhs(maxn,maxnrhs), res(maxn,maxnrhs), x(maxn,maxnrhs))
   allocate(x1(maxn), rhs1d(maxn))
   allocate(d(2,maxn), d1(maxn))
   allocate(bindex(maxn), xindex(maxn), lflag(maxn))

   options%multiplier = 1.0 ! Ensure we give reallocation a work out

   if(debug) options%print_level = 10000
   options%min_gpu_work = 0 ! alway allow some gpu working

 end subroutine test_random

end program main
