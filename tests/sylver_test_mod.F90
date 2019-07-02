module sylver_test_mod
  use spral_ssids_datatypes
  use spral_random
  use sylver_mod
  implicit none

  integer :: errors

  !   real(wp), parameter :: err_tol = 5e-12
  !   real(wp), parameter :: err_tol_scale = 2e-10
  real(wp), parameter :: err_tol = 5e-11
  real(wp), parameter :: err_tol_scale = 1e-08

  type :: matrix_type
     integer :: n
     integer(long) :: ne
     integer(long), dimension(:), allocatable :: ptr
     integer, dimension(:), allocatable :: row, col
     real(wp), dimension(:), allocatable :: val
  end type matrix_type

contains

  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine print_result(actual, expected, continued)
    integer :: actual
    integer :: expected
    logical, optional :: continued

    logical :: mycontinued

    mycontinued = .false.
    if(present(continued)) mycontinued = continued

    if(actual.eq.expected) then
       if(mycontinued) then
          write(*,"(a)", advance="no") "ok..."
       else
          write(*,"(a)") "ok"
       endif
       return
    endif

    write(*,"(a)") "fail"
    write(*,"(2(a,i4))") "returned ", actual, ", expected ", expected
    errors = errors + 1
  end subroutine print_result

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  ! generate rhs and copy into x
  subroutine gen_rhs(a, rhs, x1, x, res, nrhs, state)
    type(matrix_type), intent(inout) :: a
    real(wp), dimension(:,:), allocatable, intent(inout) :: rhs
    real(wp), dimension(:,:), allocatable, intent(inout) :: x
    real(wp), dimension(:), allocatable, intent(inout) :: x1
    real(wp), dimension(:,:), allocatable, intent(inout) :: res
    integer, intent(in) :: nrhs
    type (random_state), optional :: state

    integer :: i, j, k, n
    real(wp) :: atemp, random

    n = a%n
    if (allocated(rhs)) deallocate(rhs)
    allocate(rhs(n, nrhs))

    if (allocated(x)) deallocate(x)
    allocate(x(n, nrhs))

    if (allocated(res)) deallocate(res)
    allocate(res(n, nrhs))

    if (allocated(x1)) deallocate(x1)
    allocate(x1(n))

    rhs = zero
    if (.not.present(state)) then
       ! Generate rhs assuming x = 1
       do k = 1, n
          do j = a%ptr(k), a%ptr(k+1)-1
             i = a%row(j)
             if(i < k .or. i > n) cycle
             rhs(i, 1:nrhs) = rhs(i, 1:nrhs) + a%val(j)
             if(i.eq.k) cycle
             rhs(k, 1:nrhs) = rhs(k, 1:nrhs) + a%val(j)
          end do
       end do
    else
       do i = 1,n
          random = random_real(state,.true.)
          x(i,1:nrhs) = random
       end do
       do k = 1, n
          do j = a%ptr(k), a%ptr(k+1)-1
             i = a%row(j)
             if(i < k .or. i > n) cycle
             atemp = a%val(j)
             rhs(i, 1:nrhs) = rhs(i, 1:nrhs) + atemp*x(k,1:nrhs)
             if(i.eq.k) cycle
             rhs(k, 1:nrhs) = rhs(k, 1:nrhs) + atemp*x(i,1:nrhs)
          end do
       end do
    endif

    x(1:a%n,1:nrhs) = rhs(1:a%n,1:nrhs)
    x1(1:a%n) = rhs(1:a%n,1)

  end subroutine gen_rhs

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  subroutine simple_mat(a,extra)
    ! simple pos def test matrix (lower triangular part)
    type(matrix_type), intent(inout) :: a
    integer, optional, intent(in) :: extra

    integer :: myextra,st

    myextra = 0
    if(present(extra)) myextra = extra

    !
    ! Create the simple sparse matrix (lower and upper triangles):
    !
    ! 10.0  2.0       3.0
    !  2.0 10.0
    !           10.0  4.0
    !  3.0       4.0 10.0
    !

    a%n = 4
    a%ne = 7
    deallocate(a%ptr, a%row, a%col, a%val, stat=st)
    allocate(a%ptr(a%n+1))
    a%ptr = (/ 1, 4, 5, 7, 8 /)
    allocate(a%row(a%ne+myextra))
    allocate(a%col(a%ne+myextra))
    allocate(a%val(a%ne+myextra))

    a%row(1:7) = (/ 1,2,4,     2,    3, 4,  4 /)
    a%col(1:7) = (/ 1,1,1,     2,    3, 3,  4 /)
    a%val(1:7) = (/  10.0,  2.0, 3.0, &
         10.0,      &
         10.0,  4.0,      &
         10.0 /)

  end subroutine simple_mat

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  subroutine simple_mat_lower(a,extra)
    ! simple pos def test matrix (lower triangular part only)
    type(matrix_type), intent(inout) :: a
    integer, optional, intent(in) :: extra

    integer :: myextra,st

    myextra = 0
    if(present(extra)) myextra = extra

    !
    ! Create the simple sparse matrix (lower triangular part only):
    !
    ! 10.0  2.0       3.0
    !  2.0 10.0
    !           10.0  4.0
    !  3.0       4.0 10.0
    !

    a%n = 4
    a%ne = 7
    deallocate(a%ptr, a%row, a%col, a%val, stat=st)
    allocate(a%ptr(a%n+1))
    a%ptr = (/ 1, 4, 5, 7, 8 /)
    allocate(a%col(a%ne+myextra))
    allocate(a%row(a%ne+myextra))
    allocate(a%val(a%ne+myextra))

    a%col(1:7) = (/ 1, 1, 1,     2,    3, 3,    4 /)
    a%row(1:7) = (/ 1, 2, 4,     2,    3, 4,    4 /)
    a%val(1:7) = (/   10.0, 2.0, 3.0, &
         10.0, &
         10.0, 4.0, &
         10.0 /)

  end subroutine simple_mat_lower

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  subroutine compute_resid(nrhs,a,x,lx,rhs,lrhs,res,lres)
    integer, intent(in) :: nrhs, lrhs, lx, lres
    type(matrix_type), intent(in) :: a
    real(wp), intent(in) :: rhs(lrhs,nrhs)
    real(wp), intent(in) :: x(lx,nrhs)
    real(wp), intent(out) :: res(lres,nrhs)

    real(wp), dimension(:), allocatable :: work

    integer :: i, j, k
    real(wp) :: anorm, atemp, bnorm(1:nrhs), xnorm(1:nrhs)

    allocate(work(a%n))

    anorm = 0
    bnorm = 0
    xnorm = 0
    work = 0

    ! Check residual
    res(1:a%n,1:nrhs) = rhs(1:a%n,1:nrhs)
    do k = 1, a%n
       do j = a%ptr(k), a%ptr(k+1)-1
          i = a%row(j)
          if (i < 1 .or. i > a%n) cycle
          atemp = a%val(j)
          res(i, 1:nrhs) = res(i, 1:nrhs) - atemp*x(k,1:nrhs)
          work(i) = work(i) + abs(atemp)
          if(i.eq.k) cycle
          res(k, 1:nrhs) = res(k, 1:nrhs) - atemp*x(i,1:nrhs)
          work(k) = work(k) + abs(atemp)
       end do
    end do

    do k = 1, a%n
       anorm = max(anorm,work(k))
       do i = 1,nrhs
          bnorm(i) = max(bnorm(i),abs(rhs(k,i)))
          xnorm(i) = max(xnorm(i),abs(x(k,i)))
       end do
    end do

    do k = 1,a%n
       do i = 1,nrhs
          res(k,i) = res(k,i)/(anorm*xnorm(i) + bnorm(i))
       end do
    end do

  end subroutine compute_resid

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  subroutine chk_answer(posdef, a, spldlt_akeep, options, rhs, x, res, &
       expected_flag, fs)
    logical, intent(in) :: posdef
    type(matrix_type), intent(inout) :: a
    type(spldlt_akeep_type), target, intent(inout) :: spldlt_akeep
    type(sylver_options), intent(in) :: options
    real(wp), dimension(:,:), intent(inout) :: rhs
    real(wp), dimension(:,:), intent(inout) :: x
    real(wp), dimension(:,:), intent(inout) :: res
    integer, intent(in) :: expected_flag
    logical, optional, intent(in) :: fs

    type(spldlt_fkeep_type) :: spldlt_fkeep
    type(sylver_inform) :: info
    integer :: nrhs, cuda_error
    type(sylver_options) :: myoptions
    type(ssids_akeep), pointer :: akeep ! SSIDS akeep structure

    akeep => spldlt_akeep%akeep
    
    myoptions = options
    myoptions%unit_warning = -1 ! disable printing warnings

    write(*,"(a)",advance="no") " *    checking answer...................."

    nrhs = 1
    ! if(present(fs) .and. .false.) then
    !    call ssids_factor_solve(posdef,a%val,nrhs,x,a%n,akeep,fkeep,myoptions, &
    !         info)
    !    if(info%flag .ne. expected_flag) then
    !       write(*, "(a,2i4)") "fail on factor_solve",info%flag,expected_flag
    !       errors = errors + 1
    !       go to 99
    !    endif
    ! else
    call spldlt_factorize(spldlt_akeep, spldlt_fkeep, posdef, a%val, myoptions, &
         info, ptr=a%ptr, row=a%row)
    if(info%flag .ne. expected_flag) then
       write(*, "(a,2i4)") "fail on factor",info%flag,expected_flag
       errors = errors + 1
       go to 99
    endif

    call spldlt_solve(spldlt_akeep, spldlt_fkeep, nrhs, x, a%n, myoptions, info)
    if(info%flag .ne. expected_flag) then
       write(*, "(a,2i4)") "fail on solve", info%flag,expected_flag
       errors = errors + 1
       go to 99
    endif
    ! endif

    ! Check residual
    call compute_resid(nrhs,a,x,a%n,rhs,a%n,res,a%n)
    ! print *, 'residual =', maxval(abs(res))
    if(maxval(abs(res(1:a%n,1:nrhs))) < err_tol) then
       write(*, "(a)") "ok"
    else
       write(*, "(a,es12.4)") "fail residual = ", &
            maxval(abs(res(1:a%n,1:nrhs)))
       errors = errors + 1
    endif

    ! remember: must call finalise
99  call spldlt_fkeep_free(spldlt_fkeep)

  end subroutine chk_answer

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  subroutine simple_sing_mat(a)
    type(matrix_type), intent(inout) :: a
    integer :: st

    !
    ! Create the simple singular sparse matrix:
    !
    !  0.0  2.0 
    !  2.0  0.0  1.0
    !       1.0  0.0
    !
    ! we will not enter diagonal entries explicitly

    a%n = 3
    a%ne = 2
    deallocate(a%ptr, a%row, a%val, stat=st)
    allocate(a%ptr(a%n+1))
    a%ptr = (/ 1, 2, 3, 3 /)
    allocate(a%row(9))
    allocate(a%val(9))
    a%row(1:2) = (/ 2, 3 /)
    a%val(1:2) = (/   2.0, 1.0 /)

  end subroutine simple_sing_mat

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine simple_sing_mat2(a)
    type(matrix_type), intent(inout) :: a
    integer :: st

    !
    ! Create the simple singular sparse matrix:
    !
    ! -1.0  2.0  0.0
    !  2.0  1.0  0.0
    !  0.0  0.0  0.0
    !
    ! by entering null column.

    a%n = 3
    a%ne = 3
    deallocate(a%ptr, a%row, a%val, stat=st)
    allocate(a%ptr(a%n+1))
    a%ptr = (/ 1, 3, 4, 4 /)
    allocate(a%row(9))
    allocate(a%val(9))
    a%row(1:3) = (/ 1, 2, 2 /)
    a%val(1:3) = (/  -1.0, 2.0, 1.0 /)

  end subroutine simple_sing_mat2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  subroutine simple_mat_zero_diag(a)
    type(matrix_type), intent(inout) :: a

    integer :: st

    !
    ! Create the simple sparse matrix:
    !
    !  0.0  1.0  2.0
    !  1.0  0.0  1.0
    !  2.0  1.0  0.0

    a%n = 3
    a%ne = 3
    deallocate(a%ptr, a%row, a%val, stat=st)
    allocate(a%ptr(a%n+1))
    a%ptr = (/ 1, 3, 4, 4 /)
    allocate(a%row((a%ptr(a%n+1)-1+a%n)))
    allocate(a%val((a%ptr(a%n+1)-1+a%n)))

    a%row(1:3) = (/ 2,3,  3 /)
    a%val(1:3) = (/   1.0, 2.0, &
         1.0 /)

  end subroutine simple_mat_zero_diag

end module sylver_test_mod
