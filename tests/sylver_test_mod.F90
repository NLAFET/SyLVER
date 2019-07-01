module sylver_test_mod
  use spral_ssids_datatypes
  use spral_random
  implicit none

  integer :: errors

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

end module sylver_test_mod
