module sylver_test_mod
  use spral_ssids_datatypes
  use spral_matrix_util, only : SPRAL_MATRIX_REAL_SYM_PSDEF, &
       SPRAL_MATRIX_REAL_SYM_INDEF, print_matrix
  use spral_random
  use spral_random_matrix, only : random_matrix_generate
  use sylver_mod
  implicit none

  integer :: errors

  !   real(wp), parameter :: err_tol = 5e-12
  !   real(wp), parameter :: err_tol_scale = 2e-10
  real(wp), parameter :: err_tol = 5e-11
  real(wp), parameter :: err_tol_scale = 1e-08

  type(sylver_options) :: default_options

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
    !print *, 'residual =', maxval(abs(res))
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
  
  subroutine simple_mat_indef(a,extra)
    ! simple indef test matrix (lower triangular part)
    type(matrix_type), intent(inout) :: a
    integer, optional, intent(in) :: extra

    integer :: myextra,st

    myextra = 0
    if(present(extra)) myextra = extra

    !
    ! Create the simple sparse matrix (lower and upper triangles):
    !
    !  1.0  2.0       3.0
    !  2.0  1.0
    !           1.0  4.0
    !  3.0      4.0  1.0
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
    a%val(1:7) = (/  1.0,  2.0, 3.0, &
         1.0,      &
         1.0,  4.0,      &
         1.0 /)

  end subroutine simple_mat_indef

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

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! Generates a bordered block diagonal form
  ! blocks have size and number given in dimn
  ! border is width of border - however final var is only included in final block
  ! to discourage merging with largest block during analysis
  subroutine gen_bordered_block_diag(posdef, dimn, border, n, ptr, row, val, state)
    logical, intent(in) :: posdef
    integer, dimension(:), intent(in) :: dimn
    integer, intent(in) :: border
    integer, intent(out) :: n
    integer(long), dimension(:), allocatable :: ptr
    integer, dimension(:), allocatable :: row
    real(wp), dimension(:), allocatable :: val
    type(random_state), intent(inout) :: state

    integer :: i, j, k, blk, blk_sa
    integer :: nnz
    integer :: st

    ! Clear any previous allocs
    if (allocated(ptr)) deallocate(ptr, stat=st)
    if (allocated(row)) deallocate(row, stat=st)
    if (allocated(val)) deallocate(val, stat=st)

    ! allocate arrays
    n = sum(dimn(:)) + border
    nnz = 0
    do blk = 1, size(dimn)
       j = dimn(blk)
       nnz = nnz + j*(j+1)/2 + j*border
    end do
    nnz = nnz + border*(border+1)/2
    allocate(ptr(n+1), row(nnz), val(nnz))

    ! Generate val = unif(-1,1)
    do i = 1, nnz
       val(i) = random_real(state)
    end do

    ! Generate ptr and row; make posdef if required
    j = 1
    blk_sa = 1
    do blk = 1, size(dimn)
       do i = blk_sa, blk_sa+dimn(blk)-1
          ptr(i) = j
          if(posdef) val(j) = abs(val(j)) + n ! make diagonally dominant
          do k = i, blk_sa+dimn(blk)-1
             row(j) = k
             j = j + 1
          end do
          do k = n-border+1, n-1
             row(j) = k
             j = j + 1
          end do
       end do
       blk_sa = blk_sa + dimn(blk)
    end do
    do i = n-border+1, n
       ptr(i) = j
       if(posdef) val(j) = abs(val(j)) + n ! make diagonally dominant
       do k = i, n
          row(j) = k
          j = j + 1
       end do
    end do
    ptr(n+1) = j
  end subroutine gen_bordered_block_diag

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine gen_random_indef(a, nza, state, zr)
    implicit none

    type(matrix_type), intent(inout) :: a
    integer(long), intent(in) :: nza
    type(random_state), intent(inout) :: state
    integer, optional, intent(in) :: zr ! if present, all entries in
    ! row zr are zero
    integer, dimension(:), allocatable :: ptr32
    integer :: ptrsz
    
    integer :: i, k, l, flag

    ptrsz = size(a%ptr, 1)
    allocate(ptr32(ptrsz))
    ! Generate a FIXME: move to 64-bit
    call random_matrix_generate(state, SPRAL_MATRIX_REAL_SYM_INDEF, a%n, a%n, &
         int(nza), ptr32, a%row, flag, val=a%val, nonsingular=.true., sort=.true.)
    if(flag.ne.0) print *, "Bad flag from random_matrix_generate()"
    a%ptr = ptr32
    deallocate(ptr32)
    
    if (present(zr)) then
       ! Scan along row
       do i = a%ptr(1), a%ptr(zr)-1
          if(a%row(i).eq.zr) a%val(i) = zero
       end do
       ! Scan along column
       do i = a%ptr(zr),a%ptr(zr+1)-1
          a%val(i) = zero
       end do
    elseif(a%n.gt.3) then
       ! Put some zeros on diagonal, observing first entry in column
       ! is always the diagonal after sorting
       ! but don't have all zeros in the col.
       l = random_integer(state,  a%n/2)
       do k = 1, a%n, max(1,l)
          if (a%ptr(k+1) > a%ptr(k) + 1) then
             i = a%ptr(k)
             a%val(i) = zero
          endif
       end do
       ! also make sure we have some large off diagonals
       do k = 1, a%n
          i = a%ptr(k+1) - 1
          a%val(i) = a%val(i)*1000
       end do
    endif

  end subroutine gen_random_indef

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine gen_random_posdef(a, nza, state)
    implicit none
    
    type(matrix_type), intent(inout) :: a
    integer(long), intent(in) :: nza
    type(random_state), intent(inout) :: state
    integer, dimension(:), allocatable :: ptr32
    integer :: ptrsz

    integer :: i, j, k, flag
    real(wp) :: tempv

    ptrsz = size(a%ptr, 1)
    allocate(ptr32(ptrsz))
    ! Generate matrix FIXME: move to 64-bit
    call random_matrix_generate(state, SPRAL_MATRIX_REAL_SYM_PSDEF, a%n, a%n, &
         int(nza), ptr32, a%row, flag, val=a%val, nonsingular=.true., sort=.true.)
    if(flag.ne.0) print *, "Bad flag from random_matrix_generate()"
    a%ptr = ptr32
    deallocate(ptr32)

    ! Make a diagonally dominant, observing first entry in column
    ! is always the diagonal after sorting
    do k = 1, a%n
       tempv = zero
       do j = a%ptr(k)+1, a%ptr(k+1)-1
          tempv = tempv + abs(a%val(j))
          i = a%ptr(a%row(j))
          a%val(i) = a%val(i) + abs(a%val(j))
       end do
       i = a%ptr(k)
       a%val(i) = one + a%val(i) + tempv
    end do
  end subroutine gen_random_posdef

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine simple_metis_order(a, order)
    use spral_metis_wrapper, only : metis_order
    implicit none

    type(matrix_type), intent(in) :: a
    integer, dimension(:), allocatable :: order

    integer :: i, flag, stat
    integer, dimension(:), allocatable :: invp

    allocate(invp(a%n))

    ! Perform MeTiS
    ! FIXME: which way around should we have order and invp?
    call metis_order(a%n, a%ptr, a%row, order, invp, flag, stat)
    if(flag.ne.0) then
       ! Failed for some reason
       do i = 1, a%n
          order(i) = i
       end do
       return
    endif

  end subroutine simple_metis_order

end module sylver_test_mod
