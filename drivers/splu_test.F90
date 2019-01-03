program splu_test
  use, intrinsic :: iso_c_binding
  use spral_ssids_datatypes
  use spldlt_datatypes_mod, only: spldlt_options, splu_options 
  use spral_rutherford_boeing
  use spral_matrix_util
  use spral_ssids_inform, only : ssids_inform
  use spldlt_analyse_mod
  use splu_factorize_mod
  implicit none

  ! indexes
  integer :: r, i, j, k

  ! timings
  integer :: start_t, stop_t, rate_t

  ! matrix reader options
  type(rb_read_options) :: rb_options
  integer :: rb_flag
  integer :: flag, more
  
  integer :: ncpu ! Number of CPUs
  integer :: ngpu ! Number of GPUs

  ! right-hand side and solution
  integer :: nrhs
  double precision, dimension(:,:), allocatable :: rhs, soln 
  ! double precision, dimension(:), allocatable :: scaling
  double precision, dimension(:), allocatable :: res

  ! Matrix description
  character(len=200) :: matfile = ''
  integer :: m, n
  integer(long), dimension(:), allocatable :: ptr
  integer, dimension(:), allocatable :: row
  real(wp), dimension(:), allocatable :: val

  ! SSIDS structures
  type(ssids_inform) :: inform ! stats
  type(ssids_options), pointer :: ssids_opt ! SSIDS options

  ! SyLVER structures
  type(spldlt_options), target :: options ! SyLVER options
  type(splu_options) :: splu_opt ! SpLU options

  type(spldlt_akeep_type) :: sylver_akeep ! Analyse phase data
  type(splu_fkeep_type) :: splu_fkeep

  call proc_args(options, nrhs, ncpu, ngpu, matfile)

  ssids_opt => options%super ! Point to SSIDS options

  ! Set SSIDS options
  ! ssids_opt%print_level = 1 ! enable printing
  ssids_opt%print_level = 0 ! disable printing
  ssids_opt%use_gpu = .false. ! disable GPU

  if (matfile.eq.'') matfile = "matrix.rb" 

  print *, "[SpLU Test] ncpu = ", ncpu
  print *, "[SpLU Test] nb = ", ssids_opt%cpu_block_size
  print *, "[SpLU Test] nrhs = ", nrhs

  ! Read in a matrix
  write(*, "(a)") "Reading..."
  ! DEBUG ensure matrix is diag dominant
  ! rb_options%values = -3 ! Force diagonal dominance
  rb_options%values = 4 ! unsymmetric make up values if necessary (if only pattern is given)
  ! rb_options%values = 0 ! as per file
  rb_options%lwr_upr_full = 3 ! Value in both lower and upper triangles
  call rb_read(matfile, m, n, ptr, row, val, rb_options, rb_flag)
  if(rb_flag.ne.0) then
     print *, "Rutherford-Boeing read failed with error ", rb_flag
     stop
  endif
  write(*, "(a)") "ok"

  print *, '[SpLU Test] m = ', m, ', n = ', n

  ! ! check matrix format is correct
  ! call cscl_verify(6, SPRAL_MATRIX_REAL_UNSYM, n, n, &
  !      ptr, row, flag, more)
  ! if(flag.ne.0) then
  !    print *, "CSCL_VERIFY failed: ", flag, more
  !    stop
  ! endif

  ! Make up a rhs associated with the solution x = 1.0
  allocate(rhs(n, nrhs), soln(n, nrhs))
  rhs = 0.0
  do r = 1, nrhs
     do i = 1, n
        do j = ptr(i), ptr(i+1)-1
           k = row(j)
           rhs(k, r) = rhs(k, r) + val(j)
        end do
     end do
  end do

  ssids_opt%ordering = 1 ! use Metis ordering
  ssids_opt%scaling = 0 ! no scaling

  ! Perform spldlt analysis
  options%prune_tree = .false. ! Deactivate tree pruning
  call splu_analyse(sylver_akeep, n, ptr, row, options, inform, ncpu, val=val)
  print *, "Used order ", ssids_opt%ordering
  if (inform%flag .lt. 0) then
     print *, "oops on analyse ", inform%flag
     stop
  end if
  write (*, "(a)") "ok"

  ! Factorize with SpLU
  
  splu_opt%nb = ssids_opt%cpu_block_size

  call system_clock(start_t, rate_t)
  call splu_factor(sylver_akeep, splu_fkeep, ptr, row, val, splu_opt, inform)
  call system_clock(stop_t)
     
  soln = 0.0

  print *, "number bad cmp = ", count(abs(soln(1:n,1)-1.0).ge.1e-6)
  print *, "fwd error || ||_inf = ", maxval(abs(soln(1:n,1)-1.0))
  allocate(res(nrhs))
  call internal_calc_norm(n, ptr, row, val, soln, rhs, nrhs, res)
  print *, "bwd error scaled = ", res
  
contains
  
  ! Get argument from command line
  subroutine proc_args(options, nrhs, ncpu, ngpu, matfile)
    use spral_ssids
    use spldlt_datatypes_mod, only: spldlt_options
    implicit none

    type(spldlt_options), target, intent(inout) :: options
    integer, intent(inout) :: nrhs
    integer, intent(inout) :: ncpu
    integer, intent(inout) :: ngpu
    character(len=200), intent(inout) :: matfile

    integer :: argnum, narg
    character(len=200) :: argval
    type(ssids_options), pointer :: ssids_opts

    ssids_opts => options%super

    ! default values
    nrhs = 1
    ncpu = 1
    ngpu = 0

    ! Process args
    narg = command_argument_count()
    argnum = 1
    do while(argnum <= narg)

       call get_command_argument(argnum, argval)
       argnum = argnum + 1
       select case(argval)

       case("--nemin")
          call get_command_argument(argnum, argval)
          argnum = argnum + 1
          read( argval, * ) ssids_opts%nemin
          print *, 'Supernode amalgamation nemin = ', ssids_opts%nemin

       case("--nb")
          call get_command_argument(argnum, argval)
          argnum = argnum + 1
          read( argval, * ) ssids_opts%cpu_block_size
          print *, 'CPU block size = ', ssids_opts%cpu_block_size

       case("--ncpu")
          call get_command_argument(argnum, argval)
          argnum = argnum + 1
          read( argval, * ) ncpu
          print *, 'Number of CPUs = ', ncpu

       case("--ngpu")
          call get_command_argument(argnum, argval)
          argnum = argnum + 1
          read( argval, * ) ngpu
          print *, 'Number of GPUs = ', ngpu

       case("--mat")
          call get_command_argument(argnum, argval)
          argnum = argnum + 1
          read( argval, * ) matfile
          print *, 'Matrix = ', matfile

       case("--nrhs")
          call get_command_argument(argnum, argval)
          argnum = argnum + 1
          read( argval, * ) nrhs
          print *, 'solving for', nrhs, 'right-hand sides'         

       case default
          print *, "Unrecognised command line argument: ", argval
          stop
       end select
    end do


  end subroutine proc_args

  subroutine internal_calc_norm(n, ptr, row, val, x_vec, b_vec, nrhs, res)
    integer, intent(in) :: n
    integer(long), dimension(n+1), intent(in) :: ptr
    integer, dimension(ptr(n+1)-1), intent(in) :: row
    real(wp), dimension(ptr(n+1)-1), intent(in) :: val
    integer, intent(in) :: nrhs
    real(wp), dimension(nrhs*n), intent(in) :: x_vec
    real(wp), dimension(nrhs*n), intent(in) :: b_vec
    real(wp), dimension(nrhs), intent(out) :: res

    integer :: i, k, r
    integer(long) :: j
    double precision, allocatable, dimension(:) :: x_norm
    real(wp), dimension(:), allocatable :: res_vec
    double precision :: temp
    double precision :: normA

    ! Find the residual
    allocate(res_vec(n*nrhs), x_norm(nrhs))
    res_vec = 0
    do i = 1, n
       do j = ptr(i), ptr(i+1)-1
          r = row(j)
          do k = 0, nrhs-1
             res_vec(r+k*n) = res_vec(r+k*n) + &
                  val(j) * x_vec(i+k*n)
          end do
       end do
    end do
    res_vec(:) = res_vec(:) - b_vec(:)
    
    ! Find matrix norm
    call matrix_inf_norm(n, ptr, row, val, normA)

    ! Find x norm
    do i = 1, nrhs
       x_norm(i) = 0
       do j =1, n
          x_norm(i) = max(x_norm(i), abs(x_vec((i-1)*n+j)))
          if(x_vec((i-1)*n+j).ne.x_vec((i-1)*n+j)) then ! Tests for NaN
             x_norm(i) = x_vec((i-1)*n+j)
             exit
          endif
       end do
    end do

    ! Scaled residual = ||r|| / ( ||A|| ||x|| + ||b|| )
    do i = 1, nrhs
       temp = normA * x_norm(i) + &
            maxval(abs(b_vec((i-1)*n+1:i*n)))
       if(temp <= 0.d0) then
          res(i) = maxval(abs(res_vec((i-1)*n+1:i*n)))
       else
          res(i) = maxval(abs(res_vec((i-1)*n+1:i*n))) / temp
       endif
    end do
  end subroutine internal_calc_norm

  subroutine matrix_inf_norm(n, ptr, row, val, norm)
    integer, intent(in) :: n
    integer(long), dimension(n+1), intent(in) :: ptr
    integer, dimension(ptr(n+1)-1), intent(in) :: row
    real(wp), dimension(ptr(n+1)-1), intent(in) :: val
    real(wp), intent(out) :: norm

    real(wp), allocatable, dimension(:) :: row_norm
    integer(long) :: i

    allocate(row_norm(n))

    row_norm = 0
    do i = 1, ptr(n+1)-1
       row_norm(row(i)) = row_norm(row(i)) + abs(val(i))
    end do

    norm = maxval(row_norm) 
  end subroutine matrix_inf_norm

end program
