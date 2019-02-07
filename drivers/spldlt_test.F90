program spldlt_test
   use, intrinsic :: iso_c_binding
   use spral_ssids
   use spral_rutherford_boeing
   ! use spral_matrix_util, only : cscl_verify, SPRAL_MATRIX_REAL_SYM_INDEF
   use spldlt_analyse_mod
   use spldlt_factorize_mod
   use spldlt_mod
   use spldlt_datatypes_mod, only: spldlt_options 
   implicit none

   ! FIXME use the one defined in ssids for now
   ! integer, parameter :: wp = kind(0d0)

   type(spldlt_options), target :: options
   type(ssids_options), pointer :: ssids_opt
   logical :: pos_def ! if true, assume matrix is posdef

   ! matrix reader options
   type(rb_read_options) :: rb_options
   integer :: rb_flag

   ! Matrix description
   character(len=200) :: matfile = ''
   integer :: m, n
   integer(long), dimension(:), allocatable :: ptr
   integer, dimension(:), allocatable :: row
   real(wp), dimension(:), allocatable :: val
   
   integer :: nb ! Block-size
   integer :: ncpu ! Number of CPUs
   integer :: ngpu ! Number of GPUs

   ! right-hand side and solution
   integer :: nrhs
   double precision, dimension(:,:), allocatable :: rhs, soln 
   ! double precision, dimension(:), allocatable :: scaling
   double precision, dimension(:), allocatable :: res

   ! indexes
   integer :: r, i
   integer :: k
   integer(long) :: j

   ! timing
   integer :: start_t, stop_t, rate_t
   ! flags
   ! integer :: flag, more
   
   ! ssids structures
   type(ssids_inform) :: inform ! stats
   ! spldlt strucutres
   type(spldlt_akeep_type) :: spldlt_akeep
   type(spldlt_fkeep_type) :: spldlt_fkeep

   ! stats
   real :: smfact
   ! real :: smanal, smaflop, smafact

   ! integer :: cuda_error ! DEBUG not useful for now 

   pos_def = .false. ! Matrix assumed indef by default
   
   call proc_args(options, nrhs, pos_def, ncpu, ngpu, matfile)

   ssids_opt => options%super ! Point to SSIDS options
   
   ! ssids_opt%print_level = 1 ! enable printing
   ssids_opt%print_level = 0 ! disable printing
   ssids_opt%use_gpu = .false. ! disable GPU
   
   if (matfile.eq.'') matfile = "matrix.rb" 

   print *, "[SpLDLT Test] ncpu: ", ncpu
   print *, "[SpLDLT Test]   nb: ", ssids_opt%cpu_block_size

   ! Read in a matrix
   write(*, "(a)") "Reading..."
   ! DEBUG ensure matrix is diag dominant
    ! rb_options%values = -3 ! Force diagonal dominance
   rb_options%values = 2 ! symmetric make up values if necessary (if only pattern is given)
   ! rb_options%values = 0 ! as per file
   call rb_read(matfile, m, n, ptr, row, val, rb_options, rb_flag)
   if(rb_flag.ne.0) then
      print *, "Rutherford-Boeing read failed with error ", rb_flag
      stop
   endif
   write(*, "(a)") "ok"

   ! Make up a rhs associated with the solution x = 1.0
   allocate(rhs(n, nrhs), soln(n, nrhs))
   rhs = 0
   do r = 1, nrhs
      do i = 1, n
         do j = ptr(i), ptr(i+1)-1
            k = row(j)
            rhs(k, r) = rhs(k, r) + val(j)
            if(i.eq.k) cycle
            rhs(i, r) = rhs(i, r) + val(j)
         end do
      end do
   end do

   ! ! check matrix format is correct
   ! call cscl_verify(6, SPRAL_MATRIX_REAL_SYM_INDEF, n, n, &
   !      ptr, row, flag, more)
   ! if(flag.ne.0) then
   !    print *, "CSCL_VERIFY failed: ", flag, more
   !    stop
   ! endif

   ssids_opt%ordering = 1 ! Use Metis ordering
   ssids_opt%scaling = 0 ! No scaling

   ! Initialize SpLDLT
   call spldlt_init(ncpu, ngpu)

   ! Perform analysis
   options%prune_tree = .true. ! Deactivate tree pruning
   ! options%prune_tree = .false. ! Deactivate tree pruning
   call spldlt_analyse(spldlt_akeep, n, ptr, row, options, inform, val=val, ncpu=ncpu, ngpu=ngpu)
   ! print atree
   ! call spldlt_print_atree(akeep)
   ! print atree with partitions
   ! call spldlt_print_atree_part(spldlt_akeep%akeep)
   ! stop

   ! Factorize matrix
   call system_clock(start_t, rate_t)
   call spldlt_factorize(spldlt_akeep, spldlt_fkeep, pos_def, val, options, inform)
   call system_clock(stop_t)
   write(*, "(a)") "ok"
   print *, "Factor took ", (stop_t - start_t)/real(rate_t)
   smfact = (stop_t - start_t)/real(rate_t)

   ! Shutdown SpLDLT
   call spldlt_finalize()

   ! Solve
   write(*, "(a)") "[SpLDLT Test] Solve..."

   ! Solve SpLDLT
   call system_clock(start_t, rate_t)
   soln = rhs ! init solution with RHS
   call spldlt_fkeep%solve(spldlt_akeep, nrhs, soln, n, inform)
   call system_clock(stop_t)
   write(*, "(a)") "ok"
   print *, "Solve took ", (stop_t - start_t)/real(rate_t)

   ! print *, "RHS: ", rhs
   ! print *, "soln: ", soln

   print *, "number bad cmp = ", count(abs(soln(1:n,1)-1.0).ge.1e-6)
   print *, "fwd error || ||_inf = ", maxval(abs(soln(1:n,1)-1.0))
   allocate(res(nrhs))
   call internal_calc_norm(n, ptr, row, val, soln, rhs, nrhs, res)
   print *, "bwd error scaled = ", res

   print "(a6, es10.2)", "nfact:", real(inform%num_factor)
   print "(a6, es10.2)", "nflop:", real(inform%num_flops)
   print "(a6, i10)", "delay:", inform%num_delay
   print "(a6, i10)", "2x2piv:", inform%num_two
   print "(a6, i10)", "maxfront:", inform%maxfront
   print "(a6, i10)", "not_first_pass:", inform%not_first_pass
   print "(a6, i10)", "not_second_pass:", inform%not_second_pass

   stop
   
 contains

   ! Get argument from command line
   subroutine proc_args(options, nrhs, pos_def, ncpu, ngpu, matfile)
     use spral_ssids
     use spldlt_datatypes_mod, only: spldlt_options
     implicit none

     type(spldlt_options), target, intent(inout) :: options
     integer, intent(inout) :: nrhs
     logical, intent(inout) :: pos_def
     integer, intent(inout) :: ncpu
     integer, intent(inout) :: ngpu
     character(len=200), intent(inout) :: matfile

     integer :: argnum, narg
     character(len=200) :: argval
     type(ssids_options), pointer :: ssids_opts
     
     ssids_opts => options%super

     ! default values
     nb = 2
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
        case("--nrhs")
           call get_command_argument(argnum, argval)
           argnum = argnum + 1
           read( argval, * ) nrhs
           print *, 'solving for', nrhs, 'right-hand sides'         
        case("--indef")
           pos_def = .false.
           print *, 'Matrix assumed indefinite'
        case("--posdef")
           pos_def = .true.
           print *, 'Matrix assumed positive definite'
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
        case("--failed-pivot-method=tpp")
           ssids_opts%failed_pivot_method = 1
           print *, 'Failed pivot method TPP'
        case("--failed-pivot-method=pass")
           ssids_opts%failed_pivot_method = 2
           print *, 'Failed pivot method PASS'
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
              res_vec(i+k*n) = res_vec(i+k*n) + &
                   val(j) * x_vec(r+k*n)
           end do
           if(r.eq.i) cycle
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

end program spldlt_test
