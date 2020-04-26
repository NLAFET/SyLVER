!> \file
!> \copyright 2018- The Science and Technology Facilities Council (STFC)
!> \author    Sebastien Cayrols

module sylver_ciface
  use iso_c_binding
  implicit none

  type, bind(C) :: sylver_options_t
     integer(c_int) :: array_base
     ! Printing options
     integer(c_int) :: print_level
     integer(c_int) :: unit_diagnostics 
     integer(c_int) :: unit_error
     integer(c_int) :: unit_warning
     ! Options used in spldlt_analyse() and splu_analyse() 
     integer(c_int) :: ordering
     integer(c_int) :: nemin
     ! Tree partitioning
     logical(c_bool) :: prune_tree
     integer(c_long) :: min_gpu_work
     ! Options used by spldlt_factor() and splu_factor()
     integer(c_int) :: scaling
     ! Options used by spldlt_factor() and splu_factor() for
     ! controlling pivoting
     integer(c_int) :: pivot_method
     real(c_double) :: small
     real(c_double) :: u
     ! CPU-specific
     integer(c_long) :: small_subtree_threshold
     integer(c_int) :: nb
     integer(c_int) :: cpu_topology
     ! Options used by spldlt_factorize() with posdef=.false.
     logical(c_bool) :: action
     ! GPU-specific
     logical(c_bool) :: use_gpu
     real(c_double) :: gpu_perf_coeff
     ! Undocumented
     integer(c_int) :: failed_pivot_method
     integer(c_int) :: scheduler
  end type sylver_options_t

  type, bind(C) :: sylver_inform_t
     integer(c_int) :: flag
     integer(c_int) :: matrix_dup
     integer(c_int) :: matrix_missing_diag ! Number of missing diag. entries
     integer(c_int) :: matrix_outrange ! Number of out-of-range entries.
     integer(c_int) :: matrix_rank ! Rank of matrix (anal=structral, fact=actual)
     integer(c_int) :: maxdepth ! Maximum depth of tree
     integer(c_int) :: maxfront ! Maximum front size
     integer(c_int) :: num_delay ! Number of delayed variables
     integer(c_long) :: num_factor ! Number of entries in factors
     integer(c_long) :: num_flops ! Number of floating point operations
     integer(c_int) :: num_neg ! Number of negative pivots
     integer(c_int) :: num_sup ! Number of supernodes
     integer(c_int) :: num_two ! Number of 2x2 pivots used by factorization
     integer(c_int) :: stat ! stat parameter
     integer(c_int) :: cuda_error
     integer(c_int) :: cublas_error
     character(c_char) :: unused(80)     
     ! Undocumented FIXME: should we document them?
     ! integer(c_int) :: not_first_pass = 0
     ! integer(c_int) :: not_second_pass = 0
     ! integer(c_int) :: nparts = 0
     ! integer(c_long) :: cpu_flops = 0
     ! integer(c_long) :: gpu_flops = 0
     ! integer(c_int) :: num_part = 0 ! Number of partitions in the etree
     ! character(c_char) :: unused(80)
  end type sylver_inform_t
     
  ! interface check_backward_error
  !   module procedure check_backward_error_one
  !   module procedure check_backward_error_multi
  ! end interface check_backward_error

contains

  subroutine sylver_copy_options_in(coptions, foptions, cindexed)
    use sylver_datatypes_mod
    implicit none

    type(sylver_options_t), intent(in)    :: coptions
    type(sylver_options),   intent(inout) :: foptions ! inherit some defaults!
    logical,                intent(out)   :: cindexed

    cindexed                   = (coptions%array_base .eq. 0)
    ! Printing options
    foptions%print_level       = coptions%print_level
    foptions%unit_diagnostics  = coptions%unit_diagnostics
    foptions%unit_error        = coptions%unit_error
    foptions%unit_warning      = coptions%unit_warning
    ! Options used in spldlt_analyse() and splu_analyse() 
    foptions%ordering          = coptions%ordering
    foptions%nemin             = coptions%nemin
    ! Tree partitioning
    foptions%prune_tree        = coptions%prune_tree
    foptions%min_gpu_work      = coptions%min_gpu_work
    ! Options used by spldlt_factor() and splu_factor()
    foptions%scaling           = coptions%scaling
    ! Options used by spldlt_factor() and splu_factor() for
    ! controlling pivoting
    foptions%pivot_method      = coptions%pivot_method
    foptions%small             = coptions%small
    foptions%u                 = coptions%u
    ! CPU-specific
    foptions%small_subtree_threshold = coptions%small_subtree_threshold
    foptions%nb                = coptions%nb
    foptions%cpu_topology      = coptions%cpu_topology
    ! Options used by spldlt_factorize() with posdef=.false.
    foptions%action            = coptions%action
    ! GPU-specific    
    foptions%use_gpu           = coptions%use_gpu
    foptions%gpu_perf_coeff    = coptions%gpu_perf_coeff
    ! Undocumented
    foptions%failed_pivot_method     = coptions%failed_pivot_method
    foptions%sched         = coptions%scheduler
    
    ! foptions%max_load_inbalance= coptions%max_load_inbalance
    
  end subroutine sylver_copy_options_in

  subroutine sylver_copy_inform_out(finform, cinform)
    use sylver_inform_mod
    implicit none

    type(sylver_inform), intent(in) :: finform
    type(sylver_inform_t), intent(out) :: cinform

    cinform%flag                  = finform%flag
    cinform%matrix_dup            = finform%matrix_dup
    cinform%matrix_missing_diag   = finform%matrix_missing_diag
    cinform%matrix_outrange       = finform%matrix_outrange
    cinform%matrix_rank           = finform%matrix_rank
    cinform%maxdepth              = finform%maxdepth
    cinform%maxfront              = finform%maxfront
    cinform%num_delay             = finform%num_delay
    cinform%num_factor            = finform%num_factor
    cinform%num_flops             = finform%num_flops
    cinform%num_neg               = finform%num_neg
    cinform%num_sup               = finform%num_sup
    cinform%num_two               = finform%num_two
    cinform%stat                  = finform%stat
    cinform%cuda_error            = finform%cuda_error
    cinform%cublas_error          = finform%cublas_error
    
  end subroutine sylver_copy_inform_out

  ! subroutine compute_residual(n, ptr, row, val, nrhs, x, b, res)
  !   implicit none
    
  !   integer, intent(in)                         :: n
  !   integer(long), dimension(n+1), intent(in)   :: ptr
  !   integer, dimension(ptr(n+1)-1), intent(in)  :: row
  !   real(wp), dimension(ptr(n+1)-1), intent(in) :: val
  !   integer, intent(in)                         :: nrhs
  !   real(wp), dimension(n,nrhs), intent(in)     :: x
  !   real(wp), dimension(n,nrhs), intent(in)     :: b
  !   real(wp), dimension(n,nrhs), intent(inout)  :: res

  !   ! Find the residual
  !   res = 0

  !   call compute_Ax(n, ptr, row, val, nrhs, x, res)

  !   res = b - res
  ! end subroutine compute_residual
  

  !Compute Ax
  ! subroutine compute_Ax(n, ptr, row, val, nrhs, x, res)
  !   implicit none

  !   integer, intent(in)                         :: n
  !   integer(long), dimension(n+1), intent(in)   :: ptr
  !   integer, dimension(ptr(n+1)-1), intent(in)  :: row
  !   real(wp), dimension(ptr(n+1)-1), intent(in) :: val
  !   integer, intent(in)                         :: nrhs
  !   real(wp), dimension(n,nrhs), intent(in)     :: x
  !   real(wp), dimension(n,nrhs), intent(inout)  :: res

  !   integer :: i, j, k, r
  !   res = 0
  !   do i = 1, n
  !     do j = ptr(i), ptr(i+1)-1
  !       r = row(j)
  !       do k = 1, nrhs
  !         res(r, k) = res(r, k) + val(j) * x(i, k)
  !         if(r .eq. i) cycle
  !         res(i, k) = res(i, k) + val(j) * x(r, k)
  !       end do
  !     end do
  !   end do
  ! end subroutine compute_Ax



  ! subroutine vector_norm_2(n, val, norm)
  !   implicit none

  !   integer,  intent(in)  :: n
  !   real(wp), intent(in)  :: val(:,:)
  !   real(wp), intent(out) :: norm(:)
    
  !   integer :: i
  !   norm = 0
  
  !   do i = 1, n
  !     norm(:) = norm(:) + val(i,:) * val(i, :)
  !   end do
  !   norm = sqrt(norm)

  ! end subroutine vector_norm_2



  ! subroutine matrix_norm_max(n, ptr, row, val, norm)
  !   implicit none

  !   integer, intent(in)                         :: n
  !   integer(long), dimension(n+1), intent(in)   :: ptr
  !   integer, dimension(ptr(n+1)-1), intent(in)  :: row    ! Unused
  !   real(wp), dimension(ptr(n+1)-1), intent(in) :: val
  !   real(wp), intent(out)                       :: norm

  !   integer :: i

  !   norm = 0
  !   do i = 1, ptr(n+1)-1
  !     norm = merge(norm, abs(val(i)), norm > abs(val(i)))
  !   end do
  ! end subroutine matrix_norm_max

  

  ! subroutine check_backward_error_one(n, ptr, row, val, x, b) 
  !   implicit none

  !   integer,        intent(in) :: n
  !   integer(long),  intent(in) :: ptr(n+1)
  !   integer,        intent(in) :: row(ptr(n+1)-1)
  !   real(wp),       intent(in) :: val(ptr(n+1)-1)
  !   real(wp),       intent(in) :: x(n)
  !   real(wp),       intent(in) :: b(n)

  !   call check_backward_error_multi(n, ptr, row, val, 1, x, b)

  ! end subroutine check_backward_error_one



  ! subroutine check_backward_error_multi(n, ptr, row, val, nrhs, x, b) 
  !   implicit none

  !   integer, intent(in)                         :: n
  !   integer(long), dimension(n+1), intent(in)   :: ptr
  !   integer, dimension(ptr(n+1)-1), intent(in)  :: row
  !   real(wp), dimension(ptr(n+1)-1), intent(in) :: val
  !   integer, intent(in)                         :: nrhs
  !   real(wp), dimension(n,nrhs), intent(in)     :: x
  !   real(wp), dimension(n,nrhs), intent(in)     :: b

  !   integer           :: i
  !   real(wp)          :: norm_max
  !   real(wp)          :: res(n, nrhs)
  !   double precision  :: normRes(nrhs)
  !   double precision  :: normRHS(nrhs)
  !   double precision  :: solNorm(nrhs)
  !   double precision  :: err
  !   integer           :: cpt

  !   call compute_residual(n, ptr, row, val, nrhs, x, b, res)

  !   call vector_norm_2(n, res, normRes)
  !   call vector_norm_2(n,   b, normRHS)
  !   call vector_norm_2(n,   x, solNorm)
  !   call matrix_norm_max(n, ptr, row, val, norm_max)
  !  !print *, "resNorm", normRes
  !  !print *, "rhsNorm", normRHS
  !  !print *, "solNorm", solNorm
  !  !print *, "ANorm", norm_max

  !   cpt = 0
  !   do i = 1, nrhs
  !   err = normRes(i) / (normRHS(i) + norm_max * solNorm(i))
  !   if(err .ne. err) then
  !     print '(a, i3, a)', "Backward error of rhs ", i, " is equal to a NAN"
  !   else
  !     !if(normRes(i) / normRHS(i) .gt. 1e-14) then
  !     if(err .gt. 1e-14) then
  !       write(0, "(a, i4, a, i4, a, es10.2)") "Wrong Bwd error for ", i, &
  !         "/", nrhs, " : ", err
  !     else
  !       write(0, "(a, i4, a, i4, a, es10.2)") "Bwd error for ", i, &
  !         "/", nrhs, " : ", err
  !       cpt = cpt + 1
  !     end if
  !   end if
  !   end do
  !   write(0, "(a, i3, a, i3)") "Backward error... ok for ", cpt, "/", nrhs
  ! end subroutine check_backward_error_multi

end module sylver_ciface

subroutine sylver_c_default_options(coptions) &
    bind(C, name="sylver_default_options")
  use sylver_ciface
  use sylver_datatypes_mod
  implicit none

  type(sylver_options_t), intent(out) :: coptions

  type(sylver_options)  :: default_options

  coptions%array_base        = 0 ! C
  ! Printing options
  coptions%print_level       = default_options%print_level
  coptions%unit_diagnostics  = default_options%unit_diagnostics
  coptions%unit_error        = default_options%unit_error
  coptions%unit_warning      = default_options%unit_warning
  ! Options used in spldlt_analyse() and splu_analyse()
  coptions%ordering          = default_options%ordering
  coptions%nemin             = default_options%nemin
  ! Tree partitioning
  coptions%prune_tree                = default_options%prune_tree
  coptions%min_gpu_work      = default_options%min_gpu_work
  ! Options used by spldlt_factor() and splu_factor()
  coptions%scaling           = default_options%scaling
  ! Options used by spldlt_factor() and splu_factor() for
  ! controlling pivoting
  coptions%pivot_method      = default_options%pivot_method
  coptions%small             = default_options%small
  coptions%u                 = default_options%u
  ! CPU-specific
  coptions%small_subtree_threshold = default_options%small_subtree_threshold
  coptions%nb                = default_options%nb
  coptions%cpu_topology      = default_options%cpu_topology
  ! Options used by spldlt_factorize() with posdef=.false.
  coptions%action            = default_options%action
  ! GPU-specific    
  coptions%use_gpu           = default_options%use_gpu
  coptions%gpu_perf_coeff    = default_options%gpu_perf_coeff
  ! Undocumented
  coptions%failed_pivot_method     = default_options%failed_pivot_method
  coptions%scheduler         = default_options%sched

end subroutine sylver_c_default_options


subroutine sylver_c_init(ncpu, ngpu) &
     bind(C, name="sylver_init")
  use sylver_ciface
  use sylver_mod
  implicit none

  integer(C_INT), value :: ncpu
  integer(C_INT), value :: ngpu

  call sylver_init(ncpu, ngpu)

end subroutine sylver_c_init


subroutine sylver_c_finalize() &
    bind(C, name="sylver_finalize")
  use sylver_ciface
  use sylver_mod
  implicit none

  call sylver_finalize()

end subroutine sylver_c_finalize


subroutine spldlt_c_analyse(n, corder, cptr, crow, cval, cakeep, ccheck, coptions, &
     cinform) bind(C, name="spldlt_analyse")
  use sylver_ciface
  use sylver_datatypes_mod
  use spldlt_analyse_mod
  implicit none

  integer(c_int), value :: n
  type(c_ptr), value :: corder
  type(c_ptr), value :: cptr
  type(c_ptr), value :: crow
  type(c_ptr), value :: cval
  !   integer(c_int),   value :: ncpu
  type(c_ptr), intent(inout) :: cakeep
  logical(c_bool), value :: ccheck
  type(sylver_options_t), intent(in) :: coptions
  type(sylver_inform_t), intent(out)   :: cinform

  logical :: fcheck
  integer(C_INT), dimension(:), pointer :: forder
  integer(C_LONG),          pointer :: fptr(:)
  integer(C_INT),           pointer :: frow(:)
  real(C_DOUBLE),           pointer :: fval(:)
  type(spldlt_akeep_type),  pointer :: fakeep
  type(sylver_options)              :: foptions
  type(sylver_inform)                :: finform
  integer                           :: st
  logical :: cindexed

  ! Copy options in first to find out whether we use Fortran or C indexing
  call sylver_copy_options_in(coptions, foptions, cindexed)

  ! Translate arguments
  fcheck = ccheck

  if (C_ASSOCIATED(corder)) then
     call c_f_pointer(corder, forder, shape=(/ n /))
  else
     nullify(forder)
  end if

  if (C_ASSOCIATED(cptr)) then
     call c_f_pointer(cptr, fptr, shape=(/ n+1 /))
  else
     ! print *, "Error, cptr is not associated"
     nullify(fptr)
  end if

  if (C_ASSOCIATED(crow)) then
     call c_f_pointer(crow, frow, shape=(/ fptr(n+1)-1 /))
  else
     ! print *, "Error, crow is not associated"
     nullify(frow)
  end if

  if (C_ASSOCIATED(cval)) then
     call c_f_pointer(cval, fval, shape=(/ fptr(n+1)-1 /))
  else
     ! print *, "Warning, cval is not associated"
     nullify(fval)
  end if

  if (C_ASSOCIATED(cakeep)) then
     ! Reuse old pointer
     call c_f_pointer(cakeep, fakeep)
  else
     ! Create new pointer
     allocate(fakeep, stat=st)
     cakeep = c_loc(fakeep)
  end if

  ! Call Fortran routine
  if (associated(forder)) then
     if (associated(fval)) then
        call spldlt_analyse(fakeep, n, fptr, frow, foptions, finform, order=forder, &
             val=fval, check=fcheck)
     else
        call spldlt_analyse(fakeep, n, fptr, frow, foptions, finform, order=forder, &
             check=fcheck)
     end if
  else
     if (associated(fval)) then
        call spldlt_analyse(fakeep, n, fptr, frow, foptions, finform, &
             val=fval, check=fcheck)
     else
        call spldlt_analyse(fakeep, n, fptr, frow, foptions, finform, &
             check=fcheck)
     end if
  end if

  ! Copy arguments out
  call sylver_copy_inform_out(finform, cinform)

end subroutine spldlt_c_analyse



!subroutine spldlt_c_analyse_debug(n, cptr, crow, cval, ncpu, cakeep, coptions, &
!  cinform, dumpMat) bind(C, name="spldlt_analyse_d")
!  use spldlt_ciface
!  use spldlt_datatypes_mod
!  use spldlt_analyse_mod
!  use spral_rutherford_boeing
!  use spral_matrix_util
!  implicit none
!
!  integer(C_INT),   value :: n
!  type(C_PTR),      value :: cptr
!  type(C_PTR),      value :: crow
!  type(C_PTR),      value :: cval
!  integer(C_INT),   value :: ncpu
!  integer(C_INT),   value :: dumpMat
!  type(C_PTR),              intent(inout) :: cakeep
!  type(spldlt_options_t),   intent(in)    :: coptions
!  type(spral_ssids_inform), intent(out)   :: cinform
!
!  integer(C_LONG),          pointer :: fptr(:)
!  integer(C_INT),           pointer :: frow(:)
!  real(C_DOUBLE),           pointer :: fval(:)
!  integer                           :: st
!  character(len=1024)               :: filename
!  integer                           :: iunit
!  type(rb_write_options)            :: options
!  integer                           :: info
!
!  if (C_ASSOCIATED(cptr)) then
!    call C_F_POINTER(cptr, fptr, shape=(/ n+1 /))
!  else
!    print *, "Error, cptr is not associated"
!    nullify(fptr)
!  end if
!
!  if (C_ASSOCIATED(crow)) then
!    call C_F_POINTER(crow, frow, shape=(/ fptr(n+1)-1 /))
!  else
!    print *, "Error, crow is not associated"
!    nullify(frow)
!  end if
!
!  if (C_ASSOCIATED(cval)) then
!    call C_F_POINTER(cval, fval, shape=(/ fptr(n+1)-1 /))
!  else
!    print *, "Warning, cval is not associated"
!    nullify(fval)
!  end if
!
!  if (dumpMat .ge. 0) then
!    write (filename, "(A12,I2.2,A3)") "dump_analmat", dumpMat, ".rb"
!    call rb_write(filename, SPRAL_MATRIX_REAL_UNSYM, n, n, &
!      fptr, frow, options, info, fval)
!  end if
!
!  call spldlt_c_analyse(n, cptr, crow, cval, ncpu, cakeep, coptions, cinform)
!
!end subroutine spldlt_c_analyse_debug


subroutine spldlt_c_factorize(cposdef, cptr, crow, val, cscale, cakeep, cfkeep, coptions, &
     cinform) bind(C, name="spldlt_factorize")
  use sylver_datatypes_mod
  use spldlt_analyse_mod
  use spldlt_factorize_mod
  use sylver_ciface
  implicit none

  logical(c_bool), value :: cposdef
  type(c_ptr), value :: cptr
  type(c_ptr), value :: crow
  real(c_double), intent(in) :: val(*)
  type(c_ptr), value :: cscale
  type(c_ptr), value :: cakeep
  type(c_ptr), intent(inout) :: cfkeep
  type(sylver_options_t), intent(in) :: coptions
  type(sylver_inform_t), intent(out) :: cinform
  
  logical :: fposdef
  integer(C_LONG), dimension(:), pointer :: fptr
  integer(C_INT), dimension(:), pointer :: frow
  real(C_DOUBLE), dimension(:), pointer :: fscale
  type(spldlt_akeep_type), pointer :: fakeep
  type(spldlt_fkeep_type), pointer :: ffkeep
  type(sylver_options) :: foptions
  type(sylver_inform) :: finform

  integer :: n
  logical :: cindexed

  ! Copy options in first to find out whether we use Fortran or C indexing
  call sylver_copy_options_in(coptions, foptions, cindexed)

  ! Translate arguments
  fposdef = cposdef
  ! akeep
  call c_f_pointer(cakeep, fakeep) ! Pulled forward so we can use it

  n = fakeep%akeep%n
  ! ptr
  if (c_associated(cptr)) then
     call c_f_pointer(cptr, fptr, shape=(/ n+1 /))
     ! if (.not. cindexed) then
     !    allocate(fptr_alloc(fakeep%n+1))
     !    fptr_alloc(:) = fptr(:) + 1
     !    fptr => fptr_alloc
     ! end if
  else
     nullify(fptr)
  end if

  ! row
  if (c_associated(crow)) then
     call c_f_pointer(crow, frow, shape=(/ fptr(n+1)-1 /))
     ! if (.not. cindexed) then
     !    allocate(frow_alloc(fakeep%n+1))
     !    frow_alloc(:) = frow(:) + 1
     !    frow => frow_alloc
     ! end if
  else
     nullify(frow)
  end if

  ! scale
  if (c_associated(cscale)) then
     call C_F_POINTER(cscale, fscale, shape=(/ n /))
  else
     nullify(fscale)
  end if

  ! fkeep
  if (c_associated(cfkeep)) then
     ! Reuse old pointer
     call c_f_pointer(cfkeep, ffkeep)
  else
     ! Create new pointer
     allocate(ffkeep)
     cfkeep = c_loc(ffkeep)
  end if

  ! Call spldlt_factor Fortran routine
  if (associated(fptr) .and. associated(frow)) then
     if (associated(fscale)) then
        call spldlt_factorize(fakeep, ffkeep, fposdef, val, foptions, finform, &
             scale=fscale, ptr=fptr, row=frow)
     else
        call spldlt_factorize(fakeep, ffkeep, fposdef, val, foptions, finform, &
             ptr=fptr, row=frow)
     end if
  else
     if (associated(fscale)) then
        call spldlt_factorize(fakeep, ffkeep, fposdef, val, foptions, finform, &
             scale=fscale)
     else
        call spldlt_factorize(fakeep, ffkeep, fposdef, val, foptions, finform)
     end if
  end if
  
  ! Copy arguments out
  call sylver_copy_inform_out(finform, cinform)

end subroutine spldlt_c_factorize

! subroutine spldlt_c_solve(job, nrhs, cx, ldx, cakeep, cfkeep, cinform)&
!     bind(C, name="spldlt_solve")
!   use sylver_datatypes_mod
!   use spldlt_analyse_mod
!   use spldlt_factorize_mod
!   use spldlt_ciface
!   implicit none

!   integer(C_INT),      value            :: job ! unused right now
!   integer(C_INT),      value            :: nrhs
!   type(C_PTR),         value            :: cx
!   integer(C_INT),      value            :: ldx
!   type(C_PTR),         value            :: cakeep
!   type(C_PTR),         value            :: cfkeep
!   type(spral_ssids_inform), intent(out) :: cinform

!   real(C_DOUBLE),          pointer  :: fx(:,:)
!   type(spldlt_akeep_type), pointer  :: fakeep
!   type(spldlt_fkeep_type), pointer  :: ffkeep
!   type(ssids_inform)                :: finform

!   ! Translate arguments
!   if (C_ASSOCIATED(cx)) then
!      call C_F_POINTER(cx, fx, shape=(/ ldx,nrhs /))
!   else
!      nullify(fx)
!   end if
!   if (C_ASSOCIATED(cakeep)) then
!      call C_F_POINTER(cakeep, fakeep)
!   else
!      nullify(fakeep)
!   end if
!   if (C_ASSOCIATED(cfkeep)) then
!      call C_F_POINTER(cfkeep, ffkeep)
!   else
!      nullify(ffkeep)
!   end if

!  !print *, "Call solve"
!   ! Call Fortran routine
!   call ffkeep%solve(fakeep, nrhs, fx, ldx, finform)

!   ! Copy arguments out
!   call copy_inform_out(finform, cinform)
! end subroutine spldlt_c_solve



!subroutine spldlt_c_solve_debug(job, nrhs, cx, ldx, cakeep, cfkeep, &
!    cinform, dumpRhs) bind(C, name="spldlt_solve_d")
!  use spldlt_datatypes_mod
!  use spldlt_analyse_mod
!  use spldlt_factorize_mod
!  use spldlt_ciface
!  implicit none
!
!  integer(C_INT),      value            :: job ! unused right now
!  integer(C_INT),      value            :: nrhs
!  type(C_PTR),         value            :: cx
!  integer(C_INT),      value            :: ldx
!  type(C_PTR),         value            :: cakeep
!  type(C_PTR),         value            :: cfkeep
!  integer(C_INT),      value            :: dumpRhs
!  type(spral_ssids_inform), intent(out) :: cinform
!
!  real(C_DOUBLE),          pointer  :: fx(:,:)
!  character(len=1024)               :: filename
!  integer                           :: st
!  integer                           :: iunit
!
!  if (C_ASSOCIATED(cx)) then
!     call C_F_POINTER(cx, fx, shape=(/ ldx,nrhs /))
!  else
!     nullify(fx)
!  end if
!
!  if (dumpRhs .ge. 0) then
!    write (filename, "(A8,I2.2,A4)") "dump_rhs", dumpRhs, ".txt"
!    ! Open file
!    open(file=filename, newunit=iunit, status='replace', iostat=st)
!    if (st .eq. 0) then
!      ! Write vector size
!      write(iunit, "(I6, I6)") ldx, nrhs
!
!      ! Write vector data
!      if (associated(fx)) then
!        write(iunit, "(3e24.16)") fx(:,:)
!      end if
!
!      ! Close file
!      close(iunit)
!    end if
!  end if
!
!  call spldlt_c_solve(job, nrhs, cx, ldx, cakeep, cfkeep, cinform)
!
!  if (dumpRhs .ge. 0) then
!    write (filename, "(A8,I2.2,A4)") "dump_sol", dumpRhs, ".txt"
!    ! Open file
!    open(file=filename, newunit=iunit, status='replace', iostat=st)
!    if (st .eq. 0) then
!      ! Write vector size
!      write(iunit, "(I6, I6)") ldx, nrhs
!
!      ! Write vector data
!      if (associated(fx)) then
!        write(iunit, "(3e24.16)") fx(:,:)
!      end if
!
!      ! Close file
!      close(iunit)
!    end if
!  end if
!
!end subroutine spldlt_c_solve_debug



! integer(C_INT) function spldlt_c_free_akeep(cakeep) &
!     bind(C, name="spldlt_free_akeep")
!   use spldlt_ciface
!   use spldlt_analyse_mod
!   use spral_ssids, only: ssids_free
!   implicit none
   
!   type(C_PTR), intent(inout) :: cakeep

!   type(spldlt_akeep_type), pointer :: fakeep

!   if (.not. C_ASSOCIATED(cakeep)) then
!      ! Nothing to free
!      spldlt_c_free_akeep = 0
!      return
!   end if

!   call C_F_POINTER(cakeep, fakeep)
!   !TODO should be replaced by a spldlt_free subroutine, when it exists
!   call ssids_free(fakeep%akeep, spldlt_c_free_akeep)
!   if (allocated(fakeep%subtree_en)) then
!     deallocate(fakeep%subtree_en)
!   end if
!   deallocate(fakeep)
!   cakeep = C_NULL_PTR
! end function spldlt_c_free_akeep



! integer(C_INT) function spldlt_c_free_fkeep(cfkeep) &
!     bind(C, name="spldlt_free_fkeep")
!   use spldlt_ciface
!   use spldlt_factorize_mod
!   use spral_ssids, only: ssids_free
!   implicit none
   
!   type(C_PTR), intent(inout) :: cfkeep

!   type(spldlt_fkeep_type), pointer :: ffkeep

!   if (.not. C_ASSOCIATED(cfkeep)) then
!      ! Nothing to free
!      spldlt_c_free_fkeep = 0
!      return
!   end if

!   call C_F_POINTER(cfkeep, ffkeep)
!   call ssids_free(ffkeep%fkeep, spldlt_c_free_fkeep)
!   deallocate(ffkeep)
!   cfkeep = C_NULL_PTR
! end function spldlt_c_free_fkeep



! subroutine spldlt_c_check_backward_error(n, cptr, crow, cval, nrhs, cx, crhs)&
!     bind(C, name="spldlt_chkerr")
!   use spldlt_ciface
!   use ISO_Fortran_env, only: stderr => ERROR_UNIT
!   implicit none

!   integer(C_INT), value  :: n
!   type(C_PTR),    value  :: cptr
!   type(C_PTR),    value  :: crow
!   type(C_PTR),    value  :: cval
!   integer(C_INT), value  :: nrhs
!   type(C_PTR),    value  :: cx
!   type(C_PTR),    value  :: crhs

!   integer(C_LONG),    pointer :: fptr(:)
!   integer(C_INT),     pointer :: frow(:)
!   real(wp),           pointer :: fval(:)
!   real(wp),           pointer :: fx(:,:)
!   real(wp),           pointer :: frhs(:,:)

!   if(.not. C_ASSOCIATED(cptr)) then
!     write (stderr,*) "Error, ptr provided by the user is empty"
!   end if

!   if(.not. C_ASSOCIATED(crow)) then
!     write (stderr,*) "Error, row provided by the user is empty"
!   end if

!   if(.not. C_ASSOCIATED(cval)) then
!     write (stderr,*) "Error, val provided by the user is empty"
!   end if

!   if(.not. C_ASSOCIATED(cx)) then
!     write (stderr,*) "Error, x provided by the user is empty"
!   end if

!   if(.not. C_ASSOCIATED(crhs)) then
!     write (stderr,*) "Error, rhs provided by the user is empty"
!   end if

!   call C_F_POINTER(cptr, fptr, shape=(/ n + 1 /))
!   call C_F_POINTER(crow, frow, shape=(/ fptr(n + 1) - 1 /))
!   call C_F_POINTER(cval, fval, shape=(/ fptr(n + 1) - 1 /))
!   call C_F_POINTER(cx,   fx,   shape=(/ n, nrhs /))
!   call C_F_POINTER(crhs, frhs, shape=(/ n, nrhs /))

!   call check_backward_error(n, fptr, frow, fval, nrhs, fx, frhs)

! end subroutine spldlt_c_check_backward_error
