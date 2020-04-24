!> \file
!> \copyright 2018- The Science and Technology Facilities Council (STFC)
!> \author    Sebastien Cayrols

module spldlt_ciface
  use iso_c_binding
  use spral_ssids_ciface
  use spral_ssids_datatypes
  implicit none

  type, bind(C) :: sylver_options_t
     !
     ! Printing options
     !
     integer(c_int) :: print_level
     integer(c_int) :: unit_diagnostics 
     integer(c_int) :: unit_error
     integer(c_int) :: unit_warning
     !
     ! Options used in spldlt_analyse() and splu_analyse() 
     !
     integer(c_int) :: ordering
  end type sylver_options_t

  ! interface check_backward_error
  !   module procedure check_backward_error_one
  !   module procedure check_backward_error_multi
  ! end interface check_backward_error

contains

  ! subroutine copy_spldlt_options_in(coptions, foptions, cindexed)
  !   use sylver_datatypes_mod
  !   use spral_ssids_ciface
  !   implicit none

  !   type(spldlt_options_t), intent(in)    :: coptions
  !   type(spldlt_options),   intent(inout) :: foptions ! inherit some defaults!
  !   logical,                intent(out)   :: cindexed

  !   call copy_options_in(coptions%options, foptions%options, cindexed)
  !   foptions%prune_tree        = coptions%prune_tree

  ! end subroutine copy_spldlt_options_in



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

end module spldlt_ciface

! subroutine spldlt_c_default_options(coptions) &
!     bind(C, name="spldlt_default_options")
!   use spldlt_ciface
!   use sylver_datatypes_mod
!   implicit none

!   type(spldlt_options_t), intent(out) :: coptions

!   type(ssids_options)   :: default_options
!   type(spldlt_options)  :: default_spldlt_options

!   coptions%options%array_base        = 0 ! C
!   coptions%options%print_level       = default_options%print_level
!   coptions%options%unit_diagnostics  = default_options%unit_diagnostics
!   coptions%options%unit_error        = default_options%unit_error
!   coptions%options%unit_warning      = default_options%unit_warning
!   coptions%options%ordering          = default_options%ordering
!   coptions%options%nemin             = default_options%nemin
!   coptions%options%ignore_numa       = default_options%ignore_numa
!   coptions%options%use_gpu           = default_options%use_gpu
!   coptions%options%min_gpu_work      = default_options%min_gpu_work
!   coptions%options%max_load_inbalance= default_options%max_load_inbalance
!   coptions%options%gpu_perf_coeff    = default_options%gpu_perf_coeff
!   coptions%options%scaling           = default_options%scaling
!   coptions%options%small_subtree_threshold = default_options%small_subtree_threshold
!   coptions%options%cpu_block_size    = default_options%cpu_block_size
!   coptions%options%action            = default_options%action
!   coptions%options%pivot_method      = default_options%pivot_method
!   coptions%options%small             = default_options%small
!   coptions%options%u                 = default_options%u
!   coptions%prune_tree                = default_spldlt_options%prune_tree
! end subroutine spldlt_c_default_options



! subroutine spldlt_c_init(ncpu, ngpu) &
!     bind(C, name="spldlt_init")
!   use spldlt_ciface
!   use spldlt_mod
!   implicit none

!   integer(C_INT),   value :: ncpu
!   integer(C_INT),   value :: ngpu

!   call spldlt_init(ncpu, ngpu)

! end subroutine spldlt_c_init



! subroutine spldlt_c_finalize() &
!     bind(C, name="spldlt_finalize")
!   use spldlt_ciface
!   use spldlt_mod
!   implicit none

!   call spldlt_finalize()

! end subroutine spldlt_c_finalize



! subroutine spldlt_c_analyse(n, cptr, crow, cval, ncpu, cakeep, coptions, &
!     cinform) bind(C, name="spldlt_analyse")
!   use spldlt_ciface
!   use sylver_datatypes_mod
!   use spldlt_analyse_mod
!   implicit none

!   integer(C_INT),   value :: n
!   type(C_PTR),      value :: cptr
!   type(C_PTR),      value :: crow
!   type(C_PTR),      value :: cval
!   integer(C_INT),   value :: ncpu
!   type(C_PTR),              intent(inout) :: cakeep
!   type(spldlt_options_t),   intent(in)    :: coptions
!   type(spral_ssids_inform), intent(out)   :: cinform

!   integer(C_LONG),          pointer :: fptr(:)
!   integer(C_INT),           pointer :: frow(:)
!   real(C_DOUBLE),           pointer :: fval(:)
!   type(spldlt_akeep_type),  pointer :: fakeep
!   type(spldlt_options)              :: foptions
!   type(ssids_inform)                :: finform
!   integer                           :: st
!   logical :: cindexed

!   ! Copy options in first to find out whether we use Fortran or C indexing
!   call copy_spldlt_options_in(coptions, foptions, cindexed)

!   ! Translate arguments
!   if (C_ASSOCIATED(cptr)) then
!     call C_F_POINTER(cptr, fptr, shape=(/ n+1 /))
!   else
!     print *, "Error, cptr is not associated"
!     nullify(fptr)
!   end if

!   if (C_ASSOCIATED(crow)) then
!     call C_F_POINTER(crow, frow, shape=(/ fptr(n+1)-1 /))
!   else
!     print *, "Error, crow is not associated"
!     nullify(frow)
!   end if

!   if (C_ASSOCIATED(cval)) then
!     call C_F_POINTER(cval, fval, shape=(/ fptr(n+1)-1 /))
!   else
!     print *, "Warning, cval is not associated"
!     nullify(fval)
!   end if
!   if (C_ASSOCIATED(cakeep)) then
!     ! Reuse old pointer
!     call C_F_POINTER(cakeep, fakeep)
!   else
!     ! Create new pointer
!     allocate(fakeep, stat=st)
!     cakeep = C_LOC(fakeep)
!   end if

!  !print *, "Call analyse"
!   ! Call Fortran routine
!   if (ASSOCIATED(fval)) then
!     call spldlt_analyse(fakeep, n, fptr, frow, foptions%options, finform, ncpu, val=fval)
!   else
!     call spldlt_analyse(fakeep, n, fptr, frow, foptions%options, finform, ncpu)
!   end if

!   ! Copy arguments out
!   call copy_inform_out(finform, cinform)
! end subroutine spldlt_c_analyse



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



! subroutine spldlt_c_factor(cposdef, val, cakeep, cfkeep, coptions, cinform) &
!     bind(C, name="spldlt_factor")
!   use sylver_datatypes_mod
!   use spldlt_analyse_mod
!   use spldlt_factorize_mod
!   use spldlt_ciface
!   implicit none

!   logical(C_BOOL),  value                 :: cposdef
!   real(C_DOUBLE),           intent(in)    :: val(*)
!   type(C_PTR),      value                 :: cakeep
!   type(C_PTR),              intent(inout) :: cfkeep
!   type(spldlt_options_t),   intent(in)    :: coptions
!   type(spral_ssids_inform), intent(out)   :: cinform
  
!   logical                           :: fposdef
!   type(spldlt_akeep_type), pointer  :: fakeep
!   type(spldlt_fkeep_type), pointer  :: ffkeep
!   type(spldlt_options)              :: foptions
!   type(ssids_inform)                :: finform

!   logical :: cindexed

!   ! Copy options in first to find out whether we use Fortran or C indexing
!   call copy_spldlt_options_in(coptions, foptions, cindexed)

!   ! Translate arguments
!   fposdef = cposdef
!   call C_F_POINTER(cakeep, fakeep) ! Pulled forward so we can use it
!   if (C_ASSOCIATED(cfkeep)) then
!      ! Reuse old pointer
!      call C_F_POINTER(cfkeep, ffkeep)
!   else
!      ! Create new pointer
!      allocate(ffkeep)
!      cfkeep = C_LOC(ffkeep)
!   end if

!  !print *, "Call factor"
!   ! Call Fortran routine
!   !spldlt_factor(spldlt_akeep, spldlt_fkeep, posdef, val, options, inform)
!     call spldlt_factor(fakeep, ffkeep, fposdef, val, foptions%options, finform)

!   ! Copy arguments out
!   call copy_inform_out(finform, cinform)
! end subroutine spldlt_c_factor



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
