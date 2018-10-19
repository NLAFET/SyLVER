!> \file
!> \copyright 2018- The Science and Technology Facilities Council (STFC)
!> \author    Sebastien Cayrols

module spldlt_ciface
  use iso_c_binding
  use spral_ssids_ciface
  implicit none

  type, bind(C) :: spldlt_options_t
    type(spral_ssids_options) :: options
    logical(C_BOOL)   :: prune_tree
  end type spldlt_options_t

contains
  subroutine copy_spldlt_options_in(coptions, foptions, cindexed)
    use spldlt_datatypes_mod
    use spral_ssids_ciface
    implicit none

    type(spldlt_options_t), intent(in)    :: coptions
    type(spldlt_options),   intent(inout) :: foptions ! inherit some defaults!
    logical,                intent(out)   :: cindexed

    call copy_options_in(coptions%options, foptions%options, cindexed)
    foptions%prune_tree        = coptions%prune_tree

  end subroutine copy_spldlt_options_in

end module spldlt_ciface

!subroutine spldlt_c_default_options(coptions) &
!    bind(C, name="spldlt_default_options")
!  use spral_ssids_ciface
!  implicit none
!
!  type(spral_ssids_options), intent(out) :: coptions
!
!  type(ssids_options) :: default_options
!
!  coptions%array_base        = 0 ! C
!  coptions%print_level       = default_options%print_level
!  coptions%unit_diagnostics  = default_options%unit_diagnostics
!  coptions%unit_error        = default_options%unit_error
!  coptions%unit_warning      = default_options%unit_warning
!  coptions%ordering          = default_options%ordering
!  coptions%nemin             = default_options%nemin
!  coptions%ignore_numa       = default_options%ignore_numa
!  coptions%use_gpu           = default_options%use_gpu
!  coptions%min_gpu_work      = default_options%min_gpu_work
!  coptions%max_load_inbalance= default_options%max_load_inbalance
!  coptions%gpu_perf_coeff    = default_options%gpu_perf_coeff
!  coptions%scaling           = default_options%scaling
!  coptions%small_subtree_threshold = default_options%small_subtree_threshold
!  coptions%cpu_block_size    = default_options%cpu_block_size
!  coptions%action            = default_options%action
!  coptions%pivot_method      = default_options%pivot_method
!  coptions%small             = default_options%small
!  coptions%u                 = default_options%u
!end subroutine spldlt_c_default_options



subroutine spldlt_c_analyse(n, cptr, crow, cval, ncpu, cakeep, coptions, cinform) &
    bind(C, name="spldlt_analyse")
  use spldlt_ciface
  use spldlt_datatypes_mod
  use spldlt_analyse_mod
  implicit none

  integer(C_INT),   value :: n
  type(C_PTR),      value :: cptr
  type(C_PTR),      value :: crow
  type(C_PTR),      value :: cval
  integer(C_INT),   value :: ncpu
  type(C_PTR),              intent(inout) :: cakeep
  type(spldlt_options_t),   intent(in)    :: coptions
  type(spral_ssids_inform), intent(out)   :: cinform

  integer(C_LONG),          pointer :: fptr(:)
  integer(C_INT),           pointer :: frow(:)
  real(C_DOUBLE),           pointer :: fval(:)
  type(spldlt_akeep_type),  pointer :: fakeep
  type(spldlt_options)              :: foptions
  type(ssids_inform)                :: finform

  logical :: cindexed

  ! Copy options in first to find out whether we use Fortran or C indexing
  call copy_spldlt_options_in(coptions, foptions, cindexed)

  ! Translate arguments
  if (C_ASSOCIATED(cptr)) then
     call C_F_POINTER(cptr, fptr, shape=(/ n+1 /))
  else
     nullify(fptr)
  end if
  if (C_ASSOCIATED(crow)) then
     call C_F_POINTER(crow, frow, shape=(/ fptr(n+1)-1 /))
  else
     nullify(frow)
  end if
  if (C_ASSOCIATED(cval)) then
     call C_F_POINTER(cval, fval, shape=(/ fptr(n+1)-1 /))
  else
     nullify(fval)
  end if
  if (C_ASSOCIATED(cakeep)) then
     ! Reuse old pointer
     call C_F_POINTER(cakeep, fakeep)
  else
     ! Create new pointer
     allocate(fakeep)
     cakeep = C_LOC(fakeep)
  end if

  ! Call Fortran routine
   if (ASSOCIATED(fval)) then
      call spldlt_analyse(fakeep, n, fptr, frow, foptions%options, finform, ncpu, val=fval)
   else
      call spldlt_analyse(fakeep, n, fptr, frow, foptions%options, finform, ncpu)
   end if

  ! Copy arguments out
  call copy_inform_out(finform, cinform)
end subroutine spldlt_c_analyse



subroutine spldlt_c_factor(cposdef, val, cakeep, cfkeep, coptions, cinform) &
    bind(C, name="spldlt_factor")
  use spldlt_datatypes_mod
  use spldlt_analyse_mod
  use spldlt_factorize_mod
  use spldlt_ciface
  implicit none

  logical(C_BOOL),  value                 :: cposdef
  real(C_DOUBLE),           intent(in)    :: val(*)
  type(C_PTR),      value                 :: cakeep
  type(C_PTR),              intent(inout) :: cfkeep
  type(spldlt_options_t),   intent(in)    :: coptions
  type(spral_ssids_inform), intent(out)   :: cinform
  
  logical                           :: fposdef
  type(spldlt_akeep_type), pointer  :: fakeep
  type(spldlt_fkeep_type), pointer  :: ffkeep
  type(spldlt_options)              :: foptions
  type(ssids_inform)                :: finform

  logical :: cindexed

  ! Copy options in first to find out whether we use Fortran or C indexing
  call copy_spldlt_options_in(coptions, foptions, cindexed)

  ! Translate arguments
  fposdef = cposdef
  call C_F_POINTER(cakeep, fakeep) ! Pulled forward so we can use it
  if (C_ASSOCIATED(cfkeep)) then
     ! Reuse old pointer
     call C_F_POINTER(cfkeep, ffkeep)
  else
     ! Create new pointer
     allocate(ffkeep)
     cfkeep = C_LOC(ffkeep)
  end if

  ! Call Fortran routine
  !spldlt_factor(spldlt_akeep, spldlt_fkeep, posdef, val, options, inform)
    call spldlt_factor(fakeep, ffkeep, fposdef, val, foptions%options, finform)

  ! Copy arguments out
  call copy_inform_out(finform, cinform)
end subroutine spldlt_c_factor



subroutine spldlt_c_solve(job, nrhs, cx, ldx, cakeep, cfkeep, cinform) &
    bind(C, name="spldlt_solve")
  use spldlt_datatypes_mod
  use spldlt_analyse_mod
  use spldlt_factorize_mod
  use spldlt_ciface
  implicit none

  integer(C_INT),      value            :: job ! unused right now
  integer(C_INT),      value            :: nrhs
  type(C_PTR),         value            :: cx
  integer(C_INT),      value            :: ldx
  type(C_PTR),         value            :: cakeep
  type(C_PTR),         value            :: cfkeep
  type(spral_ssids_inform), intent(out) :: cinform

  real(C_DOUBLE),          pointer  :: fx(:,:)
  type(spldlt_akeep_type), pointer  :: fakeep
  type(spldlt_fkeep_type), pointer  :: ffkeep
  type(ssids_inform)                :: finform

  ! Translate arguments
  if (C_ASSOCIATED(cx)) then
     call C_F_POINTER(cx, fx, shape=(/ ldx,nrhs /))
  else
     nullify(fx)
  end if
  if (C_ASSOCIATED(cakeep)) then
     call C_F_POINTER(cakeep, fakeep)
  else
     nullify(fakeep)
  end if
  if (C_ASSOCIATED(cfkeep)) then
     call C_F_POINTER(cfkeep, ffkeep)
  else
     nullify(ffkeep)
  end if

  ! Call Fortran routine
  call ffkeep%solve(fakeep, nrhs, fx, ldx, finform)

  ! Copy arguments out
  call copy_inform_out(finform, cinform)
end subroutine spldlt_c_solve



integer(C_INT) function spldlt_c_free_akeep(cakeep) &
    bind(C, name="spldlt_free_akeep")
  use spldlt_ciface
  use spldlt_analyse_mod
  use spral_ssids, only: ssids_free
  implicit none
   
  type(C_PTR), intent(inout) :: cakeep

  type(spldlt_akeep_type), pointer :: fakeep

  if (.not. C_ASSOCIATED(cakeep)) then
     ! Nothing to free
     spldlt_c_free_akeep = 0
     return
  end if

  call C_F_POINTER(cakeep, fakeep)
  !TODO should be replaced by a spldlt_free subroutine, when it exists
  call ssids_free(fakeep%akeep, spldlt_c_free_akeep)
  if (allocated(fakeep%subtree_en)) then
    deallocate(fakeep%subtree_en)
  end if
  deallocate(fakeep)
  cakeep = C_NULL_PTR
end function spldlt_c_free_akeep



integer(C_INT) function spldlt_c_free_fkeep(cfkeep) &
    bind(C, name="spldlt_free_fkeep")
  use spldlt_ciface
  use spldlt_factorize_mod
  use spral_ssids, only: ssids_free
  implicit none
   
  type(C_PTR), intent(inout) :: cfkeep

  type(spldlt_fkeep_type), pointer :: ffkeep

  if (.not. C_ASSOCIATED(cfkeep)) then
     ! Nothing to free
     spldlt_c_free_fkeep = 0
     return
  end if

  call C_F_POINTER(cfkeep, ffkeep)
  call ssids_free(ffkeep%fkeep, spldlt_c_free_fkeep)
  deallocate(ffkeep)
  cfkeep = C_NULL_PTR
end function spldlt_c_free_fkeep
