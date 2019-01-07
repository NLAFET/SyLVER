!> \file
!> \copyright 2016- The Science and Technology Facilities Council (STFC)
!> \author    Florent Lopez
module sylver_inform_mod
  use spral_cuda, only : cudaGetErrorString
  use spral_scaling, only : auction_inform
  use spldlt_datatypes_mod
  implicit none

  !
  ! Data type for information returned by code
  !
  type sylver_inform
     integer :: flag = SYLVER_SUCCESS ! Takes one of the enumerated flag values:
     ! SSIDS_SUCCESS
     ! SSIDS_ERROR_XXX
     ! SSIDS_WARNING_XXX
     integer :: matrix_dup = 0 ! Number of duplicated entries.
     integer :: matrix_missing_diag = 0 ! Number of missing diag. entries
     integer :: matrix_outrange = 0 ! Number of out-of-range entries.
     integer :: matrix_rank = 0 ! Rank of matrix (anal=structral, fact=actual)
     integer :: maxdepth = 0 ! Maximum depth of tree
     integer :: maxfront = 0 ! Maximum front size
     integer :: num_delay = 0 ! Number of delayed variables
     integer(long) :: num_factor = 0_long ! Number of entries in factors
     integer(long) :: num_flops = 0_long ! Number of floating point operations
     integer :: num_neg = 0 ! Number of negative pivots
     integer :: num_sup = 0 ! Number of supernodes
     integer :: num_two = 0 ! Number of 2x2 pivots used by factorization
     integer :: stat = 0 ! stat parameter
     type(auction_inform) :: auction
     integer :: cuda_error = 0
     integer :: cublas_error = 0

     ! Undocumented FIXME: should we document them?
     integer :: not_first_pass = 0
     integer :: not_second_pass = 0
     integer :: nparts = 0
     integer(long) :: cpu_flops = 0
     integer(long) :: gpu_flops = 0
   contains
     procedure :: flag_to_character
     procedure :: print_flag
  end type sylver_inform

contains

  !
  ! Returns a string representation
  ! Member function inform%flagToCharacter
  !
  function flag_to_character(this) result(msg)
    implicit none
    class(sylver_inform), intent(in) :: this
    character(len=200) :: msg ! return value

    select case(this%flag)
       !
       ! Success
       !
    case(SYLVER_SUCCESS)
       msg = 'Success'
       !
       ! Errors
       !
    case(SYLVER_ERROR_CALL_SEQUENCE)
       msg = 'Error in sequence of calls.'
    case(SYLVER_ERROR_A_N_OOR)
       msg = 'n or ne is out of range (or has changed)'
    case(SYLVER_ERROR_A_PTR)
       msg = 'Error in ptr'
    case(SYLVER_ERROR_A_ALL_OOR)
       msg = 'All entries in a column out-of-range (ssids_analyse) &
            &or all entries out-of-range (ssids_analyse_coord)'
    case(SYLVER_ERROR_SINGULAR)
       msg = 'Matrix found to be singular'
    case(SYLVER_ERROR_NOT_POS_DEF)
       msg = 'Matrix is not positive-definite'
    case(SYLVER_ERROR_PTR_ROW)
       msg = 'ptr and row should be present'
    case(SYLVER_ERROR_ORDER)
       msg = 'Either control%ordering out of range or error in user-supplied  &
            &elimination order'
    case(SYLVER_ERROR_X_SIZE)
       msg = 'Error in size of x or nrhs'
    case(SYLVER_ERROR_JOB_OOR)
       msg = 'job out of range'
    case(SYLVER_ERROR_NOT_LLT)
       msg = 'Not a LL^T factorization of a positive-definite matrix'
    case(SYLVER_ERROR_NOT_LDLT)
       msg = 'Not a LDL^T factorization of an indefinite matrix'
    case(SYLVER_ERROR_ALLOCATION)
       write (msg,'(a,i6)') 'Allocation error. stat parameter = ', this%stat
    case(SYLVER_ERROR_VAL)
       msg = 'Optional argument val not present when expected'
    case(SYLVER_ERROR_NO_SAVED_SCALING)
       msg = 'Requested use of scaling from matching-based &
            &ordering but matching-based ordering not used'
    case(SYLVER_ERROR_UNIMPLEMENTED)
       msg = 'Functionality not yet implemented'
    case(SYLVER_ERROR_CUDA_UNKNOWN)
       write(msg,'(2a)') ' Unhandled CUDA error: ', &
            trim(cudaGetErrorString(this%cuda_error))
    case(SYLVER_ERROR_CUBLAS_UNKNOWN)
       msg = 'Unhandled CUBLAS error:'

       !
       ! Warnings
       !
    case(SYLVER_WARNING_IDX_OOR)
       msg = 'out-of-range indices detected'
    case(SYLVER_WARNING_DUP_IDX)
       msg = 'duplicate entries detected'
    case(SYLVER_WARNING_DUP_AND_OOR)
       msg = 'out-of-range indices detected and duplicate entries detected'
    case(SYLVER_WARNING_MISSING_DIAGONAL)
       msg = 'one or more diagonal entries is missing'
    case(SYLVER_WARNING_MISS_DIAG_OORDUP)
       msg = 'one or more diagonal entries is missing and out-of-range and/or &
            &duplicate entries detected'
    case(SYLVER_WARNING_ANAL_SINGULAR)
       msg = 'Matrix found to be structually singular'
    case(SYLVER_WARNING_FACT_SINGULAR)
       msg = 'Matrix found to be singular'
    case(SYLVER_WARNING_MATCH_ORD_NO_SCALE)
       msg = 'Matching-based ordering used but associated scaling ignored'
    case default
       msg = 'SYLVER Internal Error'
    end select
  end function flag_to_character

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> @brief Print out warning or error if flag is non-zero
  !> @param this Instance variable.
  !> @param options Options to be used for printing
  !> @param context Name of routine to report error from
  subroutine print_flag(this, options, context)
    implicit none
    class(sylver_inform), intent(in) :: this
    type(sylver_options), intent(in) :: options
    character (len=*), intent(in) :: context

    character(len=200) :: msg

    if (this%flag .eq. SYLVER_SUCCESS) return ! Nothing to print
    if (options%print_level .lt. 0) return ! No printing
    if (this%flag .gt. SYLVER_SUCCESS) then
       ! Warning
       if (options%unit_warning .lt. 0) return ! printing supressed
       write (options%unit_warning,'(/3a,i3)') ' Warning from ', &
            trim(context), '. Warning flag = ', this%flag
       msg = this%flag_to_character()
       write (options%unit_warning, '(a)') msg
    else
       if (options%unit_error .lt. 0) return ! printing supressed
       write (options%unit_error,'(/3a,i3)') ' Error return from ', &
            trim(context), '. Error flag = ', this%flag
       msg = this%flag_to_character()
       write (options%unit_error, '(a)') msg
    end if
  end subroutine print_flag

end module sylver_inform_mod
