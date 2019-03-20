!> @file
!> @copyright 2016- The Science and Technology Facilities Council (STFC)
!> @author    Florent Lopez
module sylver_datatypes_mod
  use spral_ssids_datatypes
  implicit none

  integer, parameter, public :: sylver_nemin_default = 32 ! Node amalgamation

  integer, parameter, public :: sylver_nb_default = 256 ! Block size

  ! Success flag
  integer, parameter, public :: SYLVER_SUCCESS                 = 0

  ! Error flags
  integer, parameter, public :: SYLVER_ERROR_CALL_SEQUENCE     = -1
  integer, parameter, public :: SYLVER_ERROR_A_N_OOR           = -2
  integer, parameter, public :: SYLVER_ERROR_A_PTR             = -3
  integer, parameter, public :: SYLVER_ERROR_A_ALL_OOR         = -4
  integer, parameter, public :: SYLVER_ERROR_SINGULAR          = -5
  integer, parameter, public :: SYLVER_ERROR_NOT_POS_DEF       = -6
  integer, parameter, public :: SYLVER_ERROR_PTR_ROW           = -7
  integer, parameter, public :: SYLVER_ERROR_ORDER             = -8
  integer, parameter, public :: SYLVER_ERROR_VAL               = -9
  integer, parameter, public :: SYLVER_ERROR_X_SIZE            = -10
  integer, parameter, public :: SYLVER_ERROR_JOB_OOR           = -11
  integer, parameter, public :: SYLVER_ERROR_NOT_LLT           = -13
  integer, parameter, public :: SYLVER_ERROR_NOT_LDLT          = -14
  integer, parameter, public :: SYLVER_ERROR_NO_SAVED_SCALING  = -15
  integer, parameter, public :: SYLVER_ERROR_ALLOCATION        = -50
  integer, parameter, public :: SYLVER_ERROR_CUDA_UNKNOWN      = -51
  integer, parameter, public :: SYLVER_ERROR_CUBLAS_UNKNOWN    = -52
  integer, parameter, public :: SYLVER_ERROR_UNIMPLEMENTED     = -98
  integer, parameter, public :: SYLVER_ERROR_UNKNOWN           = -99

  ! warning flags
  integer, parameter, public :: SYLVER_WARNING_IDX_OOR          = 1
  integer, parameter, public :: SYLVER_WARNING_DUP_IDX          = 2
  integer, parameter, public :: SYLVER_WARNING_DUP_AND_OOR      = 3
  integer, parameter, public :: SYLVER_WARNING_MISSING_DIAGONAL = 4
  integer, parameter, public :: SYLVER_WARNING_MISS_DIAG_OORDUP = 5
  integer, parameter, public :: SYLVER_WARNING_ANAL_SINGULAR    = 6
  integer, parameter, public :: SYLVER_WARNING_FACT_SINGULAR    = 7
  integer, parameter, public :: SYLVER_WARNING_MATCH_ORD_NO_SCALE=8

  ! solve job values
  integer, parameter, public :: SYLVER_SOLVE_JOB_ALL     = 0 ! PLD(PL)^TX = B
  integer, parameter, public :: SYLVER_SOLVE_JOB_FWD     = 1 ! PLX = B
  integer, parameter, public :: SYLVER_SOLVE_JOB_DIAG    = 2 ! DX = B (indef)
  integer, parameter, public :: SYLVER_SOLVE_JOB_BWD     = 3 ! (PL)^TX = B
  integer, parameter, public :: SYLVER_SOLVE_JOB_DIAG_BWD= 4 ! D(PL)^TX = B (indef)

  ! Pivot method in the factorization phase
  integer, parameter, public :: SYLVER_PIVOT_METHOD_APP_AGGRESIVE = 1
  integer, parameter, public :: SYLVER_PIVOT_METHOD_APP_BLOCK     = 2
  integer, parameter, public :: SYLVER_PIVOT_METHOD_TPP           = 3

  ! Failed pivot method in the factorization phase when using APP
  ! pivot method
  integer, parameter, public :: SYLVER_FAILED_PIVOT_METHOD_TPP    = 1
  integer, parameter, public :: SYLVER_FAILED_PIVOT_METHOD_PASS   = 2

  !
  ! Data type for control parameters in SpLDLT
  !

  type spldlt_options
     type(ssids_options) :: super
     logical :: prune_tree = .true.
  end type spldlt_options
  
  !
  ! Data type for control parameters in SpLU
  !
  
  type sylver_options

     !
     ! Printing options
     !
     integer :: print_level = 0 ! Controls diagnostic printing.
     ! Possible values are:
     !  < 0: no printing.
     !  0: error and warning messages only.
     !  1: as 0 plus basic diagnostic printing.
     !  > 1: as 1 plus some more detailed diagnostic messages.
     !  > 9999: debug (absolutely everything - really don't use this)
     integer :: unit_diagnostics = 6 ! unit number for diagnostic printing.
     ! Printing is suppressed if unit_diagnostics  <  0.
     integer :: unit_error = 6 ! unit number for error messages.
     ! Printing is suppressed if unit_error  <  0.
     integer :: unit_warning = 6 ! unit number for warning messages.
     ! Printing is suppressed if unit_warning  <  0.

     !
     ! Options used in spldlt_analyse() and splu_analyse() 
     !
     integer :: ordering = 1 ! controls choice of ordering
       ! 0 Order must be supplied by user
       ! 1 METIS ordering with default settings is used.
       ! 2 Matching with METIS on compressed matrix.
     integer :: nemin = sylver_nemin_default ! Amalgamation parameter.
     logical :: prune_tree = .true. ! Partition tree using pruning 
     
     !
     ! Options used by splu_factor()
     !
     integer :: scaling = 0 ! controls use of scaling. 
       !  <=0: user supplied (or no) scaling
       !    1: Matching-based scaling by Hungarian Algorithm (MC64-like)
       !    2: Matching-based scaling by Auction Algorithm
       !    3: Scaling generated during analyse phase for matching-based order
       !  >=4: Norm equilibriation algorithm (MC77-like)

     !
     ! Options used by splu_factor() for controlling pivoting
     !
     integer :: pivot_method = SYLVER_PIVOT_METHOD_APP_BLOCK
       ! Type of pivoting to use on CPU side:
       ! 0 - A posteori pivoting, roll back entire front on pivot failure
       ! 1 - A posteori pivoting, roll back on block column level for failure
       ! 2 - Traditional threshold partial pivoting (serial, inefficient!)
     real(wp) :: small = 1e-20_wp ! Minimum pivot size (absolute value of a
       ! pivot must be of size at least small to be accepted).
     real(wp) :: u = 0.01 ! Threshold used to decide whether a pivot
       ! is accepted or not.

     !
     ! CPU-specific
     !     
     integer :: nb = sylver_nb_default! Block size 

     !
     ! GPU-specific
     !     
     logical :: use_gpu = .true. ! Whether GPU should be used
     real :: gpu_perf_coeff = 1.0 ! How many times better is a GPU than a
       ! single NUMA region's worth of processors

     !
     ! Undocumented
     !
     real(wp) :: multiplier = 1.1 ! size to multiply expected memory size by
       ! when doing initial memory allocation to allow for delays.
     integer :: failed_pivot_method = SYLVER_FAILED_PIVOT_METHOD_TPP
       ! What to do with failed pivots:
       !     <= 1  Attempt to eliminate with TPP pass
       !     >= 2  Pass straight to parent
     
  end type sylver_options
end module sylver_datatypes_mod
