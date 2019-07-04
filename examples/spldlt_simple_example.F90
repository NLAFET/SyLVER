program spldlt_example
  use sylver_mod
  implicit none

  ! Derived types
  type (spldlt_akeep_type)   :: akeep
  type (spldlt_fkeep_type)   :: fkeep
  type (sylver_options) :: options
  type (sylver_inform)  :: inform

  ! Parameters
  !integer, parameter :: long = selected_int_kind(16)
  !integer, parameter :: wp = kind(0.0d0)

  ! Matrix data
  logical :: posdef
  integer :: n, row(9)
  integer(long) :: ptr(6)
  real(wp) :: val(9)

  ! Other variables
  integer :: ncpu, ngpu
  integer :: nrhs
  real(wp) :: x(5)
  
  ! Data for matrix:
  ! ( 2  1         )
  ! ( 1  4  1    1 )
  ! (    1  3  2   )
  ! (       2 -1   )
  ! (    1       2 )
  posdef = .false.
  n = 5
  ptr(1:n+1)        = (/ 1,        3,             6,        8,    9,  10 /)
  row(1:ptr(n+1)-1) = (/ 1,   2,   2,   3,   5,   3,   4,   4,    5   /)
  val(1:ptr(n+1)-1) = (/ 2.0, 1.0, 4.0, 1.0, 1.0, 3.0, 2.0, -1.0, 2.0 /)

  ! The right-hand side with solution (1.0, 2.0, 3.0, 4.0, 5.0)
  nrhs = 1
  x(1:n) = (/ 4.0, 17.0, 19.0, 2.0, 12.0 /)

  ncpu = 8
  ngpu = 0

  call sylver_init(ncpu, ngpu)

  ! Perform analyse and factorize
  call spldlt_analyse(akeep, n, ptr, row, options, inform, ncpu=ncpu, ngpu=ngpu)
  if(inform%flag<0) go to 100

  call spldlt_factorize(akeep, fkeep, posdef, val, options, inform)
  if(inform%flag<0) go to 100

  call spldlt_solve(akeep, fkeep, nrhs, x, n, options, inform)
  if(inform%flag<0) go to 100
  write(*,'(a,/,(3es18.10))') ' The computed solution is:', x(1:n)
  
  call sylver_finalize()

100 continue
  call spldlt_akeep_free(akeep)
  call spldlt_fkeep_free(fkeep)
   
end program spldlt_example
