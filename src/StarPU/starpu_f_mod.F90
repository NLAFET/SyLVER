module starpu_f_mod
  use iso_c_binding

  interface starpu_f_shutdown
     subroutine starpu_shutdown() bind(c)
     end subroutine starpu_shutdown
  end interface starpu_f_shutdown

  interface starpu_f_pause
     subroutine starpu_pause() bind(c)
     end subroutine starpu_pause
  end interface starpu_f_pause

  interface starpu_f_resume
     subroutine starpu_resume() bind(c)
     end subroutine starpu_resume
  end interface starpu_f_resume

  interface starpu_f_init
     function starpu_f_init_c( &
          ncpu, ngpu) bind(c)
       use iso_c_binding
       integer(c_int)        :: starpu_f_init_c
       integer(c_int), value :: ncpu 
       integer(c_int), value :: ngpu 
     end function starpu_f_init_c
     function starpu_init(conf) bind(c)
       use iso_c_binding
       integer(c_int)     :: starpu_f_init_c
       type(c_ptr), value :: conf
     end function starpu_init
  end interface starpu_f_init


  ! workers

  ! return the number of worker
  interface starpu_f_cpu_worker_get_count
     function starpu_cpu_worker_get_count() bind(c)
       use iso_c_binding
       integer(c_int) :: starpu_f_cpu_worker_get_count
     end function starpu_cpu_worker_get_count
  end interface starpu_f_cpu_worker_get_count
  
  ! return the number of workers
  interface starpu_f_worker_get_count
     function starpu_worker_get_count() bind(c)
       use iso_c_binding
       integer(c_int) :: starpu_worker_get_count
     end function starpu_worker_get_count
  end interface starpu_f_worker_get_count
  
  ! return the id of the current worker
  interface starpu_f_worker_get_id
     function starpu_worker_get_id() bind(c)
       use iso_c_binding
       integer(c_int) :: starpu_worker_get_id
     end function starpu_worker_get_id
  end interface starpu_f_worker_get_id

  ! data interfaces 

  interface starpu_f_void_data_register
     subroutine starpu_void_data_register(handle) bind(c)
       use iso_c_binding
       type(c_ptr)    :: handle
     end subroutine starpu_void_data_register
  end interface starpu_f_void_data_register

  interface starpu_f_data_unregister
     subroutine starpu_data_unregister(handle) bind(c)
       use iso_c_binding
       type(c_ptr), value            :: handle
     end subroutine starpu_data_unregister
  end interface starpu_f_data_unregister
  
  interface starpu_f_data_unregister_no_coherency
     subroutine starpu_data_unregister_no_coherency(handle) bind(c)
       use iso_c_binding
       type(c_ptr), value            :: handle
     end subroutine starpu_data_unregister_no_coherency
  end interface starpu_f_data_unregister_no_coherency
  

  ! tasks

  interface starpu_f_task_wait_for_all
     subroutine starpu_task_wait_for_all() bind(c)
     end subroutine starpu_task_wait_for_all
  end interface starpu_f_task_wait_for_all

  interface starpu_f_get_buffer
     subroutine starpu_f_get_buffer(buffers, num, a, m, n, lda) bind(C)
       use iso_c_binding
       type(c_ptr), value    :: a
       integer(c_int), value :: num
       type(c_ptr), value    :: m, n, lda, buffers
     end subroutine starpu_f_get_buffer

     subroutine starpu_f_vector_get_buffer(buffers, num, a, m) bind(C)
       use iso_c_binding
       type(c_ptr), value    :: a
       integer(c_int), value :: num
       type(c_ptr), value    :: m, buffers
     end subroutine starpu_f_vector_get_buffer
  end interface starpu_f_get_buffer



  interface starpu_f_matrix_data_register
     module procedure starpu_f_1dsmatrix_data_register, starpu_f_2dsmatrix_data_register
     module procedure starpu_f_1ddmatrix_data_register, starpu_f_2ddmatrix_data_register
     module procedure starpu_f_1dcmatrix_data_register, starpu_f_2dcmatrix_data_register
     module procedure starpu_f_1dzmatrix_data_register, starpu_f_2dzmatrix_data_register
     subroutine starpu_matrix_data_register(handle, host, a, ld, nx, ny, sizeof) bind(c)
       use iso_c_binding
       type(c_ptr)              :: handle
       type(c_ptr), value       :: a
       integer(c_int), value    :: host, ld, nx, ny
       integer(c_size_t), value :: sizeof
     end subroutine starpu_matrix_data_register
  end interface starpu_f_matrix_data_register

  interface starpu_f_vector_data_register
     subroutine starpu_vector_data_register(handle, host, a, nx, sizeof) bind(c)
       use iso_c_binding
       type(c_ptr)              :: handle
       type(c_ptr), value       :: a
       integer(c_int), value    :: host, nx
       integer(c_size_t), value :: sizeof
     end subroutine starpu_vector_data_register
  end interface starpu_f_vector_data_register

  ! interface
  ! end interface

  interface starpu_f_data_unregister_submit
     subroutine starpu_data_unregister_submit(handle) bind(c)
       use iso_c_binding
       type(c_ptr), value :: handle
     end subroutine starpu_data_unregister_submit
  end interface starpu_f_data_unregister_submit
  

  
! #if defined(have_fxt)
  interface starpu_f_fxt_start_profiling
     subroutine starpu_fxt_start_profiling() bind(c)
     end subroutine starpu_fxt_start_profiling
  end interface starpu_f_fxt_start_profiling
  interface starpu_f_fxt_stop_profiling
     subroutine starpu_fxt_stop_profiling() bind(c)
     end subroutine starpu_fxt_stop_profiling
  end interface starpu_f_fxt_stop_profiling
! #endif

  interface
     subroutine starpu_f_alloc_handle(p) bind(c)
       use iso_c_binding
       type(c_ptr) :: p
     end subroutine starpu_f_alloc_handle
  end interface

  interface starpu_f_cublas_shutdown
     subroutine starpu_cublas_shutdown() bind(c)
     end subroutine starpu_cublas_shutdown
  end interface starpu_f_cublas_shutdown
  
contains

  subroutine starpu_f_1dsmatrix_data_register(handle, host, a, ld, m, n)
    implicit none
    type(c_ptr)       :: handle
    ! real, target      :: a(:)
    real, allocatable, target      :: a(:)
    integer           :: host
    integer, optional :: ld, m, n
    integer           :: ild, im, in

    if(present(ld)) then
       ild = ld
    else
       ild = size(a,1)
    end if

    if(present(m)) then
       im = m
    else
       im = size(a,1)
    end if

    if(present(n)) then
       in = n
    else
       in = 1
    end if

    ! 4 is hardcoded. should make something better and more portable 
    call starpu_matrix_data_register(handle, host, c_loc(a(1)), ild, im, in, int(4,kind=c_size_t))

    return
  end subroutine starpu_f_1dsmatrix_data_register


  subroutine starpu_f_2dsmatrix_data_register(handle, host, a, ld, m, n)
    implicit none
    type(c_ptr)       :: handle
    real, allocatable, target      :: a(:,:)
    ! real, target      :: a(:,:)
    integer           :: host
    integer, optional :: ld, m, n
    integer           :: ild, im, in

    if(present(ld)) then
       ild = ld
    else
       ild = size(a,1)
    end if

    if(present(m)) then
       im = m
    else
       im = size(a,1)
    end if

    if(present(n)) then
       in = n
    else
       in = size(a,2)
    end if

    ! 4 is hardcoded. should make something better and more portable 
    call starpu_matrix_data_register(handle, host, c_loc(a(1,1)), ild, im, in, int(4,kind=c_size_t))

    return
  end subroutine starpu_f_2dsmatrix_data_register

  subroutine starpu_f_1ddmatrix_data_register(handle, host, a, ld, m, n)
    implicit none
    type(c_ptr)              :: handle
    ! real(kind(1.d0)), target :: a(:)
    real(kind(1.d0)), allocatable, target :: a(:)
    integer                  :: host
    integer, optional        :: ld, m, n
    integer                  :: ild, im, in

    if(present(ld)) then
       ild = ld
    else
       ild = size(a,1)
    end if

    if(present(m)) then
       im = m
    else
       im = size(a,1)
    end if

    if(present(n)) then
       in = n
    else
       in = 1
    end if

    ! 4 is hardcoded. should make something better and more portable 
    call starpu_matrix_data_register(handle, host, c_loc(a(1)), ild, im, in, int(8,kind=c_size_t))

    return
  end subroutine starpu_f_1ddmatrix_data_register


  subroutine starpu_f_2ddmatrix_data_register(handle, host, a, ld, m, n)
    implicit none
    type(c_ptr)              :: handle
    ! real(kind(1.d0)), target :: a(:,:)
    real(kind(1.d0)), allocatable, target :: a(:,:)
    integer                  :: host
    integer, optional        :: ld, m, n
    integer                  :: ild, im, in

    if(present(ld)) then
       ild = ld
    else
       ild = size(a,1)
    end if

    if(present(m)) then
       im = m
    else
       im = size(a,1)
    end if

    if(present(n)) then
       in = n
    else
       in = size(a,2)
    end if

    ! 4 is hardcoded. should make something better and more portable 
    call starpu_matrix_data_register(handle, host, c_loc(a(1,1)), ild, im, in, int(8,kind=c_size_t))

    return
  end subroutine starpu_f_2ddmatrix_data_register

  subroutine starpu_f_1dcmatrix_data_register(handle, host, a, ld, m, n)
    implicit none
    type(c_ptr)       :: handle
    ! complex, target   :: a(:)
    complex, allocatable, target   :: a(:)
    integer           :: host
    integer, optional :: ld, m, n
    integer           :: ild, im, in

    if(present(ld)) then
       ild = ld
    else
       ild = size(a,1)
    end if

    if(present(m)) then
       im = m
    else
       im = size(a,1)
    end if

    if(present(n)) then
       in = n
    else
       in = 1
    end if

    ! 4 is hardcoded. should make something better and more portable 
    call starpu_matrix_data_register(handle, host, c_loc(a(1)), ild, im, in, int(8,kind=c_size_t))

    return
  end subroutine starpu_f_1dcmatrix_data_register



  subroutine starpu_f_2dcmatrix_data_register(handle, host, a, ld, m, n)
    implicit none
    type(c_ptr)       :: handle
    complex, allocatable, target   :: a(:,:)
    ! complex, target   :: a(:,:)
    integer           :: host
    integer, optional :: ld, m, n
    integer           :: ild, im, in

    if(present(ld)) then
       ild = ld
    else
       ild = size(a,1)
    end if

    if(present(m)) then
       im = m
    else
       im = size(a,1)
    end if

    if(present(n)) then
       in = n
    else
       in = size(a,2)
    end if

    ! 4 is hardcoded. should make something better and more portable 
    call starpu_matrix_data_register(handle, host, c_loc(a(1,1)), ild, im, in, int(8,kind=c_size_t))

    return
  end subroutine starpu_f_2dcmatrix_data_register

  subroutine starpu_f_1dzmatrix_data_register(handle, host, a, ld, m, n)
    implicit none
    type(c_ptr)                 :: handle
    complex(kind(1.d0)), allocatable, target :: a(:)
    ! complex(kind(1.d0)), target :: a(:)
    integer                     :: host
    integer, optional           :: ld, m, n
    integer                     :: ild, im, in

    if(present(ld)) then
       ild = ld
    else
       ild = size(a,1)
    end if

    if(present(m)) then
       im = m
    else
       im = size(a,1)
    end if

    if(present(n)) then
       in = n
    else
       in = 1
    end if

    ! 4 is hardcoded. should make something better and more portable 
    call starpu_matrix_data_register(handle, host, c_loc(a(1)), ild, im, in, int(16,kind=c_size_t))

    return
  end subroutine starpu_f_1dzmatrix_data_register


  subroutine starpu_f_2dzmatrix_data_register(handle, host, a, ld, m, n)
    implicit none
    type(c_ptr)                 :: handle
    complex(kind(1.d0)), allocatable, target :: a(:,:)
    ! complex(kind(1.d0)), target :: a(:,:)
    integer                     :: host
    integer, optional           :: ld, m, n
    integer                     :: ild, im, in

    if(present(ld)) then
       ild = ld
    else
       ild = size(a,1)
    end if

    if(present(m)) then
       im = m
    else
       im = size(a,1)
    end if

    if(present(n)) then
       in = n
    else
       in = size(a,2)
    end if

    ! 4 is hardcoded. should make something better and more portable 
    call starpu_matrix_data_register(handle, host, c_loc(a(1,1)), ild, im, in, int(16,kind=c_size_t))

    return
  end subroutine starpu_f_2dzmatrix_data_register

end module starpu_f_mod
