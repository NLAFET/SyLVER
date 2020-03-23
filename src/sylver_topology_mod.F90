module sylver_topology_mod
  use, intrinsic :: iso_c_binding
  implicit none

  !> Fortran interoperable definition of sylver::topology::NumaRegion
  type, bind(C) :: c_numa_region
     integer(C_INT) :: nproc
     integer(C_INT) :: ngpu
     type(C_PTR) :: gpus
  end type c_numa_region

  interface
     !> Interface to sylver_topology_create()
     subroutine sylver_topology_create_c(nregion, regions) bind(C)
       use, intrinsic :: iso_c_binding
       implicit none
       integer(C_INT), intent(out) :: nregion
       type(C_PTR), intent(out) :: regions
     end subroutine sylver_topology_create_c
  end interface

contains

  ! Create flat topology with `ncpu` CPU cores and `ngpu` GPU devices
  !
  subroutine sylver_topology_create_flat(ncpu, ngpu, regions)
    use spral_hw_topology, only : numa_region
    implicit none
    ! Number of CPU wokers involved the execution
    integer, intent(in) :: ncpu
    ! Number of CUDA wokers involved the execution
    integer, intent(in) :: ngpu
    ! Machine topology
    type(numa_region), dimension(:), allocatable, intent(out) :: regions

    integer :: i
    integer :: st ! Error management

    ! Make sure akeep%topology array is not allocated
    if (allocated(regions)) deallocate(regions, stat=st)

    ! Create flat topology
    allocate(regions(1), stat=st)
    regions(1)%nproc = ncpu
    allocate(regions(1)%gpus(ngpu), stat=st)
    do i = 1, ngpu
       regions(1)%gpus(i) = i
    end do

  end subroutine sylver_topology_create_flat

  ! Create NUMA topology with `ncpu` CPU cores and `ngpu` GPU devices
  !
  subroutine sylver_topology_create_numa(ncpu, ngpu, regions)
    use spral_hw_topology, only : numa_region
    implicit none
    ! Number of CPU wokers involved the execution
    integer, intent(in) :: ncpu
    ! Number of CUDA wokers involved the execution
    integer, intent(in) :: ngpu
    ! Machine topology
    type(numa_region), dimension(:), allocatable, intent(out) :: regions

    integer :: i
    integer :: ncpu_tot
    ! Number of NUMA regions available on the architecture
    integer(C_INT) :: c_nregions
    type(C_PTR) :: c_regions
    type(c_numa_region), dimension(:), pointer, contiguous :: f_regions
    integer nregions
   
    integer :: st ! Error management

    ! print *, "[sylver_topology_create_numa] ngpu = ", ngpu
    
    ! Make sure akeep%topology array is not allocated
    if (allocated(regions)) deallocate(regions, stat=st)
    
    call sylver_topology_create_c(c_nregions, c_regions)

    if (c_nregions .le. 0) then
       ! Something went wrong and hwloc detected no NUMA nodes so we
       ! fallback to flat architecture
       call sylver_topology_create_flat(ncpu, ngpu, regions)
       return
    end if
    
    ! Create NUMA regions
    if (c_associated(c_regions)) then
       call c_f_pointer(c_regions, f_regions, shape=(/ c_nregions /))

       ! Count the number of NUMA regions involved in the computation
       ! repending on `ncpu`
       !
       ! Zero the number of NUMA region needed 
       nregions = 0
       ! Zero the total number of CPU cores available in the NUMA regions
       ncpu_tot = 0
       do i = 1, c_nregions
          if (f_regions(i)%nproc .le. 0) then
             ! If the current NUMA region has no cores, we ignore it
             cycle
          end if
          ! Add NUMA region
          nregions = nregions + 1
          ! Add corresponding CPU cores
          ncpu_tot = ncpu_tot + f_regions(i)%nproc
          ! Exit loop if we have enough CPU cores
          if (ncpu_tot .ge. ncpu) exit
       end do

       ! Copy to allocatable array
       allocate(regions(nregions), stat=st)
       if (st .ne. 0) return
       do i = 1, nregions
          if (f_regions(i)%nproc .le. 0) then
             ! If the current NUMA region has no cores, we ignore it
             cycle
          end if
          regions(i)%nproc = f_regions(i)%nproc
          allocate(regions(i)%gpus(0), stat=st)
       end do
    end if

    ! Attach all GPU devices to the first region (there should be at
    ! least one region)
    if (allocated(regions(1)%gpus)) deallocate(regions(1)%gpus)
    allocate(regions(1)%gpus(ngpu), stat=st)
    do i = 1, ngpu
       ! FIXME: should device be zero-indexed?
       regions(1)%gpus(i) = i
    end do
    
  end subroutine sylver_topology_create_numa
    
  subroutine sylver_topology_create(ncpu, ngpu, options, regions)
    use spral_hw_topology, only : numa_region
    use sylver_datatypes_mod
    implicit none
    ! Number of CPU wokers involved the execution
    integer, intent(in) :: ncpu
    ! Number of CUDA wokers involved the execution
    integer, intent(in) :: ngpu
    ! SyLVER options
    type(sylver_options), target, intent(in) :: options
    ! Machine topology
    type(numa_region), dimension(:), allocatable, intent(out) :: regions

    integer :: i

    if (options%cpu_topology .eq. SYLVER_CPU_TOPOLOGY_FLAT) then

       call sylver_topology_create_flat(ncpu, ngpu, regions)

    else if(options%cpu_topology .eq. SYLVER_CPU_TOPOLOGY_NUMA) then

       call sylver_topology_create_numa(ncpu, ngpu, regions)

    else if(options%cpu_topology .eq. SYLVER_CPU_TOPOLOGY_DEFAULT) then

       if (ngpu .gt. 0) then
          ! If CPU-only execution is requested, use NUMA topology
          call sylver_topology_create_flat(ncpu, ngpu, regions)          
       else
          ! If CPU-GPU execution is requested, use flat topology
          call sylver_topology_create_numa(ncpu, ngpu, regions)
       end if
       
    end if
    
  end subroutine sylver_topology_create
  
end module sylver_topology_mod
