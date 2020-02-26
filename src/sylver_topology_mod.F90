module sylver_topology_mod

contains

  subroutine sylver_topology_create(ncpu, ngpu, options, regions)
    use spral_hw_topology, only : numa_region
    use sylver_datatypes_mod, only: sylver_options
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

  end subroutine sylver_topology_create
  
end module sylver_topology_mod
