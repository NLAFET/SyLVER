module spldlt_analyse_mod
  use spral_ssids_akeep, only: ssids_akeep 
  use, intrinsic :: iso_c_binding
  use spral_ssids_cpu_iface ! fixme only
  use spral_ssids_datatypes
  use spral_hw_topology, only: numa_region
  use spral_ssids_inform, only : ssids_inform
  implicit none

  type spldlt_akeep_type
     type(ssids_akeep), pointer  :: akeep => null()
     type(c_ptr) :: symbolic_tree_c
  end type spldlt_akeep_type

  interface spldlt_create_symbolic_tree_c
     type(c_ptr) function spldlt_create_symbolic_tree(n, nnodes, & 
          sptr, sparent, rptr, rlist, nptr, nlist, options) &
          bind(C, name="spldlt_create_symbolic_tree")
       use, intrinsic :: iso_c_binding
       import :: cpu_factor_options
       implicit none
       integer(c_int), value :: n
       integer(c_int), value :: nnodes
       integer(c_int), dimension(*), intent(in) :: sptr
       integer(c_int), dimension(*), intent(in) :: sparent
       integer(c_long), dimension(*), intent(in) :: rptr
       integer(c_int), dimension(*), intent(in) :: rlist
       integer(c_long), dimension(*), intent(in) :: nptr
       integer(c_long), dimension(2, *), intent(in) :: nlist
       type(cpu_factor_options), intent(in) :: options
     end function spldlt_create_symbolic_tree
  end interface spldlt_create_symbolic_tree_c

contains

  subroutine spldlt_analyse(spldlt_akeep, akeep, options, inform, ncpu)
    use spral_ssids_akeep, only: ssids_akeep
    use spral_ssids_datatypes
    implicit none
    
    type(spldlt_akeep_type) :: spldlt_akeep ! spldlt akeep structure 
    type(ssids_akeep), target :: akeep ! ssids akeep structure
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(inout) :: inform
    integer :: ncpu ! number of CPU workers

    integer nnodes
    type(cpu_factor_options) :: coptions
    integer :: i
    type(numa_region), allocatable :: regions(:)
    integer, dimension(:), allocatable :: contrib_dest, exec_loc
    ! Error management
    integer :: st
    
    spldlt_akeep%akeep => akeep 
    nnodes = akeep%nnodes

    ! Create simple topology with ncpu regions, one for each CPU
    ! worker
    allocate(regions(ncpu))
    do i = 1, ncpu
       regions(i)%nproc = 1
    end do

    ! Sort out subtrees
    ! print *, "Input topology"    
    ! do i = 1, size(akeep%topology)
    !    print *, "Region ", i, " with ", akeep%topology(i)%nproc, " cores"
    !    if(size(akeep%topology(i)%gpus).gt.0) &
    !         print *, "---> gpus ", akeep%topology(i)%gpus
    ! end do

    ! Deallocate partitions given by SSIDS
    deallocate(akeep%part)
    deallocate(akeep%contrib_ptr)
    deallocate(akeep%contrib_idx)

    ! Find subtree partition
    call find_subtree_partition(akeep%nnodes, akeep%sptr, akeep%sparent,           &
         akeep%rptr, options, akeep%topology, akeep%nparts, akeep%part,            &
         exec_loc, akeep%contrib_ptr, akeep%contrib_idx, contrib_dest, inform, st)
    if (st .ne. 0) go to 100
    print *, "contrib_ptr = ", akeep%contrib_ptr(1:akeep%nparts+1)
    print *, "contrib_idx = ", akeep%contrib_idx(1:akeep%nparts)
    print *, "contrib_dest = ", &
      contrib_dest(1:akeep%contrib_ptr(akeep%nparts+1)-1)


    ! print *, "[spldlt_analyse] nnodes: ", nnodes
    ! print *, "sptr: ", akeep%sptr(1:nnodes+1)

    ! call C++ analyse routine
    call cpu_copy_options_in(options, coptions)
    spldlt_akeep%symbolic_tree_c = &
         spldlt_create_symbolic_tree_c(akeep%n, nnodes, akeep%sptr, &
         akeep%sparent, akeep%rptr, akeep%rlist, akeep%nptr, akeep%nlist, &
         coptions)

    return
100 continue
    
    
    print *, "[Error][spldlt_analyse] st: ", st

    return
  end subroutine spldlt_analyse
  
  subroutine spldlt_print_atree(akeep)
    use spral_ssids_akeep, only: ssids_akeep
    ! use spral_ssids
    
    type(ssids_akeep), intent(in) :: akeep

    integer :: num_nodes
    integer :: node
    integer :: n, m ! node sizes

    print *, "Print atree"

    num_nodes = akeep%nnodes

    print *, "num_nodes: ", num_nodes

    open(2, file="atree.dot")

    write(2, '("graph atree {")')
    write(2, '("node [")')
    write(2, '("style=filled")')
    write(2, '("]")')
    
    do node = 1, num_nodes

       n = akeep%sptr(node+1) - akeep%sptr(node) 
       m = int(akeep%rptr(node+1) - akeep%rptr(node))

       ! node id
       write(2, '(i10)', advance="no") node
       write(2, '(" ")', advance="no")
       write(2, '("[")', advance="no")

       ! node info
       write(2, '("label=""")', advance="no")
       write(2, '("node:", i5,"\n")', advance="no")node
       write(2, '("m:", i5,"\n")', advance="no")m
       write(2, '("n:", i5,"\n")', advance="no")n

       write(2, '("""")', advance="no")
       write(2, '("]")', advance="no")
       write(2, '(" ")')

       ! parent node
       if(akeep%sparent(node) .ne. -1) write(2, '(i10, "--", i10)')akeep%sparent(node), node
    end do

    write(2, '("}")')

    close(2)

  end subroutine spldlt_print_atree

  !****************************************************************************
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> @brief Partition an elimination tree for execution on different NUMA regions
  !>        and GPUs.
  !>
  !> Start with a single tree, and proceed top down splitting the largest subtree
  !> (in terms of total flops)  until we have a sufficient number of independent
  !> subtrees. A sufficient number is such that subtrees can be assigned to NUMA
  !> regions and GPUs with a load balance no worse than max_load_inbalance.
  !> Load balance is calculated as the maximum value over all regions/GPUs of:
  !> \f[ \frac{ n x_i / \alpha_i } { \sum_j (x_j/\alpha_j) } \f]
  !> Where \f$ \alpha_i \f$ is the performance coefficient of region/GPU i,
  !> \f$ x_i \f$ is the number of flops assigned to region/GPU i and \f$ n \f$ is
  !> the total number of regions. \f$ \alpha_i \f$ should be proportional to the
  !> speed of the region/GPU (i.e. if GPU is twice as fast as CPU, set alpha for
  !> CPU to 1.0 and alpha for GPU to 2.0).
  !>
  !> If the original number of flops is greater than min_gpu_work and the
  !> performance coefficient of a GPU is greater than the combined coefficients
  !> of the CPU, then subtrees will not be split to become smaller than
  !> min_gpu_work until all GPUs are filled.
  !>
  !> If the balance criterion cannot be satisfied after we have split into
  !> 2 * (total regions/GPUs), we just use the best obtained value.
  !>
  !> GPUs may only handle leaf subtrees, so the top nodes are assigned to the
  !> full set of CPUs.
  !>
  !> Parts are returned as contigous ranges of nodes. Part i consists of nodes
  !> part(i):part(i+1)-1
  !>
  !> @param nnodes Total number of nodes
  !> @param sptr Supernode pointers. Supernode i consists of nodes
  !>        sptr(i):sptr(i+1)-1.
  !> @param sparent Supernode parent array. Supernode i has parent sparent(i).
  !> @param rptr Row pointers. Supernode i has rows rlist(rptr(i):rptr(i+1)-1).
  !> @param topology Machine topology to partition for.
  !> @param min_gpu_work Minimum flops for a GPU execution to be worthwhile.
  !> @param max_load_inbalance Number greater than 1.0 representing maximum
  !>        permissible load inbalance.
  !> @param gpu_perf_coeff The value of \f$ \alpha_i \f$ used for all GPUs,
  !>        assuming that used for all NUMA region CPUs is 1.0.
  !> @param nparts Number of parts found.
  !> @param parts List of part ranges. Part i consists of supernodes
  !>        part(i):part(i+1)-1.
  !> @param exec_loc Execution location. Part i should be run on partition
  !>        mod((exec_loc(i) - 1), size(topology)) + 1.
  !>        It should be run on the CPUs if
  !>        exec_loc(i) <= size(topology),
  !>        otherwise it should be run on GPU number
  !>        (exec_loc(i) - 1)/size(topology).
  !> @param contrib_ptr Contribution pointer. Part i has contribution from
  !>        subtrees contrib_idx(contrib_ptr(i):contrib_ptr(i+1)-1).
  !> @param contrib_idx List of contributing subtrees, see contrib_ptr.
  !> @param contrib_dest Node to which each subtree listed in contrib_idx(:)
  !>        contributes.
  !> @param st Allocation status parameter. If non-zero an allocation error
  !>        occurred.
  subroutine find_subtree_partition(nnodes, sptr, sparent, rptr, options, &
       topology, nparts, part, exec_loc, contrib_ptr, contrib_idx, &
       contrib_dest, inform, st)
    implicit none
    integer, intent(in) :: nnodes
    integer, dimension(nnodes+1), intent(in) :: sptr
    integer, dimension(nnodes), intent(in) :: sparent
    integer(long), dimension(nnodes+1), intent(in) :: rptr
    type(ssids_options), intent(in) :: options
    type(numa_region), dimension(:), intent(in) :: topology
    integer, intent(out) :: nparts
    integer, dimension(:), allocatable, intent(inout) :: part
    integer, dimension(:), allocatable, intent(out) :: exec_loc
    integer, dimension(:), allocatable, intent(inout) :: contrib_ptr
    integer, dimension(:), allocatable, intent(inout) :: contrib_idx
    integer, dimension(:), allocatable, intent(out) :: contrib_dest
    type(ssids_inform), intent(inout) :: inform
    integer, intent(out) :: st

    integer :: i, j, k
    integer(long) :: jj
    integer :: m, n, node
    integer(long), dimension(:), allocatable :: flops
    integer, dimension(:), allocatable :: size_order
    logical, dimension(:), allocatable :: is_child
    real :: load_balance, best_load_balance
    integer :: nregion, ngpu
    logical :: has_parent

    ! Count flops below each node
    allocate(flops(nnodes+1), stat=st)
    if (st .ne. 0) return
    flops(:) = 0
    do node = 1, nnodes
       m = int(rptr(node+1)-rptr(node))
       n = sptr(node+1)-sptr(node)
       do jj = m-n+1, m
          flops(node) = flops(node) + jj**2
       end do
       j = sparent(node)
       flops(j) = flops(j) + flops(node)
       !print *, "Node ", node, "parent", j, " flops ", flops(node)
    end do
    !print *, "Total flops ", flops(nnodes+1)

    ! Initialize partition to be all children of virtual root
    allocate(part(nnodes+1), size_order(nnodes), exec_loc(nnodes), &
         is_child(nnodes), stat=st)
    if (st .ne. 0) return
    nparts = 0
    part(1) = 1
    do i = 1, nnodes
       if (sparent(i) .gt. nnodes) then
          nparts = nparts + 1
          part(nparts+1) = i+1
          is_child(nparts) = .true. ! All subtrees are intially child subtrees
       end if
    end do
    call create_size_order(nparts, part, flops, size_order)
    !print *, "Initial partition has ", nparts, " parts"
    !print *, "part = ", part(1:nparts+1)
    !print *, "size_order = ", size_order(1:nparts)

    ! Calculate number of regions/gpus
    nregion = size(topology)
    ngpu = 0
    do i = 1, size(topology)
       ngpu = ngpu + size(topology(i)%gpus)
    end do
    !print *, "running on ", nregion, " regions and ", ngpu, " gpus"

    ! Keep splitting until we meet balance criterion
    best_load_balance = huge(best_load_balance)
    do i = 1, 2*(nregion+ngpu)
       ! Check load balance criterion
       load_balance = calc_exec_alloc(nparts, part, size_order, is_child,  &
            flops, topology, options%min_gpu_work, options%gpu_perf_coeff, &
            exec_loc, st)
       if (st .ne. 0) return
       best_load_balance = min(load_balance, best_load_balance)
       if (load_balance .lt. options%max_load_inbalance) exit ! allocation is good
       ! Split tree further
       call split_tree(nparts, part, size_order, is_child, sparent, flops, &
            ngpu, options%min_gpu_work, st)
       if (st .ne. 0) return
    end do

    ! Consolidate adjacent non-children nodes into same part and regen exec_alloc
    !print *
    !print *, "pre merge", part(1:nparts+1)
    !print *, "exec_loc ", exec_loc(1:nparts)
    j = 1
    do i = 2, nparts
       part(j+1) = part(i)
       if (is_child(i) .or. is_child(j)) then
          ! We can't merge j and i
          j = j + 1
          is_child(j) = is_child(i)
       end if
    end do
    part(j+1) = part(nparts+1)
    nparts = j
    !print *, "post merge", part(1:nparts+1)
    call create_size_order(nparts, part, flops, size_order)
    load_balance = calc_exec_alloc(nparts, part, size_order, is_child,  &
         flops, topology, options%min_gpu_work, options%gpu_perf_coeff, &
         exec_loc, st)
    if (st .ne. 0) return
    !print *, "exec_loc ", exec_loc(1:nparts)

    ! Merge adjacent subtrees that are executing on the same node so long as
    ! there is no more than one contribution to a parent subtree
    j = 1
    k = sparent(part(j+1)-1)
    has_parent = (k .le. nnodes)
    do i = 2, nparts
       part(j+1) = part(i)
       exec_loc(j+1) = exec_loc(i)
       k = sparent(part(i+1)-1)
       if ((exec_loc(i) .ne. exec_loc(j)) .or. (has_parent .and. (k .le. nnodes))) then
          ! We can't merge j and i
          j = j + 1
          has_parent = .false. 
       end if
       has_parent = has_parent.or.(k.le.nnodes)
    end do
    part(j+1) = part(nparts+1)
    nparts = j

    ! Figure out contribution blocks that are input to each part
    allocate(contrib_ptr(nparts+3), contrib_idx(nparts), contrib_dest(nparts), &
         stat=st)
    if (st .ne. 0) return
    ! Count contributions at offset +2
    contrib_ptr(3:nparts+3) = 0
    do i = 1, nparts-1 ! by defn, last part has no parent
       j = sparent(part(i+1)-1) ! node index of parent
       if (j .gt. nnodes) cycle ! part is a root
       k = i+1 ! part index of j
       do while(j .ge. part(k+1))
          k = k + 1
       end do
       contrib_ptr(k+2) = contrib_ptr(k+2) + 1
    end do
    ! Figure out contrib_ptr starts at offset +1
    contrib_ptr(1:2) = 1
    do i = 1, nparts
       contrib_ptr(i+2) = contrib_ptr(i+1) + contrib_ptr(i+2)
    end do
    ! Drop sources into list
    do i = 1, nparts-1 ! by defn, last part has no parent
       j = sparent(part(i+1)-1) ! node index of parent
       if (j .gt. nnodes) then
          ! part is a root
          contrib_idx(i) = nparts+1
          cycle
       end if
       k = i+1 ! part index of j
       do while (j .ge. part(k+1))
          k = k + 1
       end do
       contrib_idx(i) = contrib_ptr(k+1)
       contrib_dest(contrib_idx(i)) = j
       contrib_ptr(k+1) = contrib_ptr(k+1) + 1
    end do
    contrib_idx(nparts) = nparts+1 ! last part must be a root

    ! Fill out inform
    inform%nparts = nparts
    inform%gpu_flops = 0
    do i = 1, nparts
       if (exec_loc(i) .gt. size(topology)) &
            inform%gpu_flops = inform%gpu_flops + flops(part(i+1)-1)
    end do
    inform%cpu_flops = flops(nnodes+1) - inform%gpu_flops
  end subroutine find_subtree_partition

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> @brief Allocate execution of subtrees to resources and calculate load balance
  !>
  !> Given the partition supplied, uses a greedy algorithm to assign subtrees to
  !> resources specified by topology and then returns the resulting load balance
  !> as
  !> \f[ \frac{\max_i( n x_i / \alpha_i )} { \sum_j (x_j/\alpha_j) } \f]
  !> Where \f$ \alpha_i \f$ is the performance coefficient of region/GPU i,
  !> \f$ x_i \f$ is the number of flops assigned to region/GPU i and \f$ n \f$ is
  !> the total number of regions. \f$ \alpha_i \f$ should be proportional to the
  !> speed of the region/GPU (i.e. if GPU is twice as fast as CPU, set alpha for
  !> CPU to 1.0 and alpha for GPU to 2.0).
  !>
  !> Work is only assigned to GPUs if the subtree has at least min_gpu_work flops.
  !>
  !> None-child subtrees are ignored (they will be executed using all available
  !> resources). They are recorded with exec_loc -1.
  !>
  !> @param nparts Number of parts.
  !> @param parts List of part ranges. Part i consists of supernodes
  !>        part(i):part(i+1)-1.
  !> @param size_order Lists parts in decreasing order of flops.
  !>        i.e. size_order(1) is the largest part.
  !> @param is_child True if subtree is a child subtree (has no contributions
  !>        from other subtrees).
  !> @param flops Number of floating points in subtree rooted at each node.
  !> @param topology Machine topology to allocate execution for.
  !> @param min_gpu_work Minimum work before allocation to GPU is useful.
  !> @param gpu_perf_coeff The value of \f$ \alpha_i \f$ used for all GPUs,
  !>        assuming that used for all NUMA region CPUs is 1.0.
  !> @param exec_loc Execution location. Part i should be run on partition
  !>        mod((exec_loc(i) - 1), size(topology)) + 1.
  !>        It should be run on the CPUs if
  !>        exec_loc(i) <= size(topology),
  !>        otherwise it should be run on GPU number
  !>        (exec_loc(i) - 1)/size(topology).
  !> @param st Allocation status parameter. If non-zero an allocation error
  !>        occurred.
  !> @returns Load balance value as detailed in subroutine description.
  !> @sa find_subtree_partition()
  ! FIXME: Consider case when gpu_perf_coeff > 2.0 ???
  !        (Round robin may not be correct thing)
  real function calc_exec_alloc(nparts, part, size_order, is_child, flops, &
       topology, min_gpu_work, gpu_perf_coeff, exec_loc, st)
    implicit none
    integer, intent(in) :: nparts
    integer, dimension(nparts+1), intent(in) :: part
    integer, dimension(nparts), intent(in) :: size_order
    logical, dimension(nparts), intent(in) :: is_child
    integer(long), dimension(*), intent(in) :: flops
    type(numa_region), dimension(:), intent(in) :: topology
    integer(long), intent(in) :: min_gpu_work
    real, intent(in) :: gpu_perf_coeff
    integer, dimension(nparts), intent(out) :: exec_loc
    integer, intent(out) :: st

    integer :: i, p, nregion, ngpu, max_gpu, next
    integer(long) :: pflops
    integer, dimension(:), allocatable :: map ! List resources in order of
    ! decreasing power
    real, dimension(:), allocatable :: load_balance
    real :: total_balance

    ! Initialise in case of an error return
    calc_exec_alloc = huge(calc_exec_alloc)

    !
    ! Create resource map
    !
    nregion = size(topology)
    ngpu = 0
    max_gpu = 0
    do i = 1, size(topology)
       ngpu = ngpu + size(topology(i)%gpus)
       max_gpu = max(max_gpu, size(topology(i)%gpus))
    end do
    allocate(map(nregion+ngpu), stat=st)
    if (st .ne. 0) return

    if (gpu_perf_coeff .gt. 1.0) then
       ! GPUs are more powerful than CPUs
       next = 1
       do i = 1, size(topology)
          do p = 1, size(topology(i)%gpus)
             map(next) = p*nregion + i
             next = next + 1
          end do
       end do
       do i = 1, size(topology)
          map(next) = i
          next = next + 1
       end do
    else
       ! CPUs are more powerful than GPUs
       next = 1
       do i = 1, size(topology)
          map(next) = i
          next = next + 1
       end do
       do i = 1, size(topology)
          do p = 1, size(topology(i)%gpus)
             map(next) = p*nregion + i
             next = next + 1
          end do
       end do
    end if

    !
    ! Simple round robin allocation in decreasing size order.
    !
    next = 1
    do i = 1, nparts
       p = size_order(i)
       if (.not. is_child(p)) then
          ! Not a child subtree
          exec_loc(p) = -1
          cycle
       end if
       pflops = flops(part(p+1)-1)
       if (pflops .lt. min_gpu_work) then
          ! Avoid GPUs
          do while (map(next) .gt. nregion)
             next = next + 1
             if (next .gt. size(map)) next = 1
          end do
       end if
       exec_loc(p) = map(next)
       next = next + 1
       if (next .gt. size(map)) next = 1
    end do

    !
    ! Calculate load inbalance
    !
    allocate(load_balance(nregion*(1+max_gpu)), stat=st)
    if (st .ne. 0) return
    load_balance(:) = 0.0
    total_balance = 0.0
    ! Sum total 
    do p = 1, nparts
       if (exec_loc(p) .eq. -1) cycle ! not a child subtree
       pflops = flops(part(p+1)-1)
       if (exec_loc(p) .gt. nregion) then
          ! GPU
          load_balance(exec_loc(p)) = load_balance(exec_loc(p)) + &
               real(pflops) / gpu_perf_coeff
          total_balance = total_balance + real(pflops) / gpu_perf_coeff
       else
          ! CPU
          load_balance(exec_loc(p)) = load_balance(exec_loc(p)) + real(pflops)
          total_balance = total_balance + real(pflops)
       end if
    end do
    ! Calculate n * max(x_i/a_i) / sum(x_j/a_j)
    calc_exec_alloc = (nregion+ngpu) * maxval(load_balance(:)) / total_balance
  end function calc_exec_alloc

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Split tree into an additional part as required by
!>        find_subtree_partition().
!>
!> Split largest partition into two parts, unless doing so would reduce the
!> number of subtrees with at least min_gpu_work below ngpu.
!>
!> Note: We require all input parts to have a single root.
!>
!> @param nparts Number of parts: normally increased by one on return.
!> @param part Part i consists of nodes part(i):part(i+1).
!> @param size_order Lists parts in decreasing order of flops.
!>        i.e. size_order(1) is the largest part.
!> @param is_child True if subtree is a child subtree (has no contributions
!>        from other subtrees).
!> @param sparent Supernode parent array. Supernode i has parent sparent(i).
!> @param flops Number of floating points in subtree rooted at each node.
!> @param ngpu Number of gpus.
!> @param min_gpu_work Minimum worthwhile work to give to GPU.
!> @param st Allocation status parameter. If non-zero an allocation error
!>        occurred.
!> @sa find_subtree_partition()
  subroutine split_tree(nparts, part, size_order, is_child, sparent, flops, &
       ngpu, min_gpu_work, st)
    implicit none
    integer, intent(inout) :: nparts
    integer, dimension(*), intent(inout) :: part
    integer, dimension(*), intent(inout) :: size_order
    logical, dimension(*), intent(inout) :: is_child
    integer, dimension(*), intent(in) :: sparent
    integer(long), dimension(*), intent(in) :: flops
    integer, intent(in) :: ngpu
    integer(long), intent(in) :: min_gpu_work
    integer, intent(out) :: st

    integer :: i, p, nchild, nbig, root, to_split, old_nparts
    integer, dimension(:), allocatable :: children, temp

    ! Look for all children of root in biggest child part
    nchild = 0
    allocate(children(10), stat=st) ! we will resize if necessary
    if (st.ne.0) return
    ! Find biggest child subtree
    to_split = 1
    do while(.not. is_child(size_order(to_split)))
       to_split = to_split + 1
    end do
    to_split = size_order(to_split)
    ! Find all children of root
    root = part(to_split+1)-1
    do i = part(to_split), root-1
       if (sparent(i) .eq. root) then
          nchild = nchild+1
          if (nchild .gt. size(children)) then
             ! Increase size of children(:)
             allocate(temp(2*size(children)), stat=st)
             if (st .ne. 0) return
             temp(1:size(children)) = children(:)
             deallocate(children)
             call move_alloc(temp, children)
          end if
          children(nchild) = i
       end if
    end do

    ! Check we can split safely
    if (nchild .eq. 0) return ! singleton node, can't split
    nbig = 0 ! number of new parts > min_gpu_work
    do i = to_split+1, nparts
       p = size_order(i)
       if (.not. is_child(p)) cycle ! non-children can't go on GPUs
       root = part(p+1)-1
       if (flops(root) .lt. min_gpu_work) exit
       nbig = nbig + 1
    end do
    if ((nbig+1) .ge. ngpu) then
       ! Original partition met min_gpu_work criterion
       do i = 1, nchild
          if (flops(children(i)) .ge. min_gpu_work) nbig = nbig + 1
       end do
       if (nbig .lt. ngpu) return ! new partition fails min_gpu_work criterion
    end if

    ! Can safely split, so do so. As part to_split was contigous, when
    ! split the new parts fall into the same region. Thus, we first push any
    ! later regions back to make room, then add the new parts.
    part(to_split+nchild+1:nparts+nchild+1) = part(to_split+1:nparts+1)
    is_child(to_split+nchild+1:nparts+nchild) = is_child(to_split+1:nparts)
    do i = 1, nchild
       ! New part corresponding to child i *ends* at part(to_split+i)-1
       part(to_split+i) = children(i)+1
    end do
    is_child(to_split:to_split+nchild-1) = .true.
    is_child(to_split+nchild) = .false. ! Newly created non-parent subtree
    old_nparts = nparts
    nparts = old_nparts + nchild

    ! Finally, recreate size_order array
    call create_size_order(nparts, part, flops, size_order)
  end subroutine split_tree

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Determine order of subtrees based on size
!>
!> @note Sorting algorithm could be improved if this becomes a bottleneck.
!>
!> @param nparts Number of parts: normally increased by one on return.
!> @param part Part i consists of nodes part(i):part(i+1).
!> @param flops Number of floating points in subtree rooted at each node.
!> @param size_order Lists parts in decreasing order of flops.
!>        i.e. size_order(1) is the largest part.
  subroutine create_size_order(nparts, part, flops, size_order)
    implicit none
    integer, intent(in) :: nparts
    integer, dimension(nparts+1), intent(in) :: part
    integer(long), dimension(*), intent(in) :: flops
    integer, dimension(nparts), intent(out) :: size_order

    integer :: i, j
    integer(long) :: iflops

    do i = 1, nparts
       ! We assume parts 1:i-1 are in order and aim to insert part i
       iflops = flops(part(i+1)-1)
       do j = 1, i-1
          if (iflops .gt. flops(part(j+1)-1)) exit ! node i belongs in posn j
       end do
       size_order(j+1:i) = size_order(j:i-1)
       size_order(j) = i
    end do
  end subroutine create_size_order
  
end module spldlt_analyse_mod
