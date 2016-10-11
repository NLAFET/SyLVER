module spldlt_analyse_mod
  use spral_ssids_akeep, only: ssids_akeep 
  use, intrinsic :: iso_c_binding
  use spral_ssids_cpu_iface ! fixme only
  use spral_ssids_datatypes
  implicit none

  type spldlt_akeep_type
     type(ssids_akeep), pointer  :: akeep => null()
     type(c_ptr) :: symbolic_tree_c
  end type spldlt_akeep_type

  interface spldlt_create_symbolic_tree_c
     type(c_ptr) function spldlt_create_symbolic_tree(n, nnodes, sptr, sparent, rptr, rlist, &
          nptr, nlist, options) &
          bind(C, name="spldlt_create_symbolic_tree")
       use, intrinsic :: iso_c_binding
       import :: cpu_factor_options
       implicit none
       integer(C_INT), value :: n
       integer(c_int), value :: nnodes
       integer(c_int), dimension(*), intent(in) :: sptr
       integer(c_int), dimension(*), intent(in) :: sparent
       integer(c_long), dimension(*), intent(in) :: rptr
       integer(c_int), dimension(*), intent(in) :: rlist
       integer(c_int), dimension(*), intent(in) :: nptr
       integer(c_int), dimension(2, *), intent(in) :: nlist
       type(cpu_factor_options), intent(in) :: options
     end function spldlt_create_symbolic_tree
  end interface spldlt_create_symbolic_tree_c

contains

  subroutine spldlt_analyse(spldlt_akeep, akeep, options)
    use spral_ssids_akeep, only: ssids_akeep
    use spral_ssids_datatypes
    implicit none
    
    type(spldlt_akeep_type) :: spldlt_akeep ! spldlt akeep structure 
    type(ssids_akeep), target :: akeep ! ssids akeep structure
    type(ssids_options), intent(in) :: options

    integer nnodes
   type(cpu_factor_options) :: coptions

    spldlt_akeep%akeep => akeep 
    nnodes = akeep%nnodes

    ! print *, "[spldlt_analyse] nnodes: ", nnodes
    ! print *, "sptr: ", akeep%sptr(1:nnodes+1)

    ! call C++ analyse routine
    call cpu_copy_options_in(options, coptions)
    spldlt_akeep%symbolic_tree_c = spldlt_create_symbolic_tree_c(akeep%n, nnodes, &
         akeep%sptr, akeep%sparent, akeep%rptr, akeep%rlist, &
         akeep%nptr, akeep%nlist, coptions)

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
  
end module spldlt_analyse_mod
