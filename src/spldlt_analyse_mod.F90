module spldlt_analyse_mod
  implicit none

  

contains
  
  subroutine spldlt_print_atree(akeep)
    use spral_ssids_akeep
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
