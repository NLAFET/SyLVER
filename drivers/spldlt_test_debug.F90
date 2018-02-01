program spldlt_test_debug
  use, intrinsic :: iso_c_binding
  use spral_ssids
  use spral_ssids_subtree, only: symbolic_subtree_base
  use spral_ssids_cpu_subtree, only : cpu_symbolic_subtree
  use starpu_f_mod
  use spldlt_factorize_mod
  use spldlt_analyse_mod
  implicit none
   
  type(ssids_options) :: options
  type(ssids_akeep), target :: akeep   ! analysis data
  type(ssids_fkeep), target :: fkeep   ! factorization data
  integer :: i
  class(cpu_symbolic_subtree), pointer :: subtree_ptr => null()
  integer(c_int) :: ret ! starpu_init return value
  type(c_ptr) :: symbolic_tree, numeric_tree 
  type(c_ptr) :: cakeep, cfkeep 

  if (allocated(akeep%subtree)) deallocate(akeep%subtree)

  print *, "[spldlt_test_debug]"

  ! Init akeep
  akeep%n = 1
  akeep%nparts = 4 
  akeep%nnodes = 4 

  allocate(akeep%subtree(akeep%nparts))

  nullify(subtree_ptr)
  do i = 1, akeep%nparts

     nullify(subtree_ptr)
     allocate(subtree_ptr) ! Allocate a new cpu_subtree object

     ! Setup cpu_subtree
     subtree_ptr%n = akeep%n
     subtree_ptr%csubtree = c_null_ptr

     akeep%subtree(i)%ptr => subtree_ptr   
  end do
  
  cakeep = c_loc(akeep)

  symbolic_tree = spldlt_create_symbolic_tree_c(cakeep, akeep%nnodes, akeep%nparts)
  
  allocate(fkeep%subtree(akeep%nparts))

  cfkeep = c_loc(fkeep)

  ret = starpu_f_init_c(2) ! Use 2 CPUs
  
  numeric_tree = spldlt_create_numeric_tree_dlb(cfkeep, symbolic_tree)

  call starpu_f_shutdown()

end program spldlt_test_debug
