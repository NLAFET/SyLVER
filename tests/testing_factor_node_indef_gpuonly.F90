subroutine factor_node_indef_gpuonly_test(m, n) bind(c)
  use, intrinsic :: iso_c_binding
  ! use spral_ssids_gpu_dense_factor, only: node_ldlt
  implicit none

  integer(c_int) :: m
  integer(c_int) :: n

  print *, "[factor_node_indef_gpuonly_test]"
  
end subroutine factor_node_indef_gpuonly_test
