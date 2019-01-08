module sylver_ciface_mod
  use, intrinsic :: iso_c_binding
  implicit none
 
  !> @brief Interoperable subset of sylver_options
  type , bind(C) :: spldlt_options_c
     integer(C_INT) :: print_level
     real(C_DOUBLE) :: small
     real(C_DOUBLE) :: u
     real(C_DOUBLE) :: multiplier
     integer(C_INT) :: cpu_block_size
     integer(C_INT) :: pivot_method
     integer(C_INT) :: failed_pivot_method
  end type spldlt_options_c

end module sylver_ciface_mod
