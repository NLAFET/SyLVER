module spldlt_datatypes_mod
  use spral_ssids_datatypes
  implicit none
  
  type spldlt_options
     type(ssids_options) :: options
     logical :: prune_tree = .true.
  end type spldlt_options

end module spldlt_datatypes_mod
