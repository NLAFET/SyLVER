!> \file
!> \copyright 2016- The Science and Technology Facilities Council (STFC)
!> \author    Florent Lopez
module spldlt_datatypes_mod
  use spral_ssids_datatypes
  implicit none
  
  type spldlt_options
     type(ssids_options) :: super
     logical :: prune_tree = .true.
  end type spldlt_options

  type splu_options
     integer :: nb ! Block size 
  end type splu_options

end module spldlt_datatypes_mod
