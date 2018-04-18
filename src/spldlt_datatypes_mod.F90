module spldlt_datatypes_mod

  type spldlt_options
     type(ssids_akeep) :: options
     logical :: prune_tree = .true.
  end type spldlt_options

end module spldlt_datatypes_mod
