module spldlt_factorize_mod

contains

  subroutine spldlt_factorize(spldlt_akeep, fkeep)
    use spral_ssids_akeep
    use spral_ssids_fkeep
    use spldlt_analyse_mod
    implicit none
    
    type(spldlt_akeep_type), target :: spldlt_akeep
    type(ssids_akeep) :: fkeep

    type(ssids_akeep), pointer :: akeep

    akeep => spldlt_akeep%akeep

  end subroutine spldlt_factorize
  
end module spldlt_factorize_mod
