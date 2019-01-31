/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

// SyLVER
#include "kernels/assemble.hxx"
// SSIDS
#include "ssids/contrib.h"

namespace spldlt {

   template<>
   void contrib_get_data<double>(
         const void *const contrib, int *const n,
         const double* *const val, int *const ldval, const int* *const rlist,
         int *const ndelay, const int* *const delay_perm,
         const double* *const delay_val, int *const lddelay) {
      
      // Call specialised routine
      spral_ssids_contrib_get_data(contrib, n, val, ldval, rlist, ndelay, delay_perm, delay_val, lddelay);
   }
}