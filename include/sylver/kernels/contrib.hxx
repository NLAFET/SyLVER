/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

// STD
#include <stdexcept>

// SSIDS
#include "ssids/contrib.h"

namespace sylver {
namespace spldlt {

// @brief Retrieve contrib data from subtree for assembly operations
// Note: See assemble.cxx for specialized routine
template <typename T>
void contrib_get_data(
      const void *const contrib, int *const n,
      const T* *const val, int *const ldval, const int* *const rlist,
      int *const ndelay, const int* *const delay_perm,
      const T* *const delay_val, int *const lddelay);
   
}} // End of namespace sylver::spldlt
