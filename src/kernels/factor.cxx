/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

#include "factor.hxx"

namespace spldlt {

   template<>
   void factor_subtree<double>(
         void *akeep,
         void *fkeep,
         int p,
         double *aval, 
         void **child_contrib, 
         struct spral::ssids::cpu::cpu_factor_options *options,
         spral::ssids::cpu::ThreadStats *stats) {

      spldlt_factor_subtree_c(akeep, fkeep, p, aval, child_contrib, options, stats);
   }
}
