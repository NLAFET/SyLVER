/** \file
 *  \copyright 2016- The Science and Technology Facilities Council (STFC)
 *  \author    Florent Lopez
 */

#pragma once

namespace sylver {
   
   enum struct PivotMethod : int {
      app_aggressive = 1, 
      app_block      = 2, 
      tpp            = 3
   };

   enum struct FailedPivotMethod : int {
      tpp  = 1,
      pass = 2
   };

   /// @brief Interoperable subset of sylver_options
   struct sylver_options_c {
      int print_level;
      double small;
      double u;
      double multiplier;
      int nb;
      sylver::PivotMethod pivot_method;
      sylver::FailedPivotMethod failed_pivot_method;
   };

}
