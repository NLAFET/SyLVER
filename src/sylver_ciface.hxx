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
   struct options_c {
      int print_level;
      double small;
      double u;
      double multiplier;
      int nb; // Block size
      sylver::PivotMethod pivot_method;
      sylver::FailedPivotMethod failed_pivot_method;
   };
   using options_t = struct options_c;
   
   // @brief SyLVER error/warning flags.
   //
   // Must match Fortran definitions in src/sylver_datatypes_mod.F90
   enum Flag : int {
      SUCCESS                 = 0,

      ERROR_SINGULAR          = -5,
      ERROR_NOT_POS_DEF       = -6,
      ERROR_ALLOCATION        = -50,

      WARNING_FACT_SINGULAR   = 7
   };

   struct inform_c {
      Flag flag = Flag::SUCCESS; ///< Error flag for thread
   };
   using inform_t = struct inform_c;

}
