#pragma once

// SyVLER
#include "NumericFront.hxx"

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"

namespace spldlt {

   template <typename T>
   void factor_front_unsym_rp(
         struct cpu_factor_options& options,
         NumericFront<T, PoolAlloc> &node) {

      // Extract info about front
      int m = node.get_nrow();
      int n = node.get_ncol();
      int nr = node.get_nr(); // number of block rows
      int nc = node.get_nc(); // number of block columns
      
      for(int j = 0; j < nc; ++j) {
      
      }

   }

}
