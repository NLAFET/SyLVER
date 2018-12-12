#pragma once

// SyVLER
#include "NumericFront.hxx"
#include "tasks/tasks_unsym.hxx"

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"

namespace spldlt {

   /// @brief Task-based front factorization routine using Restricted
   /// Pivoting (RP)
   /// @Note No delays, potentially unstable
   template <typename T, typename PoolAlloc>
   void factor_front_unsym_rp(
         NumericFront<T, PoolAlloc> &node) {

      // Extract front info
      int m = node.get_nrow(); // Frontal matrix order
      int n = node.get_ncol(); // Number of fully-summed rows/columns 
      int nr = node.get_nr(); // number of block rows
      int nc = node.get_nc(); // number of block columns

      printf("[factor_front_unsym_rp] m = %d\n", m);
      printf("[factor_front_unsym_rp] n = %d\n", n);
      
      for(int k = 0; k < nc; ++k) {

         BlockUnsym<T>& dblk = node.get_block_unsym(k, k);
         
         factor_block_lu_pp_task(dblk, node.perm);

         
      }

      node.nelim = n; // We eliminated all fully-summed rows/columns
      node.ndelay_out = 0; // No delays

   }

}
