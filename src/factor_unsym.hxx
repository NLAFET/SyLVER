#pragma once

// SyVLER
#include "NumericFront.hxx"
#include "tasks/tasks_unsym.hxx"

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/Workspace.hxx"

namespace spldlt {

   /// @brief Task-based front factorization routine using Restricted
   /// Pivoting (RP)
   /// @Note No delays, potentially unstable
   template <typename T, typename PoolAlloc>
   void factor_front_unsym_rp(
         NumericFront<T, PoolAlloc> &node,
         std::vector<spral::ssids::cpu::Workspace>& workspaces) {

      // Extract front info
      int m = node.get_nrow(); // Frontal matrix order
      int n = node.get_ncol(); // Number of fully-summed rows/columns 
      int nr = node.get_nr(); // number of block rows
      int nc = node.get_nc(); // number of block columns

      int blksz = node.blksz;
      
      printf("[factor_front_unsym_rp] m = %d\n", m);
      printf("[factor_front_unsym_rp] n = %d\n", n);
      
      for(int k = 0; k < nc; ++k) {

         BlockUnsym<T>& dblk = node.get_block_unsym(k, k);
         
         factor_block_lu_pp_task(dblk, &node.perm[k*blksz]);

         for (int j = 0; j < k; ++j) {
            BlockUnsym<T>& rblk = node.get_block_unsym(k, j);
            // Apply row permutation on left-diagonal blocks 
            apply_rperm_block_task(&node.perm[k*blksz], rblk, workspaces);
         }

         for (int j = k+1; j < nc; ++j) {

            BlockUnsym<T>& ublk = node.get_block_unsym(k, j);

            applyL_block_task(&node.perm[k*blksz], )
         }
      }

      node.nelim = n; // We eliminated all fully-summed rows/columns
      node.ndelay_out = 0; // No delays

   }

}
