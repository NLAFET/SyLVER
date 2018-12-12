#pragma once

// SyVLER
#include "Block.hxx"
#include "kernels/factor_unsym.hxx"

// SSIDS
#include "ssids/cpu/Workspace.hxx"

namespace spldlt {

   /// @brief Perfom LU factorization on block dblk using partial
   /// pivoting
   template <typename T>
   void factor_block_lu_pp_task(BlockUnsym<T>& dblk, int *perm) {
      
      factor_block_lu_pp(
            dblk.m, dblk.n, perm, dblk.a, dblk.lda, dblk.b, dblk.ldb);
   }

   /// @brief Apply row permutation perm on block rblk
   template <typename T>
   void apply_rperm_block_task(
         int *perm, BlockUnsym<T>& rblk,
         std::vector<spral::ssids::cpu::Workspace>& workspaces) {

      // Block dimensions
      int m = rblk.m;
      int n = rblk.n;
      
      spral::ssids::cpu::Workspace& workspace = workspaces[0]; 
      int ldw = spral::ssids::cpu::align_lda<T>(m);
      T* work = workspace.get_ptr<T>(ldw*n);
      apply_rperm_block(m, n, perm, rblk.a, rblk.lda, work, ldw);
   }

   template <typename T>
   void applyL_block_task(
         int *perm, BlockUnsym<T>& dblk, BlockUnsym<T>& ublk,
         std::vector<spral::ssids::cpu::Workspace>& workspaces) {

      // Block dimensions
      int m = ublk.m;
      int n = ublk.n;      
      
      // applyL_block(
      //    m, n, dblk.a, dblk.lda, perm, ublk, int ld_a_kj,
      //    T *work, int ldw)
   }
}
