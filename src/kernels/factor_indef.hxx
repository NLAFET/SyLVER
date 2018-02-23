#pragma once

namespace spldlt {

   template <typename T>
   void update_contrib_block(
         int m, int n,
         T *upd, int ldupd,
         int k,
         T *lik, int ld_lik, 
         T *ljk, int ld_ljk,
         bool zero_upd,
         T *dk, // Diagonal
         T *ld, int ldld // Workspace
         ) {

      T rbeta = zero_upd ? 0.0 : 1.0;
      
      // Compute Lik Dk in workspace W
      spral::ssids::cpu::calcLD<OP_N>(
            m, k, lik, ld_lik, dk, ld, ldld);
      
      // Compute U = U - W L^{T}
      host_gemm(
            OP_N, OP_T, m, n, k,
            // -1.0, ljk, ldl, ld, ldld,
            -1.0, ld, ldld, ljk, ld_ljk,
            rbeta, upd, ldupd
            );

   }
}
