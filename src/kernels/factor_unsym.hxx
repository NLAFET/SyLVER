#pragma once

// SyLVER
#include "kernels/wrappers.hxx"

namespace spldlt {

   /// @brief Perform LU factorization of a block using partial pivoting
   template<typename T>
   int factor_block_lu_pp(int m, int n, T *a, int lda, int *perm) {
      
      // Factor triangular part
      int info = host_getrf(n, n, a, lda, perm);
      
      if (m > n) {
         // Apply factor U to sub-diagonal part
         spldlt::host_trsm(
               SIDE_RIGHT, FILL_MODE_UPR, OP_N, DIAG_NON_UNIT,
               m-n, n, 1.0, a, lda,
               &a[n], lda);
      }

      return info;
   }

   template<typename T>
   void applyU_block(
         int m, int n, T *u_kk, int ld_u_kk, T *a_ik, int ld_a_ik) {
      
         spldlt::host_trsm(
               SIDE_RIGHT, FILL_MODE_UPR, OP_N, DIAG_NON_UNIT,
               m, n, 1.0, u_kk, ld_u_kk, a_ik, ld_a_ik);  
   }

   /// @brief Apply L factor, no permutation
   template<typename T>
   void applyL_block(int m, int n, T *l_kk, int ld_l_kk, T *a_kj, int ld_a_kj) {

      spldlt::host_trsm(
            SIDE_LEFT, FILL_MODE_LWR, OP_N, DIAG_UNIT,
            m, n, 1.0, l_kk, ld_l_kk, a_kj, ld_a_kj);

   }

   /// @brief Apply row permutation perm and apply L factor
   template<typename T>
   void applyL_block(int m, int n, T *l_kk, int ld_l_kk, int *perm, T *a_kj, int ld_a_kj) {
      
      // Row permuation
      spldlt::host_laswp(m, a_kj, ld_a_kj, 1, m, perm, 1);

      // Apply L
      applyL_block(m, n, l_kk, ld_l_kk, a_kj, ld_a_kj);
   }
   
   template<typename T>
   void update_block_lu(
         int m, int n, T *a_ij, int ld_a_ij,
         int k,
         T *l_ik, int ld_l_ik,
         T *u_kj, int ld_u_kj) {
   
      host_gemm(
            OP_N, OP_N,
            m, n, k,
            -1.0,
            l_ik, ld_l_ik,
            u_kj, ld_u_kj,
            1.0,
            a_ij, ld_a_ij);

   }

} // end of namespace spldlt
