#pragma once

// SyLVER
#include "kernels/wrappers.hxx"

namespace spldlt {

   /// @brief Perform LU factorization of a block using partial pivoting
   /// @param m Block height/width
   /// @param n Number of rows/columns to eliminate 
   template<typename T>
   int factor_block_lu_pp(
         int m, int n, int *perm, T *a, int lda, T *b, int ldb) {

      int info = 0;
      
      // printf("[factor_block_lu_pp] lda = %d\n", lda);
      // printf("[factor_block_lu_pp] m = %d\n", m);
      // printf("[factor_block_lu_pp] n = %d\n", n);

      // int *tmp = new int[n];
      // Factor triangular part
      // TODO use getrf for best performance
      // info = host_getrf(n, n, a, lda, perm);
      lu_pp_factor(m, n, perm, a, lda);
      
      // printf("[factor_block_lu_pp] info = %d\n", info);
      // printf("[factor_block_lu_pp] lrperm\n");
      // for (int i=0; i < n; ++i) printf(" %d ", perm[i]);
      // printf("\n");            

      // 0-index perm
      // for (int i=0; i < n; ++i) perm[i] = perm[i]-1; 
      
      if (m > n) {
         // Apply factor U to sub-diagonal part
         spldlt::host_trsm(
               SIDE_RIGHT, FILL_MODE_UPR, OP_N, DIAG_NON_UNIT,
               m-n, n, 1.0, a, lda,
               &a[n], lda);

         // Apply factor L to right-diagonal part
         spldlt::host_trsm(
               SIDE_LEFT, FILL_MODE_LWR, OP_N, DIAG_UNIT,
               m-n, n, 1.0, a, lda, b, ldb);

      }

      return info;
   }

   /// @brief Apply row permutation to block a
   /// @param work Workspace
   template<typename T>
   void apply_rperm_block(
         int m, int n, int *rperm, T* a, int lda, T *work, int ldw) {
      
      // Copy to work permuted matrix a
      for (int c=0; c<n; ++c) {
         // Copy into lwork with permutation
         for (int r=0; r<m; ++r) {
            work[r+c*ldw] = a[rperm[r]+c*lda];
         }
      }
    
      // Copy back to a
      for (int c=0; c<n; ++c)
         for (int r=0; r<m; ++r)
            a[r+c*lda] = work[r+c*ldw];
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
   void applyL_block(
         int m, int n, T *l_kk, int ld_l_kk, int *perm, T *a_kj, int ld_a_kj,
         T *work, int ldw) {
      
      // Row permuation
      apply_rperm_block(
            m, n, perm, a_kj, ld_a_kj, work, ldw);
      
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
