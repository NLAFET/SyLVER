/// \file
/// \copyright 2016- The Science and Technology Facilities Council (STFC)
/// \author    Florent Lopez
#pragma once

// SpDLTL
#include "NumericFront.hxx"

// SSIDS
#include "ssids/cpu/Workspace.hxx"

namespace spldlt {

   template <typename T>
   void update_contrib_block(
         int m, int n,
         T *upd, int ldupd,
         int k,
         const T * const lik, int ld_lik, 
         const T * const ljk, int ld_ljk,
         bool zero_upd,
         const T * const dk, // Diagonal
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

   /// @brief Form cotribution blocks using the the columns nelim_from
   /// to nelim_to in the factors
   template <typename T, typename PoolAlloc>
   void form_contrib(
         NumericFront<T, PoolAlloc>& node,
         spral::ssids::cpu::Workspace& work,
         int nelim_from, // First column in factors 
         int nelim_to // Last column in factors
         ) {

      int m = node.get_nrow();
      int n = node.get_ncol();
      size_t ldl = align_lda<T>(m);
      T *lcol = node.lcol;
      T *d = &lcol[n*ldl];
      int blksz = node.blksz;

      int fc = nelim_from/blksz; // First block-column
      int lc = nelim_to/blksz; // Last block-column
      int nr = node.get_nr();
      int rsa = n/blksz;            
      int ncontrib = nr-rsa;

      for (int k = fc; k <= lc; ++k) {

         int first_col = std::max(k*blksz, nelim_from); // first column in current block-column of L
         int last_col = std::min((k+1)*blksz, nelim_to); // last column in current block-column of L
         //int nelim_col = 0;
         int nelim_col = last_col-first_col+1;
         T *dk = &d[2*first_col];

         for (int j = rsa; j < nr; ++j) {

            int ljk_first_row = std::max(j*blksz, n);
            T *ljk = &lcol[first_col*ldl+ljk_first_row];
            //T *ljk = &lcol[k*blksz*ldl+j*blksz];

            for (int i = j; i < nr; ++i) {
                           
               int lik_first_row = std::max(i*blksz, n);
               T *lik = &lcol[first_col*ldl+lik_first_row];

               Tile<T, PoolAlloc>& upd = node.contrib_blocks[(j-rsa)*ncontrib+(i-rsa)];
                           
               int ldld = spral::ssids::cpu::align_lda<T>(blksz);
               T *ld = work.get_ptr<T>(blksz*ldld);

               update_contrib_block(
                     upd.m, upd.n, upd.a, upd.lda,  
                     nelim_col, lik, ldl, ljk, ldl,
                     (nelim_from==0), dk, ld, ldld);

            }
         }
      }
   }

   /// @brief Factor the failed pivots in a frontal matrix
   template <typename T, typename PoolAlloc>
   void factor_front_indef_failed(
         NumericFront<T, PoolAlloc>& node,
         spral::ssids::cpu::Workspace& work,
         const struct cpu_factor_options& options,
         spral::ssids::cpu::ThreadStats& stats) {

      int m = node.get_nrow();
      int n = node.get_ncol();
      size_t ldl = align_lda<T>(m);
      T *lcol = node.lcol;
      T *d = &lcol[n*ldl];
      int *perm = node.perm;
         
      int nelim = 0;
      
      // Record the number of columns eliminated during the first pass
      node.nelim1 = node.nelim; 

      // Try to eliminate the columns uneliminated at first pass
      if (node.nelim < n) {

         nelim = node.nelim;
         if(options.pivot_method!=PivotMethod::tpp)
            stats.not_first_pass += n-nelim;

         // Use TPP factor to eliminate the remaining columns in the following cases:
         // 1) options.pivot_method is set to tpp;
         // 2) We are at a root node;
         // 3) options.failed_pivot_method is set to tpp.
         if(m==n ||
            options.pivot_method==PivotMethod::tpp ||
            options.failed_pivot_method==FailedPivotMethod::tpp
               ) {

            T *ld = work.get_ptr<T>(m-nelim);
            node.nelim += ldlt_tpp_factor(
                  m-nelim, n-nelim, &perm[nelim], &lcol[nelim*(ldl+1)], ldl, 
                  &d[2*nelim], ld, m-nelim, options.action, options.u, options.small, 
                  nelim, &lcol[nelim], ldl);

            if(
                  (m-n>0) && // We're not at a root node
                  (node.nelim > nelim) // We've eliminated columns at second pass
                  ) {
                  
               // Compute contribution blocks
               form_contrib(node, work, nelim, node.nelim-1);
            }
            if(options.pivot_method==PivotMethod::tpp) {
               stats.not_first_pass += n - node.nelim;
            } else {
               // printf("[factor_front_indef_failed] Not second pass = %d\n", n-node.nelim);
               stats.not_second_pass += n - node.nelim;
            }

         }

      }
      // Update number of delayed columns
      node.ndelay_out = n - node.nelim;         
      stats.num_delay += node.ndelay_out;

      // if (node.nelim == 0) {
      //    printf("[factor_front_indef_failed]\n");
      // }
   }
   
} // end of namespace spldlt
