/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "kernels/common.hxx"
#include "kernels/wrappers.hxx"

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/ThreadStats.hxx"

namespace spldlt {

   // @brief Factor subtree kernel in double precision
   extern "C" void spldlt_factor_subtree_c(
         void *akeep, 
         void *fkeep,
         int p,
         double *aval, 
         void **child_contrib, 
         struct spral::ssids::cpu::cpu_factor_options *options,
         spral::ssids::cpu::ThreadStats *stats);

   // @brief Factor subtree kernel in double precision
   template <typename T>
   void factor_subtree(
         void *akeep,
         void *fkeep,
         int p,
         T *aval, 
         void **child_contrib, 
         struct spral::ssids::cpu::cpu_factor_options *options,
         spral::ssids::cpu::ThreadStats *stats) {
      
      throw std::runtime_error("[factor_subtree] factor_subtree NOT implemented for working precision");
   }
      
   /// @param m Number of rows in block
   /// @param n Number of columns in block
   template <typename T>
   void factorize_diag_block(int m, int n, T *a, int lda) {

      int flag = sylver::host_potrf(sylver::FILL_MODE_LWR, n, a, lda);

      if(m > n) {
         // Diagonal block factored OK, handle some rectangular part of block
         sylver::host_trsm(
               sylver::SIDE_RIGHT, sylver::FILL_MODE_LWR,
               sylver::OP_T, sylver::DIAG_NON_UNIT,
               m-n, n,
               (T)1.0,
               a, lda,
               &a[n], lda);
}
   }

   
   /// @param m Number of rows in block
   /// @param n Number of column in block
   /// @param upd Contribution block
   template <typename T>
   void factorize_diag_block(
         int m, int n,
         T *a, int lda,
         T* upd, int ldupd,
         bool zero_upd) {

      int flag = sylver::host_potrf(sylver::FILL_MODE_LWR, n, a, lda);

      if(m>n) {
         // Diagonal block factored OK, handle some rectangular part
         // of block
         sylver::host_trsm(
               sylver::SIDE_RIGHT, sylver::FILL_MODE_LWR,
               sylver::OP_T, sylver::DIAG_NON_UNIT,
               m-n, n,
               (T)1.0,
               a, lda,
               &a[n], lda);
         
         if(upd) {
            T rbeta = zero_upd ? 0.0 : 1.0;            
            sylver::host_syrk(
                  sylver::FILL_MODE_LWR, sylver::OP_N, 
                  m-n, n, 
                  (T)-1.0,
                  &a[n], lda, 
                  rbeta, 
                  upd, ldupd);
         }

      }
   }

   /*
     m: number of row in sub diag block
     n: number of column in sub diag block
    */
   template <typename T>
   void solve_block(int m, int n, 
                    T *a_kk, int ld_a_kk, 
                    T *a_ik, int ld_a_ik) {

      sylver::host_trsm(
            sylver::SIDE_RIGHT, sylver::FILL_MODE_LWR,
            sylver::OP_T, sylver::DIAG_NON_UNIT,
            m, n, (T)1.0, a_kk, ld_a_kk, a_ik, ld_a_ik);

   }

   
   /// @param m Number of rows in sub diag block
   /// @param n Number of columns in sub diag block
   /// @param upd Contribution block
   template <typename T>
   void solve_block(
         int m, int n,
         T *a_kk, int ld_a_kk,
         T *a_ik, int ld_a_ik,
         T *upd, int ldupd,
         bool zero_upd,
         int blksz
         ) {

      sylver::host_trsm(
            sylver::SIDE_RIGHT, sylver::FILL_MODE_LWR,
            sylver::OP_T, sylver::DIAG_NON_UNIT,
            m, n, (T)1.0, a_kk, ld_a_kk, a_ik, ld_a_ik);

      if (n<blksz && upd) {

         T rbeta = zero_upd ? 0.0 : 1.0;

         sylver::host_gemm(
               sylver::OP_N, sylver::OP_T,
               m, blksz-n, n,
               (T)-1.0,
               a_ik, ld_a_ik,
               &a_kk[n], ld_a_kk,
               rbeta,
               upd, ldupd);
      }
   }

   
   /// @brief Perform the update of block Aij i.e.
   ///
   ///   Aij = Aij - Aik * Ajk^T
   ///
   /// @param m Number of rows in A_ij block
   /// @param n: Number of columns in A_ij block
   /// @param k: Number of columns in A_ik and A_jk blocks
   template <typename T>
   void update_block(int m, int n, T *a_ij, int ld_a_ij,
                     int k,
                     T *a_ik, int ld_a_ik, 
                     T *a_kj, int ld_a_kj) {
      
      sylver::host_gemm(
            sylver::OP_N, sylver::OP_T,
            m, n, k,
            (T)-1.0,
            a_ik, ld_a_ik,
            a_kj, ld_a_kj,
            (T)1.0,
            a_ij, ld_a_ij);

   }

   /// @brief Perform the update of block Aij i.e.
   ///
   ///   Aij = Aij - Aik * Ajk^T
   ///   
   /// @param m Number of rows in Aij block
   /// @param n Number of columns in Aij block
   /// @param k Number of columns in Aik and Ajk blocks
   /// @param upd Contribution block
   template <typename T>
   void update_block(
         int m, int n, T *a_ij, int ld_a_ij,
         int k,
         T *a_ik, int ld_a_ik, 
         T *a_kj, int ld_a_kj,
         T *upd, int ldupd,
         int updm, int updn,
         bool zero_upd,  
         int blksz
         ) {

      sylver::host_gemm(
            sylver::OP_N, sylver::OP_T,
            m, n, k, (T)-1.0, a_ik, ld_a_ik,
            a_kj, ld_a_kj, (T)1.0, a_ij, ld_a_ij);
      
      if(n<blksz && upd) {
         T rbeta = zero_upd ? 0.0 : 1.0;
         sylver::host_gemm(
               sylver::OP_N, sylver::OP_T,
               updm, updn, k, 
               (T)-1.0,
               &a_ik[m-updm], ld_a_ik, 
               &a_kj[n], ld_a_kj, 
               rbeta,
               upd, ldupd);

      }
   }

   /// @brief Perform the update of block Aij i.e.
   ///
   ///   Aij = Aij - Aik * Ajk^T
   ///   
   /// @param m Number of rows in Aij block
   /// @param n Number of columns in Aij block
   /// @param k Number of columns in Aik and Ajk blocks
   /// @param upd Contribution block
   template <typename T>
   void update_block(
         int m, int n, 
         T *upd, int ldupd,
         int k,
         T *a_ik, int ld_a_ik, 
         T *a_kj, int ld_a_kj,
         bool zero_upd) {

      T rbeta = zero_upd ? 0.0 : 1.0;

      sylver::host_gemm(
            sylver::OP_N, sylver::OP_T, 
            m, n, k,
            (T)-1.0,
            a_ik, ld_a_ik,
            a_kj, ld_a_kj, 
            rbeta, 
            upd, ldupd);
   }

   
   // @brief Update block lying the diagonal
   // TODO: only A_ik or A_kj needed as it is the same block
   template <typename T>
   void update_diag_block(
         int m, int n, T *a_ij, int ld_a_ij,
         int k,
         T *a_ik, int ld_a_ik, 
         T *a_kj, int ld_a_kj) {
      
      // Udpate triangular part of block
      sylver::host_syrk(
            sylver::FILL_MODE_LWR, sylver::OP_N,
            n, k, 
            (T)-1.0, 
            a_ik, ld_a_ik,
            (T)1.0,
            a_ij, ld_a_ij);

      // Udpate rectangular part if present
      if (m > n) {
         sylver::host_gemm(
               sylver::OP_N, sylver::OP_T,
               m-n, n, k,
               (T)-1.0,
               &a_ik[n], ld_a_ik,
               a_kj, ld_a_kj,
               (T)1.0,
               &a_ij[n], ld_a_ij);
      }      
   }

   // Following kernels are specific to the supernodal method
   
   // template <typename T>
   // void expand_buffer_block(
   //       SymbolicSNode &snode, // symbolic source node  
   //       SymbolicSNode &asnode,  // symbolic destination node
   //       int ii, int jj, 
   //       int blksz,
   //       T *a, int lda,  
   //       int cptr, int rptr,
   //       int m, int n, T const*buffer,
   //       int *rlst, int *clst
   //       ) {

   //    int sa = asnode.sa; // first column in destination node

   //    // int *rlst = rowmap.get_ptr<int>(m); // row mapping, src to dest block 
   //    // int *clst = colmap.get_ptr<int>(n); // col mapping, src to dest block

   //    int acol = 0; // column index in ancestor
   //    int arow = 0; // row index in ancestor

   //    // build colmap
   //    acol = 0;
   //    for (int j = 0; j < n; ++j) {
   //       for (; sa+acol != snode.rlist[cptr+j]; acol++);
   //       clst[j] = acol - (jj*blksz); // compute local column index wihtin block
   //    }

   //    // build rowmap
   //    arow = 0;
   //    for (int i = 0; i < m; ++i) {
   //       for (; asnode.rlist[arow] != snode.rlist[rptr+i]; arow++);
   //       rlst[i] = arow - (ii*blksz);
   //    }

   //    // scatter buffer into destination block
   //    // TODO: only expand under diag
   //    for (int j = 0; j < n; ++j) {
   //       for (int i = 0; i < m; ++i) {

   //          arow = rlst[i];
   //          acol = clst[j];

   //          a[ acol*lda + arow ] += buffer[j*m+i];
   //       }
   //    }

   // }

   // template <typename T, typename PoolAlloc>
   // void update_between_block(
   //       int n, // destination block column width
   //       int kk, // block column index in source node 
   //       int ii, int jj, // block row and block column index of A_ij block in destination node
   //       int blksz, // block size
   //       int cptr, int cptr2, // local row indexes of a_kj elements 
   //       // in source node
   //       int rptr, int rptr2, // local row indexes of a_ik elements 
   //       // in source node
   //       SymbolicSNode &snode, // symbolic source node  
   //       NumericNode<T, PoolAlloc> &node, // numeric source node
   //       SymbolicSNode &asnode,  // symbolic destination node
   //       T *a_ij, int lda, // block to be updated in destination node  
   //       T *work, // workspace
   //       int *rowmap, int *colmap // workpaces for col and row mapping
   //       ) {
      
   //    int sa = asnode.sa;
   //    int en = asnode.en;

   //    int ldl = align_lda<T>(snode.nrow);
   //    T *lcol = node.lcol;

   //    int acol = 0; // column index in ancestor
   //    int arow = 0; // row index in ancestor

   //    int mr = rptr2-rptr+1; // number of rows in Aik
   //    int mc = cptr2-cptr+1; // number of rows in Ajk
      
   //    // Block on the diagonal
   //    if (ii == jj) {

   //       host_syrk(FILL_MODE_LWR, OP_N, mc, n, -1.0, 
   //                 &lcol[cptr + kk*blksz*ldl], ldl,
   //                 0.0,
   //                 work, mr);

   //       if (mr > mc) {

   //          host_gemm(
   //                OP_N, OP_T, mr-mc, mc, n, -1.0, 
   //                &lcol[rptr + mc + kk*blksz*ldl], ldl,
   //                &lcol[cptr + kk*blksz*ldl], ldl,
   //                0.0,
   //                &work[mc], mr);

   //       }         
   //    }
   //    else {

   //       host_gemm(
   //             OP_N, OP_T, mr, mc, n, -1.0, 
   //             &lcol[rptr + kk*blksz*ldl], ldl,
   //             &lcol[cptr + kk*blksz*ldl], ldl,
   //             0.0,
   //             work, mr);
   //    }

   //    // if (mr > blksz) {
   //    //    printf("[update_between_block] mr > blksz!!!, mr: %d, blksz: %d\n", mr, blksz);
   //    // }

   //    // if (mc > blksz) {
   //    //    printf("[update_between_block] mc > blksz!!!, mc: %d, blksz: %d\n", mc, blksz);
   //    //    printf("[update_between_block] mr > blksz!!!, cptr: %d, cptr2: %d\n", cptr, cptr2);
   //    // }

   //    // expand buffer into destination block
   //    expand_buffer_block(
   //          snode, // symbolic source node
   //          asnode,  // symbolic destination node
   //          // a, lda,// numeric destination node
   //          ii, jj, blksz,
   //          a_ij, lda, // numeric destination node
   //          cptr, rptr,
   //          mr, mc, 
   //          work, rowmap, colmap);
   // }

} // End of namespace spldlt
