#pragma once

#include "ssids/cpu/cpu_iface.hxx"

namespace spldlt {
   
   /*
     m: number of row in block
     n: number of column in block
    */
   template <typename T>
   void factorize_diag_block(int m, int n, T *a, int lda) {

      int flag = lapack_potrf(FILL_MODE_LWR, n, a, lda);
      if(m > n) {
         // Diagonal block factored OK, handle some rectangular part of block
         host_trsm(SIDE_RIGHT, FILL_MODE_LWR, OP_T, DIAG_NON_UNIT,
                   m-n, n, 1.0, a, lda,
                   &a[n], lda);
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

      host_trsm(SIDE_RIGHT, FILL_MODE_LWR, OP_T, DIAG_NON_UNIT,
                m, n, 1.0, a_kk, ld_a_kk, a_ik, ld_a_ik);

   }

   /*
     A_ij <- A_ij - Aik A_jk^T
     m: number of row in A_ij block
     n: number of column in A_ij block
     k: number of column in A_ik and A_jk blocks
    */
   template <typename T>
   void update_block(int m, int n, T *a_ij, int ld_a_ij,
                     int k,
                     T *a_ik, int ld_a_ik, 
                     T *a_kj, int ld_a_kj) {
      
      host_gemm(OP_N, OP_T, m, n, k, -1.0, a_ik, ld_a_ik,
                a_kj, ld_a_kj, 1.0, a_ij, ld_a_ij);

   }

   /*
     Update block lying the diagonal
     TODO: only A_ik or A_kj needed as it is the same block
   */
   template <typename T>
   void update_diag_block(
         int m, int n, T *a_ij, int ld_a_ij,
         int k,
         T *a_ik, int ld_a_ik, 
         T *a_kj, int ld_a_kj) {
      
      host_syrk(FILL_MODE_LWR, OP_N, n, k, -1.0, 
                a_ik, ld_a_ik,
                1.0, 
                a_ij, ld_a_ij);

      if (m > n) {

         host_gemm(OP_N, OP_T, m-n, n, k, -1.0,
                   &a_ik[n], ld_a_ik,
                   a_kj, ld_a_kj, 1.0, &a_ij[n], ld_a_ij);
      }      
   }

   template <typename T>
   void expand_buffer_block(
         SymbolicSNode &snode, // symbolic source node  
         SymbolicSNode &asnode,  // symbolic destination node
         int ii, int jj, 
         int blksz,
         T *a, int lda,  
         int cptr, int rptr,
         int m, int n, T const*buffer,
         int *rlst, int *clst
         ) {

      int sa = asnode.sa; // first column in destination node

      // int *rlst = rowmap.get_ptr<int>(m); // row mapping, src to dest block 
      // int *clst = colmap.get_ptr<int>(n); // col mapping, src to dest block

      int acol = 0; // column index in ancestor
      int arow = 0; // row index in ancestor

      // build colmap
      acol = 0;
      for (int j = 0; j < n; ++j) {
         for (; sa+acol != snode.rlist[cptr+j]; acol++);
         clst[j] = acol - (jj*blksz); // compute local column index wihtin block
      }

      // build rowmap
      arow = 0;
      for (int i = 0; i < m; ++i) {
         for (; asnode.rlist[arow] != snode.rlist[rptr+i]; arow++);
         rlst[i] = arow - (ii*blksz);
      }

      // scatter buffer into destination block
      // TODO: only expand under diag
      for (int j = 0; j < n; ++j) {
         for (int i = 0; i < m; ++i) {

            arow = rlst[i];
            acol = clst[j];

            a[ acol*lda + arow ] += buffer[j*m+i];
         }
      }

   }

   template <typename T, typename PoolAlloc>
   void update_between_block(
         int n, // destination block column width
         int kk, // block column index in source node 
         int ii, int jj, // block row and block column index of A_ij block in destination node
         int blksz, // block size
         int cptr, int cptr2, // local row indexes of a_kj elements 
         // in source node
         int rptr, int rptr2, // local row indexes of a_ik elements 
         // in source node
         SymbolicSNode &snode, // symbolic source node  
         NumericNode<T, PoolAlloc> &node, // numeric source node
         SymbolicSNode &asnode,  // symbolic destination node
         T *a_ij, int lda, // block to be updated in destination node  
         T *work, // workspace
         int *rowmap, int *colmap // workpaces for col and row mapping
         ) {
      
      int sa = asnode.sa;
      int en = asnode.en;

      int ldl = align_lda<T>(snode.nrow);
      T *lcol = node.lcol;

      int acol = 0; // column index in ancestor
      int arow = 0; // row index in ancestor

      int mr = rptr2-rptr+1; // number of rows in Aik
      int mc = cptr2-cptr+1; // number of rows in Ajk
      
      // Block on the diagonal
      if (ii == jj) {

         host_syrk(FILL_MODE_LWR, OP_N, mc, n, -1.0, 
                   &lcol[cptr + kk*blksz*ldl], ldl,
                   0.0,
                   work, mr);

         if (mr > mc) {

            host_gemm(
                  OP_N, OP_T, mr-mc, mc, n, -1.0, 
                  &lcol[rptr + mc + kk*blksz*ldl], ldl,
                  &lcol[cptr + kk*blksz*ldl], ldl,
                  0.0,
                  &work[mc], mr);

         }         
      }
      else {

         host_gemm(
               OP_N, OP_T, mr, mc, n, -1.0, 
               &lcol[rptr + kk*blksz*ldl], ldl,
               &lcol[cptr + kk*blksz*ldl], ldl,
               0.0,
               work, mr);
      }

      // if (mr > blksz) {
      //    printf("[update_between_block] mr > blksz!!!, mr: %d, blksz: %d\n", mr, blksz);
      // }

      // if (mc > blksz) {
      //    printf("[update_between_block] mc > blksz!!!, mc: %d, blksz: %d\n", mc, blksz);
      //    printf("[update_between_block] mr > blksz!!!, cptr: %d, cptr2: %d\n", cptr, cptr2);
      // }

      // expand buffer into destination block
      expand_buffer_block(
            snode, // symbolic source node
            asnode,  // symbolic destination node
            // a, lda,// numeric destination node
            ii, jj, blksz,
            a_ij, lda, // numeric destination node
            cptr, rptr,
            mr, mc, 
            work, rowmap, colmap);
   }

} /* end of namespace spldlt */
