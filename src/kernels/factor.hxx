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
     m: number of row in A_ij block
     n: number of column in A_ij block
     k: number of column in A_ik and A_jk blocks
    */
   template <typename T>
   void update_block(int m, int n, T *a_ij, int ld_a_ij,
                     int k,
                     T *a_ik, int ld_a_ik, 
                     T *a_kj, int ld_a_kj) {
      
      // TODO: use syrk on diag blocks      
      host_gemm(OP_N, OP_T, m, n, k, -1.0, a_ik, ld_a_ik,
                a_kj, ld_a_kj, 1.0, a_ij, ld_a_ij);

   }

   template <typename T>
   void expand_buffer_block(
         SymbolicSNode const& snode, // symbolic source node  
         SymbolicSNode const& asnode,  // symbolic destination node
         T *a, int lda,  
         int cptr, int rptr,
         int m, int n, T const*buffer,
         Workspace& rowmap, Workspace& colmap
         ) {

      int sa = asnode.sa; // first column in destination node

      int *rlst = rowmap.get_ptr<int>(m); // row mapping, src to dest block 
      int *clst = colmap.get_ptr<int>(n); // col mapping, src to dest block

      int acol = 0; // column index in ancestor
      int arow = 0; // row index in ancestor

      // build colmap
      acol = 0;
      for (int j = 0; j < n; ++j) {
         for (; sa+acol != snode.rlist[cptr+j]; acol++);
         clst[j] = acol;
      }

      // build rowmap
      arow = 0;
      for (int i = 0; i < m; ++i) {
         for (; asnode.rlist[arow] != snode.rlist[rptr+i]; arow++);
         rlst[i] = arow;
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
         int n, // column width
         int kk, // column index in source node 
         int cptr, int cptr2, // local row indexes of a_kj elements 
         // in source node
         int rptr, int rptr2, // local row indexes of a_ik elements 
         // in source node
         SymbolicSNode const& snode, // symbolic source node  
         NumericNode<T, PoolAlloc> &node, // numeric source node
         SymbolicSNode const& asnode,  // symbolic destination node
         T *a, int lda,  
         Workspace& work, // workspace
         Workspace& rowmap, Workspace& colmap // workpaces for col and row mapping
         ) {
      
      int sa = asnode.sa;
      int en = asnode.en;

      int ldl = align_lda<T>(snode.nrow);
      T *lcol = node.lcol;

      int acol = 0; // column index in ancestor
      int arow = 0; // row index in ancestor

      int mr = rptr2-rptr+1; // number of rows in Aik
      int mc = cptr2-cptr+1; // number of rows in Ajk
      T *buffer = work.get_ptr<T>(mr*mc);

      // TODO: use syrk on diag blocks
      host_gemm(OP_N, OP_T, mr, mc, n, -1.0, 
                &lcol[rptr + kk*ldl], ldl,
                &lcol[cptr + kk*ldl], ldl,
                0.0,
                buffer, mr);

      // expand buffer into destination block
      expand_buffer_block(
            snode, // symbolic source node
            asnode,  // symbolic destination node
            a, lda,// numeric destination node
            cptr, rptr,
            mr, mc, buffer,
            rowmap, colmap);
   }


} /* end of namespace spldlt */
