#pragma once

#include "kernels/common.hxx"

namespace spldlt {

   template <typename T>
   void swap_rc(int r1, int r2, int c1, int c2, int m, int *rperm, int *cperm, 
                T *a, int lda) {

      if ((r1==r2) && (c1==c2)) return;
      
      // Swap rperm entries
      std::swap( rperm[r1], rperm[r2] );
      // Swap cperm entries
      std::swap( cperm[c1], cperm[c2] );
      
      // Swap rows
      if (r1 != r2) {
         for (int c=0; c<m; ++c) {
            // Swap matrix entries
            std::swap( a[lda*c+r1], a[lda*c+r2]); 
         }
      }

      // Swap columns
      if (c1 != c2) {
         for (int r=0; r<m; ++r) {
            // Swap matrix entries
            std::swap( a[lda*c1+r], a[lda*c2+r]); 
         }
      }
   }

   /// @brief Simple LU with threshold partial pivoting.
   /// @param m number of rows
   /// @param k number of rows/columns to be eliminated
   /// @param u threshold parameter where 0 < u <= 1
   template <typename T>
   void lu_tpp_factor(double u, int m, int k, int* rperm, int* cperm, T *a, int lda) {
      
      assert(m >= k);
      
      int nelim = 0; // Number of eliminated variables
      int c = 0; // Candidate column

      while(c < k) {
         
         int maxidx;
         T maxc;

         // Find largest element in candidate column
         find_col_abs_max(nelim, m-1, &a[c*lda], maxidx, maxc);
         // TODO: manage zero pivots

         // Try diagonal element as pivot in candidate column
         if (fabs(a[c*lda+c]) >= u*maxc) {
            // Accept pivot and swap if necessary
            swap_rc(nelim, c, nelim, c, m, rperm, cperm, a, lda);
            
            T d = 1.0 / a[nelim*lda+nelim];
            for (int i=nelim+1; i<m; ++i)
               a[nelim*lda+i] = a[nelim*lda+i]*d;

            host_gemm(
                  OP_N, OP_N,
                  m-nelim-1, m-nelim-1, 1,
                  -1.0,
                  &a[nelim*lda+nelim+1], lda,
                  &a[(nelim+1)*lda+nelim], lda,
                  1.0,
                  &a[(nelim+1)*lda+nelim+1], lda);

            nelim++;
            c = nelim+1;
         }
         // Try largest off-diagonal element as pivot in candidate column
         else {
            int p;
            T maxp;
            
            // Find largest element in fully-summed coefficients of
            // candidate column
            find_col_abs_max(nelim, k-1, &a[c*lda], p, maxp);
            // Try largest off-diagonal element as pivot in candidate
            // column
            if (fabs(a[c*lda+p]) >= u*maxc) {
               // Accept pivot and swap
               swap_rc(nelim, p, nelim, c, m, rperm, cperm, a, lda);

               T d = 1.0 / a[nelim*lda+nelim];
               for (int i=nelim+1; i<m; ++i)
                  a[nelim*lda+i] = a[nelim*lda+i]*d;

               host_gemm(
                     OP_N, OP_N,
                     m-nelim-1, m-nelim-1, 1,
                     -1.0,
                     &a[nelim*lda+nelim+1], lda,
                     &a[(nelim+1)*lda+nelim], lda,
                     1.0,
                     &a[(nelim+1)*lda+nelim+1], lda);

               nelim++;
               c = nelim+1;
               
            }
            else {
               // Move to next candidate column
               c++;
            }
         }
         
      }

   }

} // end of namespace spldlt
