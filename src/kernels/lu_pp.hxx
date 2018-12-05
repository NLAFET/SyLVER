#pragma once

#include <cassert>
#include <cmath>

namespace spldlt {

   /// @brief find index of max abs value in a column starting at a
   template <typename T>
   int find_col_abs_max(int from, int to, T const* a) {
      
      assert(from>=0);
      assert(to>=from);

      T maxval = fabs(a[from]); 
      int maxidx = from;

      for (int idx=from; idx <= to; ++idx) {
         if (fabs(a[idx]) > maxval) {
            maxval = fabs(a[idx]);
            maxidx = idx;
         }
      }

      return maxidx;
   }

   /// @brief Permute rows r1 and r2
   /// @param m matrix order
   template <typename T>
   void permute_rows(int r1, int r2, int m, int *perm, T *a, int lda) {

      if (r1==r2) return;
      
      assert(r1 < m);
      assert(r2 < m);
      
      // Swap perm entries
      std::swap( perm[r1], perm[r2] );
      
      for (int c=0; c<m; ++c) {
         // Swap matrix entries
         std::swap( a[lda*c+r1], a[lda*c+r2]); 
      }
   }
   
   /// @param m number of rows
   /// @param n number of columns
   template <typename T>
   void lu_pp_factor(int m, int nelim, int* perm, T *a, int lda) {

      // Only consider case where there are more rows than columns
      assert(m >= n);

      for (int k=0; k<nelim; ++k) {

         // Find row with largest entry
         int idx = find_col_abs_max(k, m-1, &a[lda*k+k]);
         // Permute rows (LAPACK style)
         if(idx != k) permute_rows(k, idx, m, perm, a, lda);
         // printf("[lu_pp_factor] k = %d, idx = %d\n", k, idx);
         T d = 1.0 / a[k*lda+k];

         for (int i=k+1; i<m; ++i)
            a[k*lda+i] = a[k*lda+i]*d;

         host_gemm(
               OP_N, OP_N,
               m-k-1, m-k-1, 1,
               -1.0,
               &a[k*lda+k+1], lda,
               &a[(k+1)*lda+k], lda,
               1.0,
               &a[(k+1)*lda+k+1], lda);

      }
      
   }
} // end of namespace spldlt
