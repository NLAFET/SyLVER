#pragma once

#include <cassert>

namespace spldlt {

   /// @param m number of rows
   /// @param n number of columns
   template <typename T>
   void lu_nopiv_factor(int m, int n, T *a, int lda) {

      // Only consider case where ther are more rows than columns
      assert(m >= n);
      int nelim = n;
      
      for (int k=0; k<nelim; ++k) {

         T d = 1.0 / a[k*lda+k];
         
         for (int i=k+1; i<m; ++i)
            a[k*lda+i] = a[k*lda+i]*d;
         
         // for (int j=k+1; j<n; ++j) {
         //    for (int i=k+1; i<m; ++i) {
         //       a[j*lda+i] -= a[k*lda+i]*a[j*lda+k]  
         //    }
         // }
         host_gemm(
               OP_N, OP_T,
               m-k-1, n-k-1, 1,
               -1.0,
               &a[k*lda+k+1], lda,
               &a[(k+1)*lda+k], lda,
               1.0,
               &a[(k+1)*lda+k+1], lda);
      }
   }

   void lu_nopiv_fwd(int m, int n, T *a, int lda, int nrhs, T *x, int ldx) {
      
   }
   
} // end of namespace spldlt
