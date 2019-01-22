/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include <cassert>

namespace spldlt {

   /// @param m number of rows
   /// @param n number of columns
   template <typename T>
   void lu_nopiv_factor(int m, int nelim, T *a, int lda) {

      // Only consider case where there are more rows than columns
      assert(m >= nelim);
      
      for (int k=0; k<nelim; ++k) {

         T d = 1.0 / a[k*lda+k];
         
         for (int i=k+1; i<m; ++i)
            a[k*lda+i] = a[k*lda+i]*d;
         
         // for (int j=k+1; j<n; ++j) {
         //    for (int i=k+1; i<m; ++i) {
         //       a[j*lda+i] -= a[k*lda+i]*a[j*lda+k]  
         //    }
         // }
         // a(k+1:m,k+1:m) = a(k+1:m,k+1:m) - l(k+1:m,1)*u(1,k+1:m)
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

   /// @param a input matrix containing factors L in the lower
   /// triangular part and, U in the upper triangular part
   template <typename T>
   void lu_nopiv_fwd(int m, int n, T *a, int lda, int nrhs, T *x, int ldx) {

      // Solve L(1:n,:) y(1:n,:) = b (:,:)
      host_trsm(SIDE_LEFT, FILL_MODE_LWR, OP_N, DIAG_UNIT, n, nrhs, 1.0, a, lda, x, ldx);

      if (m > n) {
         // Update y(n+1:m, :) if m > n
         // y(n+1:m,:) = x(n+1:m,:) - L(n+1:m, 1:n) x(1:n)
         host_gemm(OP_N, OP_N, m-n, nrhs, n, -1.0, &a[n], lda, x, ldx, 1.0, &x[n], ldx);
      }
      
   }

   /// @param a input matrix containing factors L in the lower
   /// triangular part and, U in the upper triangular part
   template <typename T>
   void lu_nopiv_bwd(int m, int n, T *a, int lda, int nrhs, T *x, int ldx) {

      if (m > n) {
         // Update x(1:n, :) if m > n
         // x(1:n) = x(1:n) - U(1:n, n+1:m) y(n+1:m)
         host_gemm(OP_N, OP_N, n, nrhs, m-n, -1.0, &a[n*lda], lda, &x[n], ldx, 1.0, x, ldx);
      }      

      // Solve U(:,:1:n) x(1:n,:) = y (:,:)
      host_trsm(SIDE_LEFT, FILL_MODE_UPR, OP_N, DIAG_NON_UNIT, n, nrhs, 1.0, a, lda, x, ldx);

   }
   
} // end of namespace spldlt
