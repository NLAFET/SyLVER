/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "sylver/kernels/common.hxx"

#include <cassert>
#include <cmath>

namespace sylver {
namespace spldlt {

   /// @brief Permute rows r1 and r2 in a.
   template <typename T>
   void permute_rows(int r1, int r2, int m, T *a, int lda) {

      if (r1==r2) return;

      for (int c=0; c<m; ++c) {
         // Swap matrix entries
         std::swap( a[lda*c+r1], a[lda*c+r2]); 
      }      
   }

   /// @brief Permute rows r1 and r2 in a and update permutation
   /// vector perm.
   /// @param m matrix order.
   template <typename T>
   void permute_rows(int r1, int r2, int m, int *perm, T *a, int lda) {

      if (r1==r2) return;
      
      assert(r1 < m);
      assert(r2 < m);
      
      // Swap perm entries
      std::swap( perm[r1], perm[r2] );
      
      // Swap matrix entries
      permute_rows(r1, r2, m, a, lda);
   }
   
   /// @param m number of rows
   /// @param nelim number of rows/columns to be eliminated 
   template <typename T>
   void lu_pp_factor(int m, int nelim, int* perm, T *a, int lda) {

      assert(m >= nelim);

      for (int k=0; k<nelim; ++k) {

         // printf("[lu_pp_factor] piv = %e\n", a[k*lda+k]);
         // Find row with largest entry
         int idx = find_col_abs_max(k, m-1, &a[k*lda]);
         // Permute rows (LAPACK style)
         permute_rows(k, idx, m, perm, a, lda);

         // printf("[lu_pp_factor] pivot = %.2f\n", a[k*lda+k]);
         // printf("[lu_pp_factor] k = %d, idx = %d, piv = %d\n", k, idx, perm[k]);

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

   /// @param aleft Matrix of size m rows and nleft columns on which
   /// we apply the same row permutations as in a
   template <typename T>
   void lu_pp_factor(
         int m, int nelim, int* perm, T *a, int lda, int nleft, T *aleft, int ldleft) {

      for (int k=0; k<nelim; ++k) {

         // Find row with largest entry
         int idx = find_col_abs_max(k, m-1, &a[k*lda]);
         // Permute rows (LAPACK style)
         permute_rows(k, idx, m, perm, a, lda);
         // Permute rows in aleft
         permute_rows(k, idx, nleft, aleft, ldleft);

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

}} // End of namespace syler::spldlt
