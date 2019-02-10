/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "kernels/common.hxx"
// STD
#include <vector>

namespace sylver {
namespace spldlt {

   // Forwards solve
   template<typename T>
   void cholesky_solve_fwd(int m, int n, T const* a, int lda, int nrhs, T* x, int ldx) {
      // if(nrhs==1) {
      //    host_trsv(FILL_MODE_LWR, OP_N, DIAG_NON_UNIT, n, a, lda, x, 1);
      //    if(m > n)
      //       gemv(OP_N, m-n, n, -1.0, &a[n], lda, x, 1, 1.0, &x[n], 1);
      // } else {
      sylver::host_trsm(sylver::SIDE_LEFT, sylver::FILL_MODE_LWR, sylver::OP_N, sylver::DIAG_NON_UNIT, n, nrhs, (T)1.0, a, lda, x, ldx);
      if(m > n)
         sylver::host_gemm(sylver::OP_N, sylver::OP_N, m-n, nrhs, n, (T)-1.0, &a[n], lda, x, ldx, (T)1.0, &x[n], ldx);
      // }
   }

   // Backwards solve
   template<typename T>
   void cholesky_solve_bwd(int m, int n, T const* a, int lda, int nrhs, T* x, int ldx) {
      // if(nrhs==1) {
      //    if(m > n)
      //       gemv(OP_T, m-n, n, -1.0, &a[n], lda, &x[n], 1, 1.0, x, 1);
      //    host_trsv(FILL_MODE_LWR, OP_T, DIAG_NON_UNIT, n, a, lda, x, 1);
      // } else {
      if(m > n)
         sylver::host_gemm(sylver::OP_T, sylver::OP_N, n, nrhs, m-n, (T)-1.0, &a[n], lda, &x[n], ldx, (T)1.0, x, ldx);
      sylver::host_trsm(sylver::SIDE_LEFT, sylver::FILL_MODE_LWR, sylver::OP_T, sylver::DIAG_NON_UNIT, n, nrhs, (T)1.0, a, lda, x, ldx);
      // }
   }

   template<typename T>
   void pcg(int m, T const* a, int lda, T const* rhs, T tol, int maxit, T* x) {

      std::vector<T> r(m); 

      
      
   }
   
}}
