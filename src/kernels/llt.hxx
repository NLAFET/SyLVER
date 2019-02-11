/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "kernels/common.hxx"
// STD
#include <vector>
#include <cstring>

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
   void cholesky_solve(int m, int n, T const* a, int lda, int nrhs, T* x, int ldx) {
      cholesky_solve_fwd(m, n, a, lda, nrhs, x, ldx);
      cholesky_solve_bwd(m, n, a, lda, nrhs, x, ldx);
   }

   /// @brief Preconditioned conjugate gradient
   template<typename T>
   int pcg(int m, T const* a, int lda, T const* rhs, T tol, int maxit, T* x, T& resid, int& iter) {

      iter = 0;
      std::vector<T> r(m); 
      // Copy b into r
      std::memcpy(&r[0], rhs, m*sizeof(T));
      // Compute residual r = b - Ax
      T mone = -1.0;
      T one = 1.0;
      T zero = 0.0;
      sylver::host_gemv(sylver::OP_N, m, m, mone, a, lda, x, 1, one, &r[0], 1);

      // Compute norm(b)
      T bnorm = sylver::host_nrm2(m, rhs, 1);
      // Compute norm(r)
      T rnorm = sylver::host_nrm2(m, &r[0], 1);
      // Compute relative residual
      resid = rnorm / bnorm;
      if (resid <= tol) {
         return 0; // Reached convergence
      }

      std::vector<T> z(m), p(m), q(m);
      z = r;
      // Apply precond to the residual
      // z0 = M^{-1}*r0
      cholesky_solve(m, m, a, lda, 1, &z[0], m);
      p = z;
      
      T d1, d2;
      T alpha, beta;
      while(iter <= maxit) {

         ++iter;
         
         // q = A*pj
         sylver::host_gemv(sylver::OP_N, m, m, one, a, lda, &p[0], 1, zero, &q[0], 1);         
         // Compute (rj, zj)
         d1 = host_dot(m, &r[0], 1, &z[0], 1);
         d2 = host_dot(m, &q[0], 1, &p[0], 1);
         // alpha = (rj, zj) / (Apj, pj)
         alpha = d1 / d2;

         // Update x
         // xj+1 = xj + alphaj * pj
         sylver::host_axpy(m, alpha, &p[0], 1, x, 1);
         // Update r 
         // rj+1 = rj + alphaj * q
         sylver::host_axpy(m, -alpha, &q[0], 1, &r[0], 1);
                  
         // Check for convergence
         // Compute norm(r)
         rnorm = sylver::host_nrm2(m, &r[0], 1);
         // Compute relative residual
         resid = rnorm / bnorm;
         if (resid <= tol) {
            return 0; // Reached convergence
         }

         // zj+1 = M^{-1}*rj+1
         z = r;
         cholesky_solve(m, m, a, lda, 1, &z[0], m);
         // Compute (rj+1, zj+1)
         d2 = host_dot(m, &r[0], 1, &z[0], 1);
         // beta = (rj+1, zj+1) / (rj, zj)
         beta = d2 / d1;
         // Compute p
         // pj+1 = zj+1 + beta * pj 
         sylver::host_axpy(m, beta, &p[0], 1, &z[0], 1);
         p = z;

      }

      return 1; // Failed to converge
   }
   
}}
