/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "common.hxx"

namespace sylver {

   enum norm {
      // 
      NORM_M,
      // One norm
      NORM_ONE,
      // Infinity norm
      NORM_INF,
      // Frobenius norm
      NORM_FRO
   };

   // _LASWP
   template <typename T> 
   void host_laswp(int n, T *a, int lda, int k1, int k2, int *perm, int incx);

   template <typename T> 
   void host_axpy(int n, const T a, const T *x, const int incx, T *y, const int incy);

   template <typename T> 
   double host_lange(sylver::norm norm, const int m, const int n, const T *a, const int lda);

   /* _POTRF */
   template <typename T>
   int host_potrf(enum sylver::fillmode uplo, int n, T* a, int lda);

   /* _TRSM */
   template <typename T>
   void host_trsm(enum sylver::side side, enum sylver::fillmode uplo,
                  enum sylver::operation transa, enum sylver::diagonal diag,
                  int m, int n, T alpha, const T* a, int lda, T* b, int ldb);

   /* _SYRK */
   template <typename T>
   void host_syrk(enum sylver::fillmode uplo, enum sylver::operation trans,
                  int n, int k, T alpha, const T* a, int lda, T beta, T* c, int ldc);

   /* _GEMM */
   template <typename T>
   void host_gemm(enum sylver::operation transa, enum sylver::operation transb,
                  int m, int n, int k, T alpha, const T* a, int lda, const T* b,
                  int ldb, T beta, T* c, int ldc);

   // GETRF
   template <typename T>
   int host_getrf(int m, int n, T* a, int lda, int *ipiv);


   // GEQRF
   template <typename T>
   int host_geqrf(int m, int n, T *a, int lda, T *tau, T *work, int lwork);

   // ORMQR
   template <typename T>
   int host_ormqr(enum sylver::side side, enum sylver::operation trans,
                  int m, int n, int k, T *a, int lda, T *tau, T *c, int ldc,
                  T *work, int lwork);

   // GEMV
   template <typename T>
   void host_gemv(enum sylver::operation trans, int m, int n, T alpha, T const* a, int lda,
             T const* x, int incx, T beta, T *y, int incy);

   // NRM2
   template <typename T>
   T host_nrm2(int n, T const* x, int incx);

   // DOT
   template <typename T>
   T host_dot(int n, T const* x, int incx, T const* y, int incy);
}
