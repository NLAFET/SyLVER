#pragma once

#include "common.hxx"

namespace spldlt {

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

   template <typename T> 
   void host_axpy(const int n, const T a, const T *x, const int incx, T *y, const int incy);

   template <typename T> 
   double host_lange(spldlt::norm norm, const int m, const int n, const T *a, const int lda);

   /* _POTRF */
   template <typename T>
   int host_potrf(enum spldlt::fillmode uplo, int n, T* a, int lda);

   /* _TRSM */
   template <typename T>
   void host_trsm(enum spldlt::side side,enum spldlt::fillmode uplo,
                  enum spldlt::operation transa, enum spldlt::diagonal diag,
                  int m, int n, T alpha, const T* a, int lda, T* b, int ldb);

   /* _SYRK */
   template <typename T>
   void host_syrk(enum spldlt::fillmode uplo, enum spldlt::operation trans,
                  int n, int k, T alpha, const T* a, int lda, T beta, T* c, int ldc);

   /* _GEMM */
   template <typename T>
   void host_gemm(enum spldlt::operation transa, enum spldlt::operation transb,
                  int m, int n, int k, T alpha, const T* a, int lda, const T* b,
                  int ldb, T beta, T* c, int ldc);

   // GETRF
   template <typename T>
   int host_getrf(int m, int n, T* a, int lda, int *ipiv);
}
