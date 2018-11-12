#include "kernels/common.hxx"

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
   int lapack_potrf(enum spldlt::fillmode uplo, int n, T* a, int lda);

}
