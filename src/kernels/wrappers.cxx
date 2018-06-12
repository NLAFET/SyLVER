#include "wrappers.hxx"

extern "C" {
   void daxpy_(const int *n, const double *a, const double *x, const int *incx, double *y, const int *incy);
   double dlange_(char *norm, int *m, int *n, const double *a, int *lda);
}

namespace spldlt {
   // _AXPY
   template <>
   void host_axpy<double>(const int n, double a, const double *x, int incx, double *y, int incy) {
      daxpy_(&n, &a, x, &incx, y, &incy);
   }

   // _LANGE
   template <>
   double host_lange<double>(spldlt::norm norm, int m, int n, const double *a, int lda){
      char fnorm;
      switch(norm) {
      case spldlt::NORM_M:
         fnorm = 'M';
         break;
      case spldlt::NORM_ONE:
         fnorm = '1';
         break;
      case spldlt::NORM_INF:
         fnorm = 'I';
         break;
      case spldlt::NORM_FRO:
         fnorm = 'F';
         break;
      }
      return dlange_(&fnorm, &m, &n, a, &lda);
   }
} // end of namespace spldlt
