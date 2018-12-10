#include "wrappers.hxx"
#include <stdexcept>

extern "C" {
   void daxpy_(const int *n, const double *a, const double *x, const int *incx, double *y, const int *incy);
   double dlange_(char *norm, int *m, int *n, const double *a, int *lda);
   void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
   void dtrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, const double *alpha, const double *a, int *lda, double *b, int *ldb);
   void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, const double* a, int* lda, const double* b, int* ldb, double *beta, double* c, int* ldc);
   void dsyrk_(char *uplo, char *trans, int *n, int *k, double *alpha, const double *a, int *lda, double *beta, double *c, int *ldc);
   void dgetrf_(int *m, int *n, double* a, int *lda, int *ipiv, int *info);
   void dlaswp_(int *n, double* a, int *lda, int *k1, int *k2, int *ipiv, const int *incx);
}

namespace spldlt {

   // _LASWP
   template<>
   void host_laswp<double>(int n, double *a, int lda, int k1, int k2, int *perm, int incx) {
      dlaswp_(&n, a, &lda, &k1, &k2, perm, &incx);
   }

   // _AXPY
   template<>
   void host_axpy<double>(int n, double a, const double *x, int incx, double *y, int incy) {
      daxpy_(&n, &a, x, &incx, y, &incy);
   }

   // _LANGE
   template<>
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

   /* _POTRF */
   template<>
   int host_potrf<double>(enum spldlt::fillmode uplo, int n, double* a, int lda) {
      char fuplo;
      switch(uplo) {
      case spldlt::FILL_MODE_LWR: fuplo = 'L'; break;
      case spldlt::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
      }
      int info;
      dpotrf_(&fuplo, &n, a, &lda, &info);
      return info;
   }

   /* _TRSM */
   template <>
   void host_trsm<double>(
         enum spldlt::side side, enum spldlt::fillmode uplo,
         enum spldlt::operation transa, enum spldlt::diagonal diag,
         int m, int n,
         double alpha, const double* a, int lda,
         double* b, int ldb) {
      char fside = (side==spldlt::SIDE_LEFT) ? 'L' : 'R';
      char fuplo = (uplo==spldlt::FILL_MODE_LWR) ? 'L' : 'U';
      char ftransa = (transa==spldlt::OP_N) ? 'N' : 'T';
      char fdiag = (diag==spldlt::DIAG_UNIT) ? 'U' : 'N';
      dtrsm_(&fside, &fuplo, &ftransa, &fdiag, &m, &n, &alpha, a, &lda, b, &ldb);
   }

   /* _GEMM */
   template <>
   void host_gemm<double>(
         enum spldlt::operation transa, enum spldlt::operation transb,
         int m, int n, int k, double alpha, const double* a, int lda,
         const double* b, int ldb, double beta, double* c, int ldc) {
      char ftransa = (transa==spldlt::OP_N) ? 'N' : 'T';
      char ftransb = (transb==spldlt::OP_N) ? 'N' : 'T';
      dgemm_(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
   }

   /* _SYRK */
   template <>
   void host_syrk<double>(
         enum spldlt::fillmode uplo, enum spldlt::operation trans,
         int n, int k, double alpha, const double* a, int lda,
         double beta, double* c, int ldc) {
      char fuplo = (uplo==spldlt::FILL_MODE_LWR) ? 'L' : 'U';
      char ftrans = (trans==spldlt::OP_N) ? 'N' : 'T';
      dsyrk_(&fuplo, &ftrans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
   }

   // GETRF
   template <>
   int host_getrf<double>(int m, int n, double* a, int lda, int *ipiv) {
      int info;
      dgetrf_(&m, &n, a, &lda, ipiv, &info);
      return info;
   }

} // end of namespace spldlt
