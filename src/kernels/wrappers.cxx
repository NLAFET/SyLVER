/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

// SyLVER 
#include "wrappers.hxx"

// STD
#include <stdexcept>

extern "C" {
   // AXPY
   void daxpy_(const int *n, const double *a, const double *x, const int *incx, double *y, const int *incy);
   void saxpy_(const int *n, const float *a, const float *x, const int *incx, float *y, const int *incy);
   // LANGE
   double dlange_(char *norm, int *m, int *n, const double *a, int *lda);
   // POTRF
   void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
   void spotrf_(char *uplo, int *n, float  *a, int *lda, int *info);
   // TRSM
   void dtrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, const double *alpha, const double *a, int *lda, double *b, int *ldb);
   void strsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, const float  *alpha, const float  *a, int *lda, float  *b, int *ldb);
   // GEMM
   void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, const double* a, int* lda, const double* b, int* ldb, double *beta, double* c, int* ldc);
   void sgemm_(char* transa, char* transb, int* m, int* n, int* k, float* alpha, const float* a, int* lda, const float* b, int* ldb, float *beta, float* c, int* ldc);
   // SYRK
   void dsyrk_(char *uplo, char *trans, int *n, int *k, double *alpha, const double *a, int *lda, double *beta, double *c, int *ldc);
   void ssyrk_(char *uplo, char *trans, int *n, int *k, float *alpha, const float *a, int *lda, float *beta, float *c, int *ldc);
   void dgetrf_(int *m, int *n, double* a, int *lda, int *ipiv, int *info);
   // void dlaswp_(int *n, double* a, int *lda, int *k1, int *k2, int *ipiv, const int *incx);
   // GEQRF
   void sgeqrf_(int *m, int *n, float* a, int* lda, float *tau, float *work, int *lwork, int *info);
   void dgeqrf_(int *m, int *n, double* a, int* lda, double *tau, double *work, int *lwork, int *info);
   // ORMQR
   // void sormqr_(char *side, char* trans, int *m, int *n, int *k, float* a, int* lda, float *tau, float* c, int* ldc, float *work, int *lwork, int *info);
   // GEMV
   void sgemv_(char* trans, int *m, int *n, float* alpha, float const* a, int* lda, const float *x, int const* incx, float *beta, float *y, int const *incy);
   void dgemv_(char* trans, int *m, int *n, double* alpha, double const* a, int* lda, const double *x, int const* incx, double *beta, double *y, int const *incy);
   // NRM2
   float snrm2_(int *n, float const* x, int const* incx);
   double dnrm2_(int *n, double const* x, int const* incx);
   // DOT
   float sdot_(int *n, float const* x, int const* incx, float const* y, int const* incy);
   double ddot_(int *n, double const* x, int const* incx, double const* y, int const* incy);
   
}

namespace sylver {

   // SDOT
   template<>
   float host_dot<float>(int n, float const* x, int incx, float const* y, int incy) {
      return sdot_(&n, x, &incx, y, &incy);
   }
   // DDOT
   template<>
   double host_dot<double>(int n, double const* x, int incx, double const* y, int incy) {
      return ddot_(&n, x, &incx, y, &incy);
   }
       
   // SNRM2
   template<>
   float host_nrm2<float>(int n, float const* x, int incx) {
      return snrm2_(&n, x, &incx);
   }
   // DNRM2
   template<>
   double host_nrm2<double>(int n, double const* x, int incx) {
      return dnrm2_(&n, x, &incx);
   }
   
   // SGEMV
   template<>
   void host_gemv<float>(enum sylver::operation trans, int m, int n, float alpha, float const* a, int lda,
                    float const* x, int incx, float beta, float *y, int incy) {
      char ftrans = (trans==sylver::OP_N) ? 'N' : 'T';
      sgemv_(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
   }
   // DGEMV
   template<>
   void host_gemv<double>(enum sylver::operation trans, int m, int n, double alpha, double const* a, int lda,
                    double const* x, int incx, double beta, double *y, int incy) {
      char ftrans = (trans==sylver::OP_N) ? 'N' : 'T';
      dgemv_(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
   }
   
   // _LASWP
   // template<>
   // void host_laswp<double>(int n, double *a, int lda, int k1, int k2, int *perm, int incx) {
   //    dlaswp_(&n, a, &lda, &k1, &k2, perm, &incx);
   // }

   // SAXPY
   template<>
   void host_axpy<float>(int n, float a, const float *x, int incx, float *y, int incy) {
      saxpy_(&n, &a, x, &incx, y, &incy);
   }
   // DAXPY
   template<>
   void host_axpy<double>(int n, double a, const double *x, int incx, double *y, int incy) {
      daxpy_(&n, &a, x, &incx, y, &incy);
   }

   // _LANGE
   template<>
   double host_lange<double>(sylver::norm norm, int m, int n, const double *a, int lda){
      char fnorm;
      switch(norm) {
      case sylver::NORM_M:
         fnorm = 'M';
         break;
      case sylver::NORM_ONE:
         fnorm = '1';
         break;
      case sylver::NORM_INF:
         fnorm = 'I';
         break;
      case sylver::NORM_FRO:
         fnorm = 'F';
         break;
      }
      return dlange_(&fnorm, &m, &n, a, &lda);
   }

   // DPOTRF
   template<>
   int host_potrf<double>(enum sylver::fillmode uplo, int n, double* a, int lda) {
      char fuplo;
      switch(uplo) {
      case sylver::FILL_MODE_LWR: fuplo = 'L'; break;
      case sylver::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
      }
      int info;
      dpotrf_(&fuplo, &n, a, &lda, &info);
      return info;
   }
   // SPOTRF
   template<>
   int host_potrf<float>(enum sylver::fillmode uplo, int n, float* a, int lda) {
      char fuplo;
      switch(uplo) {
      case sylver::FILL_MODE_LWR: fuplo = 'L'; break;
      case sylver::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
      }
      int info;
      spotrf_(&fuplo, &n, a, &lda, &info);
      return info;
   }

   /* _TRSM */
   template <>
   void host_trsm<double>(
         enum sylver::side side, enum sylver::fillmode uplo,
         enum sylver::operation transa, enum sylver::diagonal diag,
         int m, int n,
         double alpha, const double* a, int lda,
         double* b, int ldb) {
      char fside = (side==sylver::SIDE_LEFT) ? 'L' : 'R';
      char fuplo = (uplo==sylver::FILL_MODE_LWR) ? 'L' : 'U';
      char ftransa = (transa==sylver::OP_N) ? 'N' : 'T';
      char fdiag = (diag==sylver::DIAG_UNIT) ? 'U' : 'N';
      dtrsm_(&fside, &fuplo, &ftransa, &fdiag, &m, &n, &alpha, a, &lda, b, &ldb);
   }
   // STRSM
   template <>
   void host_trsm<float>(
         enum sylver::side side, enum sylver::fillmode uplo,
         enum sylver::operation transa, enum sylver::diagonal diag,
         int m, int n,
         float alpha, const float* a, int lda,
         float* b, int ldb) {
      char fside = (side==sylver::SIDE_LEFT) ? 'L' : 'R';
      char fuplo = (uplo==sylver::FILL_MODE_LWR) ? 'L' : 'U';
      char ftransa = (transa==sylver::OP_N) ? 'N' : 'T';
      char fdiag = (diag==sylver::DIAG_UNIT) ? 'U' : 'N';
      strsm_(&fside, &fuplo, &ftransa, &fdiag, &m, &n, &alpha, a, &lda, b, &ldb);
   }

   // DGEMM
   template <>
   void host_gemm<double>(
         enum sylver::operation transa, enum sylver::operation transb,
         int m, int n, int k, double alpha, const double* a, int lda,
         const double* b, int ldb, double beta, double* c, int ldc) {
      char ftransa = (transa==sylver::OP_N) ? 'N' : 'T';
      char ftransb = (transb==sylver::OP_N) ? 'N' : 'T';
      dgemm_(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
   }
   // SGEMM
   template <>
   void host_gemm<float>(
         enum sylver::operation transa, enum sylver::operation transb,
         int m, int n, int k, float alpha, const float * a, int lda,
         const float * b, int ldb, float beta, float* c, int ldc) {
      char ftransa = (transa==sylver::OP_N) ? 'N' : 'T';
      char ftransb = (transb==sylver::OP_N) ? 'N' : 'T';
      sgemm_(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
   }
   
   // DSYRK
   template <>
   void host_syrk<double>(
         enum sylver::fillmode uplo, enum sylver::operation trans,
         int n, int k, double alpha, const double* a, int lda,
         double beta, double* c, int ldc) {
      char fuplo = (uplo==sylver::FILL_MODE_LWR) ? 'L' : 'U';
      char ftrans = (trans==sylver::OP_N) ? 'N' : 'T';
      dsyrk_(&fuplo, &ftrans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
   }
   // SSYRK
   template <>
   void host_syrk<float>(
         enum sylver::fillmode uplo, enum sylver::operation trans,
         int n, int k, float alpha, const float* a, int lda,
         float beta, float* c, int ldc) {
      char fuplo = (uplo==sylver::FILL_MODE_LWR) ? 'L' : 'U';
      char ftrans = (trans==sylver::OP_N) ? 'N' : 'T';
      ssyrk_(&fuplo, &ftrans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
   }

   // DGETRF
   template <>
   int host_getrf<double>(int m, int n, double* a, int lda, int *ipiv) {
      int info;
      dgetrf_(&m, &n, a, &lda, ipiv, &info);
      return info;
   }

   // DGEQRF
   template <>
   int host_geqrf<double>(int m, int n, double *a, int lda, double *tau, double *work, int lwork) {
      int info;
      dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
      return info;
   }
   template <>
   int host_geqrf<float>(int m, int n, float *a, int lda, float *tau, float *work, int lwork) {
      int info;
      sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
      return info;
   }

   // SORMQR
   // template <>
   // int host_ormqr<float>(enum sylver::side side, enum sylver::operation trans,
   //                       int m, int n, int k, float *a, int lda, float *tau, float *c, int ldc,
   //                       float *work, int lwork) {
   //    int info;
   //    char fside = (side==sylver::SIDE_LEFT) ? 'L' : 'R';
   //    char ftrans = (trans==sylver::OP_N) ? 'N' : 'T';
   //    sormqr_(&fside, &ftrans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info);
   //    return info;
   // }

} // end of namespace sylver
