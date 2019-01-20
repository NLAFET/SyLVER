/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

#include "kernels/gpu/wrappers.hxx"

// CuSOLVER
#include <cusolverDn.h>

namespace sylver {
namespace gpu {

   // SPOTRF BufferSize
   template<>
   cusolverStatus_t dev_potrf_buffersize<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *a, int lda, int *lwork) {
      return cusolverDnSpotrf_bufferSize(handle, uplo, n, a, lda, lwork);
   }
   template<>
   cusolverStatus_t dev_potrf_buffersize<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *a, int lda, int *lwork) {
      return cusolverDnDpotrf_bufferSize(handle, uplo, n, a, lda, lwork);
   }

   ////////////////////////////////////////

   // SPOTRF
   template<>
   cusolverStatus_t dev_potrf<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *a, int lda, float *work, int lwork, int *info) {
      return cusolverDnSpotrf(handle, uplo, n, a, lda, work, lwork, info);
   }
   // DPOTRF
   template<>
   cusolverStatus_t dev_potrf<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *a, int lda, double *work, int lwork, int *info) {
      return cusolverDnDpotrf(handle, uplo, n, a, lda, work, lwork, info);
   }

   ////////////////////////////////////////

   // SSYRK
   template<>
   cublasStatus_t dev_syrk<float>(
         cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, 
         int n, int k, const float *alpha, const float *a, int lda, const float *beta, float *c, int ldc) {
      return cublasSsyrk(handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
   }
   // DSYRK
   template<>
   cublasStatus_t dev_syrk<double>(
         cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, 
         int n, int k, const double *alpha, const double *a, int lda, const double *beta, double *c, int ldc) {
      return cublasDsyrk(handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
   }

   ////////////////////////////////////////

   // SGEMM
   template<>
   cublasStatus_t dev_gemm<float>(
         cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
         int m, int n, int k, const float *alpha, const float *a, int lda,
         const float *b, int ldb, const float *beta, float *c, int ldc) {
      return cublasSgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   }
   // DGEMM
   template<>
   cublasStatus_t dev_gemm<double>(
         cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
         int m, int n, int k, const double *alpha, const double *a, int lda,
         const double *b, int ldb, const double *beta, double *c, int ldc) {
      return cublasDgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   }

   ////////////////////////////////////////
   
   // STRSM
   template<>
   cublasStatus_t dev_trsm<float>(
         cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
         cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
         const float *alpha,
         const float *a, int lda,
         float *b, int ldb) {
      return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
   }
   // DTRSM
   template<>
   cublasStatus_t dev_trsm<double>(
         cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
         cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
         const double *alpha,
         const double *a, int lda,
         double *b, int ldb) {
      return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
   }

}}
