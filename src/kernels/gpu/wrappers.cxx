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

   // SPOTRF
   template<>
   cusolverStatus_t dev_potrf<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *a, int lda, float *work, int lwork, int *info) {
      return cusolverDnSpotrf(handle, uplo, n, a, lda, work, lwork, info);
   }
   template<>
   cusolverStatus_t dev_potrf<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *a, int lda, double *work, int lwork, int *info) {
      return cusolverDnDpotrf(handle, uplo, n, a, lda, work, lwork, info);
   }
   
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

   // SGEMM
   template<>
   cublasStatus_t dev_gemm<float>(
         cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
         int m, int n, int k, const float *alpha, const float *a, int lda,
         const float *b, int ldb, const float *beta, float *c, int ldc) {
      return cublasSgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   }
}}
