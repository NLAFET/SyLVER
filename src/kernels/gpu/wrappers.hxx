/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author    Jonathan Hogg

#pragma once

// CuSOLVER
#include <cusolverDn.h>

namespace sylver {
namespace gpu {

   // _POTRF BufferSize
   template <typename T> 
   cusolverStatus_t dev_potrf_buffersize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T *a, int lda, int *lwork);

   // _POTRF
   template <typename T>
   cusolverStatus_t dev_potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T *a, int lda, T *work, int lwork, int *info);

   // _SYRK
   template <typename T>
   cublasStatus_t dev_syrk(
         cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, 
         int n, int k, const T *alpha, const T *a, int lda, const T *beta, T *c, int ldc);

   // _GEMM
   template <typename T>
   cublasStatus_t dev_gemm(
         cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
         int m, int n, int k, const T *alpha, const T *a, int lda,
         const T *b, int ldb, const T *beta, T *c, int ldc);
   
}}
