/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

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

   // _TRSM
   template <typename T>
   cublasStatus_t dev_trsm(
         cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
         cublasOperation_t trans, cublasDiagType_t diag,
         int m, int n,
         const T *alpha,
         const T *a, int lda,
         T *b, int ldb);
   
   // _GEQRF bufferSize
   template <typename T>
   cusolverStatus_t dev_geqrf_buffersize(
         cusolverDnHandle_t handle, int m, int n, T *a, int lda, int *lwork);
      
   // _GEQRF
   template <typename T>
   cusolverStatus_t dev_geqrf(
         cusolverDnHandle_t handle,
         int m, int n, T *a, int lda,
         T *tau, T *work, int lwork,
         int *info);

   // _ORMQR bufferSize
   template <typename T>
   cusolverStatus_t dev_ormqr_buffersize(
         cusolverDnHandle_t handle,
         cublasSideMode_t side, cublasOperation_t trans,
         int m, int n, int k,
         const T *a, int lda,
         const T *tau,
         const T *c, int ldc,
         int *lwork);

   // _ORMQR
   template <typename T>
   cusolverStatus_t dev_ormqr(
         cusolverDnHandle_t handle,
         cublasSideMode_t side, cublasOperation_t trans,
         int m, int n, int k,
         const T *a, int lda,
         const T *tau,
         T *c, int ldc,
         T *work, int lwork,
         int *dinfo);
         
}}
