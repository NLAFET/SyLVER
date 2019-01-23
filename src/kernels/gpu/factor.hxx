/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author    Florent Lopez
#pragma once

// Sylver
#include "sylver_ciface.hxx"
#include "kernels/gpu/common.hxx"
#include "kernels/gpu/wrappers.hxx"
 
// STD
#include <iostream>
#include <algorithm>
#include <string>

// CuBLAS
#include "cublas_v2.h"
#include <cusolverDn.h>

#define BLOCK_SIZE 8 // Thread block size
// #define BLOCK_SIZE 16 // Thread block size
#define OUTER_BLOCK_SIZE 128
// #define OUTER_BLOCK_SIZE 256
// #define OUTER_BLOCK_SIZE 512
// #define OUTER_BLOCK_SIZE 768

namespace sylver {
namespace spldlt {
namespace gpu {

   template<typename T>
   void factor_ll_hp(
         const cublasHandle_t cuhandle, 
         int m, // Number of rows 
         int n, // Number of columns
         T *const d_a, // Matrix pointer on device 
         int ldda, // Matrix leadind dim on device
         inform_t& inform, // Info host
         int *d_info // Info device
         );

   // Perform the Cholesky factorization of a block-column matrix size
   // m x n with m >= n and n <= TILE_SIZE
   template<typename T>
   void factor_bcol(
         const cudaStream_t stream, 
         int m, int n,
         T const *const d, int ldd,
         T *const a, int lda,
         int *const stat);


   // @brief Perform the Cholesky factorization on the GPU for a
   // matrix size m x n such that m >= n
   //
   // @param d_a Data pointer on the device
   // @param d_info Info value for factorization on the device 
   template<typename T>
   void factor_ll(
         const cublasHandle_t cuhandle, 
         int m, // Number of rows 
         int n, // Number of columns
         T *const d_a, // Matrix pointer on device 
         int ldda, // Matrix leadind dim on device
         inform_t& inform, // Info host
         int *d_info // Info device
         ) {

      // Error handling
      std::string context = "spldlt::gpu::factor_ll";
      cudaError_t cuerr; // CUDA error
      cublasStatus_t custat; // CuBLAS status

      int const ib = BLOCK_SIZE; // Inner block size
      // int const nb = BLOCK_SIZE; // Outer block size 
      int const nb = OUTER_BLOCK_SIZE; // Outer block size         
      // Number of block columns
      int const nc = (n-1) / nb +1;
      
      // std::cout << "[spldlt::gpu::factor] nc = " << nc << std::endl;

      cudaStream_t stream; // CUDA Stream
      // Retreive CUDA stream from cuBLAS handle
      custat = cublasGetStream(cuhandle, &stream);
      sylver::gpu::cublas_check_error(custat, context, inform);
            
      // CuSOLVER
      // cusolverStatus_t cusolstat;
      // cusolverDnHandle_t cusolhandle;
      // cusolstat = cusolverDnCreate(&cusolhandle);
      // cusolverDnSetStream(cusolhandle, stream);
      // int worksz; // Workspace size
      // sylver::gpu::dev_potrf_buffersize(cusolhandle, CUBLAS_FILL_MODE_LOWER, m, d_a, ldda, &worksz);
      // T *d_work = nullptr;
      // cuerr = cudaMalloc((void**)&d_work, worksz*sizeof(T)); 

      // Workspace
      T *d_d = nullptr;
      int lddd = ib;
      cuerr = cudaMalloc((void**)&d_d, lddd*ib*sizeof(T));
      sylver::gpu::cuda_check_error(cuerr, context, inform);
      
      T alpha = -1.0, beta = 1.0;

      // We use a 2 level blocking for maximizing the performance of
      // the GEMM operaion on the GPU

      for (int kk = 0; kk < nc; ++kk) {

         int ofs = kk*nb;
         int in = std::min(n-ofs, nb);
         int inc = (in-1) / ib + 1; 
         int updm = m-ofs;
         
         // Update trailing outer block in a left-looking fashion
         if (ofs > 0){

            // std::cout << "[spldlt::gpu::factor] updm = " << updm << ", updn = " << in << ", k = " << ofs << std::endl;

            custat = sylver::gpu::dev_gemm(
                  cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
                  updm, in, ofs, &alpha,
                  &d_a[ofs], ldda,
                  &d_a[ofs], ldda,
                  &beta, &d_a[ofs+ofs*ldda], ldda);
               // cudaStreamSynchronize(stream);
            sylver::gpu::cublas_check_error(custat, context, inform);
            // cudaStreamSynchronize(stream);

         }         

         // Factor outer block
         for (int k = 0; k < inc; ++k) {
         
            // std::cout << "[spldlt::gpu::factor] k = " << k << std::endl;
            // Factor kth block column
            int iofs = k*ib; // Number of eliminated columns
            int cblkm = m-ofs-iofs; // Block column height
            int cblkn = std::min(in-iofs, ib); // Block column width

            // Copy diagonal tile into workspace d_d
            cudaMemcpy2DAsync(
                  d_d, lddd*sizeof(T),
                  &d_a[ofs+iofs+(ofs+iofs)*ldda], ldda*sizeof(T),
                  cblkn*sizeof(T), cblkn,
                  cudaMemcpyDeviceToDevice,
                  stream);
            
             factor_bcol(
                  stream, cblkm, cblkn,
                  d_d, lddd,
                  &d_a[ofs+iofs+(ofs+iofs)*ldda], ldda,
                  d_info);
            // cudaStreamSynchronize(stream);
            
            // sylver::gpu::dev_potrf(
            //       cusolhandle, CUBLAS_FILL_MODE_LOWER, cblkn, &d_a[ofs+iofs+(ofs+iofs)*ldda], ldda,
            //       d_work, worksz, d_info);

            // factor_bcol(
            //       stream, cblkn, cblkn,
            //       &d_a[ofs+iofs+(ofs+iofs)*ldda], ldda,
            //       d_info);

            // T alp = 1.0;
            // sylver::gpu::dev_trsm(
            //       cuhandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            //       cblkm-cblkn, cblkn, &alp,
            //       &d_a[ofs+iofs+(ofs+iofs)*ldda], ldda,
            //       &d_a[ofs+iofs+cblkn+(ofs+iofs)*ldda], ldda);
            
            // cudaStreamSynchronize(stream);

            // cuerr = cudaMemcpyAsync(info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
            // cudaStreamSynchronize(stream);
            // if (*info < cblkn) { // Not positive-definite
            //    std::cout << "[spldlt::gpu::factor][error] negative or null pivot" << info << std::endl;
            //    inform.flag = ERROR_NOT_POS_DEF;
            //    return;
            // }
         
            // Update trailing submatrix
            int iofst = (k+1)*ib; // Offset to trailing submatrix in outer block
            int tblkm = m-ofs-iofst; // Width of trailing submatrix in outer block
            int tblkn = in-iofst; // Width of trailing submatrix in outer block
            if (tblkn>0) {
            
               // Update trailing submatrix (inner & outer)
               // cubstat = sylver::gpu::dev_syrk(
               //       cuhandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
               //       tblkn, ib, 
               //       &alpha, &d_a[ofs+iofst+ (ofs+iofs)*ldda], ldda, 
               //       &beta,  &d_a[ofs+iofst+(ofs+iofst)*ldda], ldda);

               // Update trailing submatrix (inner & outer)
               // cubstat = sylver::gpu::dev_gemm(
               //       cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
               //       tblkn, tblkn, ib, &alpha,
               //       &d_a[ofst+ofs*ldda], ldda,
               //       &d_a[ofst+ofs*ldda], ldda,
               //       &beta, &d_a[ofst+ofst*ldda], ldda);

               // Update trailing submatrix (inner only)
               // std::cout << "[spldlt::gpu::factor] cblkm = " << cblkm << ", tblkn = " << tblkn << ", inc = " << inc << std::endl;
               custat = sylver::gpu::dev_gemm(
                     cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
                     tblkm, tblkn, ib, &alpha,
                     &d_a[ofs+iofst+ (ofs+iofs)*ldda], ldda,
                     &d_a[ofs+iofst+ (ofs+iofs)*ldda], ldda,
                     &beta, &d_a[ofs+iofst+(ofs+iofst)*ldda], ldda);
               // cudaStreamSynchronize(stream);
               sylver::gpu::cublas_check_error(custat, context, inform);
                              
               // cudaStreamSynchronize(stream);
               // std::cout << "[spldlt::gpu::factor] cubstat = " << cubstat << std::endl;
            }
         }
      }

      cuerr = cudaStreamSynchronize(stream);
      sylver::gpu::cuda_check_error(cuerr, context, inform);

      // // Cleanup memory
      // cublasDestroy(cuhandle);
      cuerr = cudaFree(d_d);
      sylver::gpu::cuda_check_error(cuerr, context, inform);

   }

   template<typename T>
   void factor_rl(
         const cublasHandle_t cuhandle, // cuBLAS handle 
         int m, // Number of rows 
         int n, // Number of columns
         T *const d_a, 
         int ldda, 
         inform_t& inform,
         int* d_inform) {

      // Error handling
      std::string context = "spldlt::gpu::factor_rl";
      cudaError_t cuerr; // CUDA error
      cublasStatus_t custat; // CuBLAS status

      int const ib = BLOCK_SIZE; // Inner block size
      // int const nb = BLOCK_SIZE; // Outer block size 
      int const nb = OUTER_BLOCK_SIZE; // Outer block size         
      // Number of block columns
      int const nc = (n-1) / nb +1;

      cudaStream_t stream; // CUDA Stream
      // Retreive CUDA stream from cuBLAS handle
      custat = cublasGetStream(cuhandle, &stream);
      sylver::gpu::cublas_check_error(custat, context, inform);
      
      // Workspace
      T *d_d = nullptr;
      int lddd = ib;
      cuerr = cudaMalloc((void**)&d_d, lddd*ib*sizeof(T));
      sylver::gpu::cuda_check_error(cuerr, context, inform);

      T alpha = -1.0, beta = 1.0;

      // We use a 2 level blocking for maximizing the performance of
      // the GEMM operaion on the GPU
      for (int kk = 0; kk < nc; ++kk) {

         int ofs = kk*nb;
         int in = std::min(n-ofs, nb);
         int inc = (in-1) / ib + 1; 

         // Factor outer block
         for (int k = 0; k < inc; ++k) {

            // Factor kth block column
            int iofs = k*ib; // Number of eliminated columns
            int cblkm = m-ofs-iofs; // Block column height
            int cblkn = std::min(in-iofs, ib); // Block column width

            // Copy diagonal tile into workspace d_d
            cuerr = cudaMemcpy2DAsync(
                  d_d, lddd*sizeof(T),
                  &d_a[ofs+iofs+(ofs+iofs)*ldda], ldda*sizeof(T),
                  cblkn*sizeof(T), cblkn,
                  cudaMemcpyDeviceToDevice,
                  stream);
            sylver::gpu::cuda_check_error(cuerr, context, inform);
            
            factor_bcol(
                  stream, cblkm, cblkn,
                  d_d, lddd,
                  &d_a[ofs+iofs+(ofs+iofs)*ldda], ldda,
                  d_inform);

            // Update trailing submatrix
            int iofst = (k+1)*ib; // Offset to trailing submatrix in outer block
            int itblkm = m-ofs-iofst; // Width of trailing submatrix in outer block
            int itblkn = in-iofst; // Width of trailing submatrix in outer block
            if (itblkn>0) {
               custat = sylver::gpu::dev_gemm(
                     cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
                     itblkm, itblkn, ib, &alpha,
                     &d_a[ofs+iofst+ (ofs+iofs)*ldda], ldda,
                     &d_a[ofs+iofst+ (ofs+iofs)*ldda], ldda,
                     &beta, &d_a[ofs+iofst+(ofs+iofst)*ldda], ldda);
               // cudaStreamSynchronize(stream);
               sylver::gpu::cublas_check_error(custat, context, inform);
               
            }
         }

         int ofst = (kk+1)*nb;
         int tm = m-ofst; // Width of trailing submatrix
         int tn = n-ofst; // Width of trailing submatrix
         if (tn>0) {
            int tnc = (tn-1) / nb +1; // Number of blocks in the trailing submatrix 
            for (int jj = 0; jj < tnc; ++jj) {
               int tblko = jj*nb; 
               int tblkm = tm - tblko;
               int tblkn = std::min(tn, nb);
               // std::cout << "tblko = " << tblko << ", tblkm = " << tblkm << ", tblkn = " << tblkn << std::endl;
               custat = sylver::gpu::dev_gemm(
                     cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
                     tblkm, tblkn, nb, &alpha,
                     &d_a[ofst+tblko+ ofs*ldda], ldda,
                     &d_a[ofst+tblko+ ofs*ldda], ldda,
                     &beta,
                     &d_a[ofst+tblko+(ofst+tblko)*ldda], ldda);
               // cudaStreamSynchronize(stream);
               sylver::gpu::cublas_check_error(custat, context, inform);
               
            }
         }
      }
   }
   
}}} // End of sylver::spldlt::gpu
