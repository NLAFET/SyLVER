/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author    Florent Lopez

// Sylver
#include "sylver_ciface.hxx"
#include "kernels/gpu/wrappers.hxx"
 
// STD
#include <iostream>
#include <algorithm>

// CuBLAS
#include "cublas_v2.h"
#include <cusolverDn.h>

#define BLOCK_SIZE 8 // Thread block size
// #define BLOCK_SIZE 16 // Thread block size
// #define OUTER_BLOCK_SIZE 256
#define OUTER_BLOCK_SIZE 128

namespace sylver {
namespace spldlt {
namespace gpu {

   // Perform the Cholesky factorization of a block-column matrix size
   // m x n with m >= n and n <= TILE_SIZE
   template<typename T>
   void factor_bcol(
         const cudaStream_t stream, int m, int n, T *const a, int lda, int *const stat);


   // @brief Perform the Cholesky factorization on the GPU for a
   // matrix size m x n such that m >= n
   //
   // @param d_a Data pointer on the device
   // @param d_info Info value for factorization on the device 
   template<typename T>
   void factor(
         const cudaStream_t stream, int m, int n, T *const d_a, int ldda, inform_t& inform) {

      int const ib = BLOCK_SIZE; // Inner block size
      int const nb = OUTER_BLOCK_SIZE; // Outer block size 
         
      // Number of block columns
      int const nc = (n-1) / nb +1;
      
      // std::cout << "[spldlt::gpu::factor] nc = " << nc << std::endl;

      cudaError_t cuerr;

      // CuBLAS handle
      cublasHandle_t cuhandle;
      cublasCreate(&cuhandle);
      cublasSetStream(cuhandle, stream);

      // CuSOLVER
      cusolverStatus_t cusolstat;
      cusolverDnHandle_t cusolhandle;
      cusolstat = cusolverDnCreate(&cusolhandle);
      cusolverDnSetStream(cusolhandle, stream);
      int worksz; // Workspace size
      sylver::gpu::dev_potrf_buffersize(cusolhandle, CUBLAS_FILL_MODE_LOWER, m, d_a, ldda, &worksz);
      T *d_work = nullptr;
      cuerr = cudaMalloc((void**)&d_work, worksz*sizeof(T)); 

      int *info;
      int *d_info;
      cudaMallocHost((void**)&info, sizeof(int)); // Page-locked memory on host side
      cudaMalloc((void**)&d_info, sizeof(int));

      T alpha = -1.0, beta = 1.0;

      for (int kk = 0; kk < nc; ++kk) {

         int ofs = kk*nb;
         int in = std::min(n-ofs, nb);
         int inc = (in-1) / ib + 1; 
         int updm = m-ofs;
         
         // Update trailing outer block in a left-looking fashion
         if (ofs > 0){

            // std::cout << "[spldlt::gpu::factor] updm = " << updm << ", updn = " << in << ", k = " << ofs << std::endl;

            sylver::gpu::dev_gemm(
                  cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
                  updm, in, ofs, &alpha,
                  &d_a[ofs], ldda,
                  &d_a[ofs], ldda,
                  &beta, &d_a[ofs+ofs*ldda], ldda);         
            // cudaStreamSynchronize(stream);

         }         

         for (int k = 0; k < inc; ++k) {
         
            // std::cout << "[spldlt::gpu::factor] k = " << k << std::endl;
            // Factor kth block column
            int iofs = k*ib; // Number of eliminated columns
            int cblkm = m-ofs-iofs; // Block column height
            int cblkn = std::min(in-iofs, ib); // Block column width

            factor_bcol(
                  stream, cblkm, cblkn,
                  &d_a[ofs+iofs+(ofs+iofs)*ldda], ldda,
                  d_info);

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
            // // std::cout << "[spldlt::gpu::factor] num_elim = " << info << std::endl;
            // if (*info < cblkn) { // Not positive-definite
            //    inform.flag = ERROR_NOT_POS_DEF;
            //    return;
            // }
         
            // Update trailing submatrix
            int iofst = (k+1)*ib; // Offset to trailing submatrix in outer block
            int tblkn = in-iofst; // Width of trailing submatrix in outer block
            if (tblkn>0) {
            
               cublasStatus_t cubstat;

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
               cubstat = sylver::gpu::dev_gemm(
                     cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
                     cblkm, tblkn, ib, &alpha,
                     &d_a[ofs+iofst+ (ofs+iofs)*ldda], ldda,
                     &d_a[ofs+iofst+ (ofs+iofs)*ldda], ldda,
                     &beta, &d_a[ofs+iofst+(ofs+iofst)*ldda], ldda);
               
               // cudaStreamSynchronize(stream);
               // std::cout << "[spldlt::gpu::factor] cubstat = " << cubstat << std::endl;
            }
         }
      }
      cudaStreamSynchronize(stream);

      // Cleanup memory
      cublasDestroy(cuhandle);
      cudaFreeHost(info);
      cudaFree(d_info);
   }
   
}}} // End of sylver::spldlt::gpu
