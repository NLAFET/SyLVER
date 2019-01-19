/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author    Florent Lopez

// Sylver
#include "kernels/gpu/wrappers.hxx"
 
// STD
#include <iostream>
#include <algorithm>

// CuBLAS
#include "cublas_v2.h"
#include <cusolverDn.h>

// Thread block size
#define BLOCK_SIZE 2

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
         const cudaStream_t stream, int m, int n, T *const d_a, int ldda, int *const d_info) {

      // Number of block columns
      int const nc = (n-1) / BLOCK_SIZE + 1; 
      
      // std::cout << "[spldlt::gpu::factor] nc = " << nc << std::endl;

      cudaError_t cuerr;

      // CuBLAS handle
      cublasHandle_t cuhandle;
      cublasCreate(&cuhandle);
      cublasSetStream(cuhandle, stream);

      // CuSOLVER
      // cusolverStatus_t cusolstat;
      // cusolverDnHandle_t cusolhandle;
      // cusolstat = cusolverDnCreate(&cusolhandle);
      // cusolverDnSetStream(cusolhandle, stream);
      // int worksz; // Workspace size
      // sylver::gpu::dev_potrf_buffersize(cusolhandle, CUBLAS_FILL_MODE_LOWER, m, d_a, ldda, &worksz);
      // T *d_work = nullptr;
      // cuerr = cudaMalloc((void**)&d_work, worksz*sizeof(T)); 
   
      for (int k = 0; k < nc; ++k) {
         
         // std::cout << "[spldlt::gpu::factor] k = " << k << std::endl;
         // Factor kth block column
         int ofs = k*BLOCK_SIZE; // Number of eliminated columns
         int cblkm = m-ofs; // Block column height
         int cblkn = std::min(n-ofs, BLOCK_SIZE); // Block column width
         factor_bcol(stream, cblkm, cblkn, &d_a[ofs+ofs*ldda], ldda, d_info);
         // sylver::gpu::dev_potrf(cusolhandle, CUBLAS_FILL_MODE_LOWER, cblkn, &d_a[ofs+ofs*ldda], ldda, d_work, worksz, d_info);
         // cudaStreamSynchronize(stream);

         // Update trailing submatrix
         int ofst = (k+1)*BLOCK_SIZE; // Offset to trailing submatrix
         int tblkn = n - ofst;
         if (tblkn>0) {
            T alpha = -1.0, beta = 1.0;
            
            // T *d_alpha = nullptr, *d_beta = nullptr;
            // cudaMalloc((void**)&d_beta, sizeof(T));
            // cudaMalloc((void**)&d_alpha, sizeof(T));
            // cudaMemcpy(d_alpha, &alpha, sizeof(T), cudaMemcpyHostToDevice);
            // cudaMemcpy(d_beta, &beta, sizeof(T), cudaMemcpyHostToDevice);
            // cudaDeviceSynchronize();

            // cublasStatus_t cubstat;
            // cubstat = sylver::gpu::dev_syrk(
            //       cuhandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
            //       tblkn, BLOCK_SIZE, 
            //       d_alpha, &d_a[ofst+ ofs*ldda], ldda, 
            //       d_beta,  &d_a[ofst+ofst*ldda], ldda);
            sylver::gpu::dev_gemm(
                  cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
                  tblkn, tblkn, BLOCK_SIZE, &alpha,
                  &d_a[ofst+ofs*ldda], ldda,
                  &d_a[ofst+ofs*ldda], ldda,
                  &beta, &d_a[ofst+ofst*ldda], ldda);
               
            // cudaStreamSynchronize(stream);
            // std::cout << "[spldlt::gpu::factor] cubstat = " << cubstat << std::endl;

         }
      }

      cudaStreamSynchronize(stream);
      cublasDestroy(cuhandle);

   }
   
}}} // End of sylver::spldlt::gpu
