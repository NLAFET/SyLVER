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

// Thread block size
#define BLOCK_SIZE 8

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
      
      std::cout << "[spldlt::gpu::factor] nc = " << nc << std::endl;
      
      // CuBLAS handle
      cublasHandle_t cuhandle;
      cublasCreate(&cuhandle);
      cublasSetStream(cuhandle, stream);

      for (int k = 0; k < nc; ++k) {

         std::cout << "[spldlt::gpu::factor] k = " << k << std::endl;
         // Factor kth block column
         int ofs = k*BLOCK_SIZE; // Number of eliminated columns
         int cblkm = m-ofs; // Block column height
         int cblkn = std::min(n-ofs, BLOCK_SIZE); // Block column width
         factor_bcol(stream, cblkm, cblkn, &d_a[ofs+ofs*ldda], ldda, d_info);
         // cudaStreamSynchronize(stream);

         // Update trailing submatrix
         int ofst = (k+1)*BLOCK_SIZE; // Offset to trailing submatrix
         int tblkn = n - ofst;
         std::cout << "[spldlt::gpu::factor] tblkn = " << tblkn << std::endl;
         T alpha = -1.0, beta = 1.0;
         sylver::gpu::dev_syrk(
               cuhandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, tblkn, BLOCK_SIZE, 
               &alpha, &d_a[ofst+ofs*ldda], ldda, 
               &beta, &d_a[ofst+ofst*ldda], ldda);
         // cudaStreamSynchronize(stream);

      }

      cudaStreamSynchronize(stream);
      cublasDestroy(cuhandle);

   }
   
}}} // End of sylver::spldlt::gpu
