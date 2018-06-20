/// \file
/// \copyright 2016- The Science and Technology Facilities Council (STFC)
/// \author    Florent Lopez
#pragma once

// cuBLAS
#include "cublas_v2.h"

namespace spldlt { namespace gpu {

      // template<typename T>
      void calc_ld(
            const cudaStream_t stream,
            int m, int n,
            double *const l, int ldl,
            double *const d,
            double *ld, int ldld
            );

      /// @bried Perform the update operation of a block on the GPU:
      /// A_{ij} = A_{ij} - L_{ik} D_k L_{jk}^T
      template<typename T>
      void update_block(
            const cudaStream_t stream, const cublasHandle_t handle,
            int m, int n,
            T *d_upd, int d_ldupd,
            int k,
            T * const d_lik, int d_ld_lik, 
            T * const d_ljk, int d_ld_ljk,
            bool zero_upd,
            T * const d_d, // Diagonal for block-column k
            T *d_ld, int d_ldld  // Workspace
            ) {
         
         T ralpha = -1.0;         
         T rbeta = zero_upd ? 0.0 : 1.0;

         spldlt::gpu::calc_ld(
               stream, m, k,
               d_lik, d_ld_lik,
               d_d,
               d_ld, d_ldld);

         cublasDgemm(
               handle,
               CUBLAS_OP_N, CUBLAS_OP_T,
               m, n, k,
               &ralpha, 
               d_ld, d_ldld, d_ljk, d_ld_ljk,
               &rbeta,
               d_upd, d_ldupd);
      }
      
}} // End of namespace spldlt::gpu
