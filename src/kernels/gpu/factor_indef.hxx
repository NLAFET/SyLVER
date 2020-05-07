/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author    Florent Lopez
#pragma once

#include "NumericFront.hxx"
#include "sylver/kernels/ColumnData.hxx"

// cuBLAS
#include "cublas_v2.h"

namespace sylver {
namespace spldlt {
namespace gpu {

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

      // printf("[update_block] m = %d, n = %d, k = %d\n", m, n, k);
         
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

   template <typename T, typename IntAlloc, typename PoolAlloc>
   void update_contrib_block_app(
         const cudaStream_t stream, const cublasHandle_t handle,
         NumericFront<T, PoolAlloc>& node,
         int k, int i, int j,
         T *d_lik, int ld_lik,
         T *d_ljk, int ld_ljk,
         int updm, int updn, T *d_upd, int ld_upd, 
         T *d_d, // Diagonal for block-column k
         T *d_ld, int ldld) {

      int blksz = node.blksz();
         
      int nrow = node.nrow();
      int ncol = node.ncol();

      int ljk_first_row = std::max(0, ncol-j*blksz);
      int lik_first_row = std::max(0, ncol-i*blksz);

      sylver::ColumnData<T, IntAlloc> *cdata = node.cdata;
      int cnelim = (*cdata)[k].nelim;
      if (cnelim <= 0) return; // No factors to update in current block-column
      bool first_elim = (*cdata)[k].first_elim;
      int idx = (*cdata)[k].d - (*cdata)[0].d;

      spldlt::gpu::update_block(
            stream, handle,
            updm, updn,
            d_upd, ld_upd,
            cnelim,
            &d_lik[lik_first_row], ld_lik, 
            &d_ljk[ljk_first_row], ld_ljk,
            first_elim,
            &d_d[idx],
            d_ld, ldld);

   }
      
}}} // End of namespace sylver::spldlt::gpu
