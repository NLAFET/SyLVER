/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "kernels/ldlt_app.hxx"
#include "NumericFront.hxx"

#include <starpu.h>
#if defined(SPLDLT_USE_GPU)
#include <starpu_cublas_v2.h>
#endif

namespace sylver {
namespace spldlt {
namespace starpu {

   ////////////////////////////////////////////////////////////
   // updateN_block_app StarPU task

   // CUDA kernel
   template<typename T, int iblksz, typename Backup, typename IntAlloc>
   void updateN_block_app_cuda_func(void *buffers[], void *cl_arg) {

      T *d_lik = (T *)STARPU_MATRIX_GET_PTR(buffers[0]); // Get diagonal block pointer
      unsigned ld_lik = STARPU_MATRIX_GET_LD(buffers[0]); // Get leading dimensions

      T *d_ljk = (T *)STARPU_MATRIX_GET_PTR(buffers[1]); // Get diagonal block pointer
      unsigned ld_ljk = STARPU_MATRIX_GET_LD(buffers[1]); // Get leading dimensions

      T *d_lij = (T *)STARPU_MATRIX_GET_PTR(buffers[2]); // Get diagonal block pointer
      unsigned updm = STARPU_MATRIX_GET_NX(buffers[2]); // Get diagonal block pointer
      unsigned updn = STARPU_MATRIX_GET_NY(buffers[2]); // Get diagonal block pointer
      unsigned ld_lij = STARPU_MATRIX_GET_LD(buffers[2]); // Get leading dimensions

      T *d_d = (T *)STARPU_VECTOR_GET_PTR(buffers[3]);
      unsigned d_dimn = STARPU_VECTOR_GET_NX(buffers[3]);

      T *d_ld = (T *)STARPU_MATRIX_GET_PTR(buffers[4]); // Get pointer on scratch memory
      unsigned ldld = STARPU_MATRIX_GET_LD(buffers[4]); // Get leading dimensions
               
      int id = starpu_worker_get_id();

      int m, n; // node's dimensions
      int iblk; // destination block's row index
      int jblk; // destination block's column index     
      int blk; // source block's column index     

      ::spldlt::ldlt_app_internal::ColumnData<T,IntAlloc> *cdata = nullptr;
      Backup *backup = nullptr;
         
      T beta;
      T* upd = nullptr;
      int ldupd;

      std::vector<spral::ssids::cpu::Workspace> *work;
      int blksz;

      starpu_codelet_unpack_args (
            cl_arg,
            &m, &n,
            &iblk, &jblk, &blk,
            &cdata, &backup,
            &beta, &upd, &ldupd,
            &work, &blksz);

      // Calculate offset in `d` matrix to access diagonal elements.
      int idx = (*cdata)[blk].d - (*cdata)[0].d;
         
      // Number of eliminated columns in block-column `blk`
      int cnelim = (*cdata)[blk].nelim;

      // If nothing no column has been eliminated in block-column
      // `blk` then there is nothing to update. Return straight away
      // in this case.
      if (cnelim == 0) return;

      cudaStream_t stream = starpu_cuda_get_local_stream();
      cublasHandle_t handle = starpu_cublas_get_local_handle();

      ::spldlt::gpu::update_block(
            stream, handle,
            updm, updn,
            d_lij, ld_lij,
            cnelim,
            d_lik, ld_lik, 
            d_ljk, ld_ljk,
            false,
            &d_d[idx],
            d_ld, ldld);

   }

   ////////////////////////////////////////////////////////////
   // update_contrib_block_app StarPU task

   // CUDA kernel
   template <typename T, typename IntAlloc, typename PoolAlloc>
   void update_contrib_block_app_cuda_func(void *buffers[], void *cl_arg) {

      // printf("[update_contrib_block_app_gpu_func]\n");

      T *upd = (T *)STARPU_MATRIX_GET_PTR(buffers[0]);
      unsigned ldupd = STARPU_MATRIX_GET_LD(buffers[0]); // Leading dimensions
      unsigned updm = STARPU_MATRIX_GET_NX(buffers[0]);
      unsigned updn = STARPU_MATRIX_GET_NY(buffers[0]);

      T *lik = (T *)STARPU_MATRIX_GET_PTR(buffers[1]);
      unsigned ld_lik = STARPU_MATRIX_GET_LD(buffers[1]); // Leading dimensions

      T *ljk = (T *)STARPU_MATRIX_GET_PTR(buffers[2]);
      unsigned ld_ljk = STARPU_MATRIX_GET_LD(buffers[2]); // Leading dimensions

      T *d_d = (T *)STARPU_VECTOR_GET_PTR(buffers[3]);
      unsigned d_dimn = STARPU_VECTOR_GET_NX(buffers[3]);

      T *d_ld = (T *)STARPU_MATRIX_GET_PTR(buffers[4]); // Get pointer on scratch memory
      unsigned ldld = STARPU_MATRIX_GET_LD(buffers[4]); // Get leading dimensions

      ::spldlt::NumericFront<T, PoolAlloc> *node = nullptr;
      int k, i, j;
      std::vector<spral::ssids::cpu::Workspace> *workspaces;

      starpu_codelet_unpack_args(
            cl_arg, &node, &k, &i, &j, &workspaces);

         
      cudaStream_t stream = starpu_cuda_get_local_stream();
      cublasHandle_t handle = starpu_cublas_get_local_handle();
         
      ::spldlt::gpu::update_contrib_block_app
           <T, IntAlloc, PoolAlloc>(
                 stream, handle,
                 *node,
                 k, i, j,
                 lik, ld_lik,
                 ljk, ld_ljk,
                 updm, updn, upd, ldupd,
                 d_d,
                 d_ld, ldld);
   }

}}} // End of namespace sylver::spldlt::starpu
