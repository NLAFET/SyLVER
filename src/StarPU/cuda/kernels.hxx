/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

#include <starpu.h>
#if defined(SPLDLT_USE_GPU)
#include <starpu_cublas_v2.h>
#endif

namespace sylver {
namespace spldlt {
namespace starpu {

   // GPU task
   template<typename T>
   void update_block_gpu_func(void *buffers[], void *cl_arg) {

      // Get pointer on L_ij block
      T *lij = (T*)STARPU_MATRIX_GET_PTR(buffers[0]);
      unsigned m = STARPU_MATRIX_GET_NX(buffers[0]);
      unsigned n = STARPU_MATRIX_GET_NY(buffers[0]);
      unsigned ld_lij = STARPU_MATRIX_GET_LD(buffers[0]);
                  
      // Get pointer on L_ik block
      T *lik = (T*)STARPU_MATRIX_GET_PTR(buffers[1]);
      unsigned k = STARPU_MATRIX_GET_NY(buffers[1]);
      unsigned ld_lik = STARPU_MATRIX_GET_LD(buffers[1]); 

      // Get pointer on L_jk block
      T *ljk = (T*)STARPU_MATRIX_GET_PTR(buffers[2]);
      unsigned ld_ljk = STARPU_MATRIX_GET_LD(buffers[2]); 
        
      T ralpha = -1.0;
      T rbeta = 1.0;

      cublasHandle_t handle = starpu_cublas_get_local_handle();
         
      sylver::gpu::dev_gemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            m, n, k,
            &ralpha, 
            lik, ld_lik, ljk, ld_ljk,
            &rbeta,
            lij, ld_lij);
         
   }

}}} // End of namespace sylver::spldlt::starpu
