#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define BLOCK_SIZE 8  // Number of threads

namespace /* anon */ {

   __global__ void
   cu_calc_ld(
         int m,
         int n,
         double *const d) {

   }

}

namespace spldlt {
   
   extern "C" {
      
      void calc_ld(
            const cudaStream_t stream,
            int m,
            int n,
            double *const d
            ) {
         
      }
      
   }

}
