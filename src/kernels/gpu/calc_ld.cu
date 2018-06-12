#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define CALC_LD_NTX 8  // Number of threads x

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
            double *const d,
            ) {

      }
      
   }

}
