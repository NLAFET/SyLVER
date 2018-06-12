#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace spldlt {

   __global__ void
   cu_calc_ld(
         double *const d) {

   }
   
   extern "C" {
      
      void calc_ld(
            const cudaStream_t stream,
            double *const d,
            ) {

      }
      
   }

}
