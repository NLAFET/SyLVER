#include<stdio.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define BLOCK_SIZE 256  // Number of threads

namespace /* anon */ {

   template<typename T>
   __global__ void
   cu_calc_ld(
         int m,
         int n,
         T *const l, int ldl,
         T *const d,
         double *ld, int ldld) {

      // printf("[cu_calc_ld]\n");
      
      for (int col = 0; col < n; ) {

         if(col+1==n || std::isfinite(d[2*col+2])) {
            // 1x1 pivot

            // printf("[cu_calc_ld] 1x1\n");

            T d11 = d[2*col];
            if(d11 != 0.0) d11 = 1/d11;

            for(int row = threadIdx.x + blockIdx.x * blockDim.x; row < m; row += blockDim.x * gridDim.x)
               ld[col*ldld+row] = d11 * l[col*ldl+row];

            col++;
         }
         else {
            // 2x2 pivot

            // printf("[cu_calc_ld] 2x2\n");

            T d11 = d[2*col];
            T d21 = d[2*col+1];
            T d22 = d[2*col+3];

            // printf("[cu_calc_ld] d11 = %f, d21 = %f, d22 = %f\n", d11, d21, d22);

            T det = d11*d22 - d21*d21;
            d11 = d11/det;
            d21 = d21/det;
            d22 = d22/det;
            for(int row = threadIdx.x + blockIdx.x * blockDim.x; row < m; row += blockDim.x * gridDim.x) {
               T a1, a2;
               a1 = l[col*ldl+row];
               a2 = l[(col+1)*ldl+row];
               ld[col*ldld+row]     =  d22*a1 - d21*a2;
               ld[(col+1)*ldld+row] = -d21*a1 + d11*a2;
            }
            
            col += 2;
         }
      }
   }

}

namespace spldlt {
   
      void calc_ld(
            const cudaStream_t stream,
            int m,
            int n,
            double *const l, int ldl,
            double *const d,
            double *ld, int ldld
            ) {

         // dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

         int nb = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
         
         cu_calc_ld
            <double>
            // <<<1, 1, 0, stream>>>
            <<<nb, BLOCK_SIZE, 0, stream>>>
            (m, n, l, ldl, d, ld, ldld);

      }
      
}
