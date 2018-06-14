#include<stdio.h>
#include <limits>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// #define BLOCK_SIZE 128  // Number of threads
// #define BLOCK_SIZE 16  // Number of threads
#define BLOCK_SIZE 32  // Number of threads

namespace /* anon */ {

   template<typename T>
   __global__ void
   cu_calc_ld(
         int m, int n,
         T *const l, int ldl,
         T *const d,
         double *ld, int ldld) {

      int bx = blockIdx.x;
      int by = blockIdx.y;

      int tx = threadIdx.x;
      int ty = threadIdx.y;

      for (int col = ty + by * blockDim.y; col < n; col += blockDim.y * gridDim.y) {

         // if (tx == 0) {
         //    printf("[cu_calc_ld] col = %d, d[2*col] = %f\n", col, d[2*col]);
         // }
         
         // Check if we are halfway trhough a 2x2 pivot
         // if (col+1 < n && !std::isfinite(d[2*col]))
         //    continue;
         
         if((col+1==n || std::isfinite(d[2*col+2])) && std::isfinite(d[2*col])) {
            // 1x1 pivot

            // printf("[cu_calc_ld] 1x1 ty = %d\n", ty);
            // printf("[cu_calc_ld] tx = %d, ty = %d\n", tx, ty);
            
            T d11 = d[2*col];
            if(d11 != 0.0) d11 = 1/d11;

            for(int row = tx + bx * blockDim.x; row < m; row += blockDim.x * gridDim.x)
               ld[col*ldld+row] = d11 * l[col*ldl+row];

         }
         // else {
         else if (std::isfinite(d[2*col])) {
         // else if (d[2*col]==std::numeric_limits<T>::infinity()) {
            // 2x2 pivot

            // if (tx == 0)
            // printf("[cu_calc_ld] TETETETETTET ty = %d\n", ty);
            // printf("[cu_calc_ld] tx = %d, ty = %d\n", tx, ty);
            // printf("[cu_calc_ld] col = %d\n");

            T d11 = d[2*col];
            T d21 = d[2*col+1];
            T d22 = d[2*col+3];

            // printf("[cu_calc_ld] d11 = %f, d21 = %f, d22 = %f\n", d11, d21, d22);

            T det = d11*d22 - d21*d21;
            d11 = d11/det;
            d21 = d21/det;
            d22 = d22/det;
            for(int row = tx + bx * blockDim.x; row < m; row += blockDim.x * gridDim.x) {
               T a1, a2;
               a1 = l[col*ldl+row];
               a2 = l[(col+1)*ldl+row];
               ld[col*ldld+row]     =  d22*a1 - d21*a2;
               ld[(col+1)*ldld+row] = -d21*a1 + d11*a2;
            }

         }
         // else {

         //    T d11 = d[2*col-2];
         //    T d21 = d[2*col-1];
         //    T d22 = d[2*col+1];

         //    // if (tx == 0) {
         //    //    printf("[cu_calc_ld] ty = %d\n", ty);
         //    //    printf("[cu_calc_ld] d11 = %f, d21 = %f, d22 = %f\n", d11, d21, d22);
         //    // }
               
         //    T det = d11*d22 - d21*d21;
         //    d11 = d11/det;
         //    d21 = d21/det;
         //    d22 = d22/det;
         //    for(int row = tx + bx * blockDim.x; row < m; row += blockDim.x * gridDim.x) {
         //       T a1, a2;
         //       a1 = l[(col-1)*ldl+row];
         //       a2 = l[col*ldl+row];
         //       // ld[(col-1)*ldld+row] =  d22*a1 - d21*a2;
         //       ld[col*ldld+row]     = -d21*a1 + d11*a2;
         //    }

         // } 
      }

   }

   // template<typename T>
   // __global__ void
   // cu_calc_ld(
   //       int m,
   //       int n,
   //       T *const l, int ldl,
   //       T *const d,
   //       double *ld, int ldld) {

   //    // printf("[cu_calc_ld] blockIdx.x = %d\n", blockIdx.x);
      
   //    for (int col = 0; col < n; ) {

   //       if(col+1==n || std::isfinite(d[2*col+2])) {
   //          // 1x1 pivot

   //          // printf("[cu_calc_ld] 1x1\n");

   //          T d11 = d[2*col];
   //          if(d11 != 0.0) d11 = 1/d11;

   //          for(int row = threadIdx.x + blockIdx.x * blockDim.x; row < m; row += blockDim.x * gridDim.x)
   //             ld[col*ldld+row] = d11 * l[col*ldl+row];

   //          col++;
   //       }
   //       else {
   //          // 2x2 pivot

   //          // printf("[cu_calc_ld] 2x2\n");

   //          T d11 = d[2*col];
   //          T d21 = d[2*col+1];
   //          T d22 = d[2*col+3];

   //          // printf("[cu_calc_ld] d11 = %f, d21 = %f, d22 = %f\n", d11, d21, d22);

   //          T det = d11*d22 - d21*d21;
   //          d11 = d11/det;
   //          d21 = d21/det;
   //          d22 = d22/det;
   //          for(int row = threadIdx.x + blockIdx.x * blockDim.x; row < m; row += blockDim.x * gridDim.x) {
   //             T a1, a2;
   //             a1 = l[col*ldl+row];
   //             a2 = l[(col+1)*ldl+row];
   //             ld[col*ldld+row]     =  d22*a1 - d21*a2;
   //             ld[(col+1)*ldld+row] = -d21*a1 + d11*a2;
   //          }
            
   //          col += 2;
   //       }
   //    }
   // }

} // end of anon namespace

namespace spldlt {
   
      void calc_ld(
            const cudaStream_t stream,
            int m, int n,
            double *const l, int ldl,
            double *const d,
            double *ld, int ldld
            ) {

         dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
         dim3 grid((m + threads.x - 1) / threads.x, (n + threads.y -1) / threads.y);

         printf("[calc_ld] m = %d, n = %d\n", m, n);
         printf("[calc_ld] thx = %d, thy = %d\n", threads.x, threads.y);
         printf("[calc_ld] grx = %d, gry = %d\n", grid.x, grid.y);

         // int nb = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
         
         cu_calc_ld
            <double>
            // <<<1, 1, 0, stream>>>
            // <<<nb, BLOCK_SIZE, 0, stream>>>
            <<<grid, threads, 0, stream>>>
            (m, n, l, ldl, d, ld, ldld);

      }
      
}
