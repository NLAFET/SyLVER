/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

#include "kernels/gpu/factor.hxx"

#include <iostream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace /* anon */ {

   // Dynamically allocated shared memory
   extern __shared__ char SharedMemory[];

   ////////////////////////////////////////

   template
   < 
      typename ELEMENT_TYPE, 
      unsigned int TILE_SIZE, 
      unsigned int TILES 
      >
   __device__ void
   dev_init_chol_fact(
         unsigned int block,
         int nrows, // number of rows of the factorized matrix
         int ncols, // number of columns thereof
         ELEMENT_TYPE* a, // array of elements of A
         int lda, // leading dimension of a
         ELEMENT_TYPE* fs // initial L factor (shared mem)
         )
   {
      const int SIZE_X = TILES*TILE_SIZE;

      int x; // row index

      for ( int tile = 0; tile < TILES; tile++ ) {
         if ( tile ) { // load A's offdiagonal tiles into shared memory
            x = ncols + threadIdx.x + (tile - 1)*TILE_SIZE +
               (TILES - 1)*TILE_SIZE*block; // offdiagonal row index in A
            fs[threadIdx.x + tile*TILE_SIZE + SIZE_X*threadIdx.y] =
               ( x < nrows && threadIdx.y < ncols ) ? 
                     a[x + lda*threadIdx.y] : 0.0;
         }
         else { // load the diagonal (pivot) tile
            fs[threadIdx.x + SIZE_X*threadIdx.y] =
               ( threadIdx.x < ncols && threadIdx.y < ncols ) ?
                               a[threadIdx.x + lda*threadIdx.y] : 0.0;
         }
      }

   }

   template
   < 
      typename ELEMENT_TYPE, 
      unsigned int TILE_SIZE, 
      unsigned int TILES 
      >
   __device__ void
   dev_save_chol_fact(
         unsigned int block,
         int nrows, // number of rows of the factorized matrix
         int ncols, // number of columns thereof
         ELEMENT_TYPE* fs, // initial L factor (shared mem)
         ELEMENT_TYPE* f, // array of elements of L
         int ldf // leading dimension of f
         )
   {
      const int SIZE_X = TILES*TILE_SIZE;

      int x; // row index

      for ( int tile = 0; tile < TILES; tile++ ) {
         if ( tile ) { // upload the relevant elements of fs to f
            x = ncols + threadIdx.x + (tile - 1)*TILE_SIZE +
               (TILES - 1)*TILE_SIZE*block;
            if ( x < nrows && threadIdx.y < ncols )
               f[x + ldf*threadIdx.y] =
                     fs[threadIdx.x + tile*TILE_SIZE + SIZE_X*threadIdx.y];
         }
         else if ( block == 0 ) { 
            // upload to f and fd
            if ( threadIdx.x < ncols && threadIdx.y < ncols )
               f[threadIdx.x + ldf*threadIdx.y] =
                               fs[threadIdx.x + SIZE_X*threadIdx.y];
         }
      } // loop through tiles ends here
   }

   template
   < 
      typename ELEMENT_TYPE, 
      unsigned int TILE_SIZE,
      unsigned int TILES 
      >
   __device__ void
   dev_block_chol(
         int block,
         int nrows, int ncols,
         ELEMENT_TYPE* a, int lda, 
         ELEMENT_TYPE* f, int ldf, 
         int* stat
         )
   {
      const int SIZE_X = TILES*TILE_SIZE;

      int ip;
      ELEMENT_TYPE v;

      ELEMENT_TYPE *work = (ELEMENT_TYPE*)SharedMemory;

      // load A into shared memory
      dev_init_chol_fact< ELEMENT_TYPE, TILE_SIZE, TILES >
         ( block, nrows, ncols, a, lda, work );
      __syncthreads();

      for ( ip = 0; ip < ncols; ip++ ) {

         v = work[ip + SIZE_X*ip];
         if ( v <= 0.0 ) {
            if ( block == 0 && threadIdx.x == 0 && threadIdx.y == 0 )
               stat[0] = ip;
            return;
         }
         v = sqrt(v);
         __syncthreads();

         if ( threadIdx.y < TILES )
            work[threadIdx.x + TILE_SIZE*threadIdx.y + SIZE_X*ip] /= v;
         __syncthreads();
    
         if ( threadIdx.y > ip && threadIdx.y < ncols ) {
            for ( int x = threadIdx.x + TILE_SIZE; x < SIZE_X; x += TILE_SIZE )
               work[x + SIZE_X*threadIdx.y] -= 
                  work[threadIdx.y + SIZE_X*ip]*work[x + SIZE_X*ip];
            if ( threadIdx.x > ip )
               work[threadIdx.x + SIZE_X*threadIdx.y]
                  -= work[threadIdx.y + SIZE_X*ip]
                  *work[threadIdx.x + SIZE_X*ip];
         }
         __syncthreads();

      }
      if ( block == 0 && threadIdx.x == 0 && threadIdx.y == 0 )
         stat[0] = ncols;

      // save the L factor
      dev_save_chol_fact< ELEMENT_TYPE, TILE_SIZE, TILES >
         ( block, nrows, ncols, work, f, ldf );
   }

   ////////////////////////////////////////
   
   // Load diagonal block as well as block bx into shared memory
   // workspace
   // Note: we assume that m > n
   template<typename T,
            int TILE_SIZE>
   __device__ void
   dev_block_load(
         unsigned int bx, // Block row index
         int m, // Number of rows in matrix
         int n, // Number of columns in matrix
         T *const a, // Data pointer
         int lda, // Input matrix leading dimensions
         T *const sdata // Shared memory data dimn (2*TILE_SIZE,TILE_SIZE)
         ) {

      int tx = threadIdx.x;
      int ty = threadIdx.y;

      int ld_sdata = 2*TILE_SIZE;

      // Load diagonal block A_kk
      sdata[tx + ty*ld_sdata] =
         ( (tx < n) && (ty < n)) ? // Note that m > n
         a[tx + ty*lda] : (T) 0.0;

      // Load off-diag block A_ik
      int a_x = tx + TILE_SIZE*bx; // Row index in a
      int sdata_x = tx + TILE_SIZE; // Row index in sdata
      sdata[sdata_x + ty*ld_sdata] =
         ( (a_x < m) && (ty < n)) ?
         a[a_x + ty*lda] : (T) 0.0;

   }

   // Store sub-diagonal block in shared memory into block bx in a
   template<typename T,
            int TILE_SIZE>
   __device__ void
   dev_block_store(
         unsigned int bx, // Block row index
         int m, // Number of rows in matrix
         int n, // Number of columns in matrix
         T *const sdata, // Shared memory data dimn (2*TILE_SIZE,TILE_SIZE)
         T *const a, // Data pointer
         int lda // Input matrix leading dimensions
         ) {

      int tx = threadIdx.x;
      int ty = threadIdx.y;

      int ld_sdata = 2*TILE_SIZE;

      // Store off-diag block A_ik
      int a_x = tx + TILE_SIZE*bx; // Row index in a
      int sdata_x = tx + TILE_SIZE; // Row index in sdata
      if ((a_x < m) && (ty < n))
         a[a_x + ty*lda] = sdata[sdata_x + ty*ld_sdata];

   }
  
   // Note: assume that m > n and n <= TILE_SIZE 
   template<typename T,
            int TILE_SIZE>
   __device__ void
   dev_llt_block(
         unsigned int bx,
         int m, int n,
         T *const l, int ldl,
         int *const stat // Info parameter
         ) {

      // printf("[dev_llt_block] m = %d, n = %d, lda = %d, TILE_SIZE = %d\n", m, n, ldl, TILE_SIZE);
      
      // Dynamically allocated shared memory
      // T * swork = (T*) SharedMemory; // Contains 2 tile i.e. dimensions (2*TILE_SIZE,TILE_SIZE) 
      // extern __shared__ __align__(sizeof(T)) unsigned char SharedMemory[];
      T *swork = reinterpret_cast<T*>(SharedMemory); // Contains 2 tile i.e. dimensions (2*TILE_SIZE,TILE_SIZE)
 
      int ld_swork = 2*TILE_SIZE;
      
      // Load A (A_kk and A_ik) into shared memory workspace W
      dev_block_load<T, TILE_SIZE>(bx, m, n, l, ldl, swork);
      // dev_init_chol_fact<T, TILE_SIZE, 2>(bx, m, n, l, ldl, swork);
      __syncthreads();
   
      // Block info
      // int bx = blockIdx.x;
      // int by = blockIdx.y;
      // Thread info
      int tx = threadIdx.x;
      int ty = threadIdx.y;

      // printf("[dev_llt_block] bx = %d, tx = %d, ty = %d\n", bx, tx, ty);

      // Compute cholesky factor of W in shared memory  
      for (int k = 0; k < n; ++k) {

         T d11 = swork[k+ld_swork*k];
         if ( d11 <= 0.0 ) {
            // zero or negative pivot detected , stop factorization
            // and record column index
            if ((bx == 0) && (ty == 0) && (ty == 0)) {
               // printf("[dev_llt_block] Zero or negative pivot detected\n", bx, tx, ty);
               stat[0] = k;
            }
            return;
         }

         d11 = sqrt(d11); // Compute pivot
         __syncthreads();
         
         // Apply pivot
         int idx = tx + ty*TILE_SIZE;
         if (idx < ld_swork) {
         // for (int idx = tx + ty*TILE_SIZE; idx < ld_swork; idx += TILE_SIZE*TILE_SIZE)
            swork[idx + k*ld_swork] /= d11;
         }
         __syncthreads();
         
         // Update trailing submatrix
         if ((ty > k)) {
            // Update A_kk
            if (tx > k)
               swork[tx + ty*ld_swork] -= swork[tx + k*ld_swork]*swork[ty + k*ld_swork];
            // Update A_ik
            int sdata_x = tx + TILE_SIZE; // Row index in sdata
            swork[sdata_x + ty*ld_swork] -= swork[sdata_x + k*ld_swork]*swork[ty + k*ld_swork];
         }
         __syncthreads();
         
      }

      // We manage to eliminate all columns
      if ((bx == 0) && (threadIdx.x == 0) && (threadIdx.y == 0))
         stat[0] = n;
      
      // Store W into A (A_ik)
      dev_block_store<T, TILE_SIZE>(bx, m, n, swork, l, ldl);
      // dev_save_chol_fact<T, TILE_SIZE, 2>(bx, m, n, swork, l, ldl);
      __syncthreads();

   }

   // Naive implementation not using shared memory buffer
   // template<typename T,
   //          int TILE_SIZE>
   // __device__ void
   // dev_llt_block_inplace(
   //       unsigned int bx,
   //       int m, int n,
   //       T *const l, int ldl,
   //       int *const stat // Info parameter
   //       ) {

   //    int tx = threadIdx.x;
   //    int ty = threadIdx.y;

   //    for (int k = 0; k < n; ++k) {

   //       T d11 = l[k+ldl*k];

   //       d11 = sqrt(d11); // Compute pivot
   //       __syncthreads();

   //    }
      
   // }
   
   // Perform the Cholesky factorization of a block-column matrix size
   // m x n with m >= n and n <= TILE_SIZE
   // TODO introduce input and output data buffers?
   template<typename T,
            int TILE_SIZE>
   __global__ void
   dev_llt_bcol(
         int m, int n,
         T *const l, int ldl,
         int *const stat // Info parameter
         ) {

      unsigned int bx = blockIdx.x;
      // printf("[dev_llt_bcol] bx = %d\n", bx);
      dev_llt_block<T, TILE_SIZE>(bx, m, n, l, ldl, stat);
      // dev_block_chol<T, TILE_SIZE, 2>(bx, m, n, l, ldl, l, ldl, stat);
   }
   
}

namespace sylver {
namespace spldlt {
namespace gpu {

   template<>
   void factor_bcol<float>(
         const cudaStream_t stream,
         int m, int n,
         float *const a, int lda,
         int *const stat) {

      dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
      // std::cout << "[factor] blockDim.x = " << threads.x << ", blockDim.y = " << threads.y << std::endl; 
      // dim3 grid((m + threads.x - 1) / threads.x,
                // (n + threads.y -1) / threads.y);
      dim3 grid((m + threads.x - 1) / threads.x);
      // std::cout << "[factor] gridDim.x = " << grid.x << std::endl; 

      // Calculate the size of the shared memory workspace per thread
      // blocks
      size_t smsize = 2*BLOCK_SIZE*BLOCK_SIZE*sizeof(float); // 2 tiles per blocks
      
      dev_llt_bcol
         <float, BLOCK_SIZE>
         <<<grid, threads, smsize, stream>>>
         (m, n, a, lda, stat);
   }

   template<>
   void factor_bcol<double>(
         const cudaStream_t stream,
         int m, int n,
         double *const a, int lda,
         int *const stat) {

      dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
      // dim3 grid((m + threads.x - 1) / threads.x,
      // (n + threads.y -1) / threads.y);
      dim3 grid((m + threads.x - 1) / threads.x);
      // std::cout << "[factor] gridDim.x = " << grid.x << std::endl; 

      // Calculate the size of the shared memory workspace per thread
      // blocks
      size_t smsize = 2*BLOCK_SIZE*BLOCK_SIZE*sizeof(double); // 2 tiles per blocks
      
      dev_llt_bcol
         <double, BLOCK_SIZE>
         <<<grid, threads, smsize, stream>>>
         (m, n, a, lda, stat);
   }

}}} // End of namespace sylver::spldlt::gpu
