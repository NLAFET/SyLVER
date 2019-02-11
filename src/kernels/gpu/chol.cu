/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

// SyLVER
#include "kernels/gpu/convert.cuh"
#include "kernels/gpu/factor.hxx"
//STD
#include <iostream>
#include <chrono>
// CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

// #if defined (HAVE_CUTLASS)
// #else
// #endif

namespace /* anon */ {

   // Dynamically allocated shared memory
   extern __shared__ char SharedMemory[];

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
         T const *const d, // Workspace with copy of the digonal tile
         int ldd, // Workspace leading dimension
         T *const a, // Matrix block column pointer
         int lda, // Matrix leading dimensions
         T *const sdata // Shared memory data dimn (2*TILE_SIZE,TILE_SIZE)
         ) {

      int tx = threadIdx.x;
      int ty = threadIdx.y;

      int ld_sdata = 2*TILE_SIZE;

      // Load diagonal block A_kk
      sdata[tx + ty*ld_sdata] =
         ( (tx < n) && (ty < n)) ? // Note that m > n
         d[tx + ty*ldd] : (T) 0.0;
         // a[tx + ty*lda] : (T) 0.0;

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
         T const *const d, int ldd,
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
      dev_block_load<T, TILE_SIZE>(bx, m, n, d, ldd, l, ldl, swork);
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
         if ( d11 <=  0.0 ) {
            // zero or negative pivot detected , stop factorization
            // and record column index
            if ((bx == 0) && (ty == 0) && (tx == 0)) {
               printf("[dev_llt_block] Zero or negative pivot detected, d11 = %.3e\n", d11);
               stat[0] = k;
            }
            return;
         }

         // if (isnan(d11) || isinf(d11)) {
         //    if ((bx == 0) && (ty == 0) && (tx == 0)) {
         //       printf("[dev_llt_block] NaN detected\n");
         //       printf("[dev_llt_block] m = %d, n = %d\n", m, n);
         //    }
         //    continue;
         // }

         d11 = sqrt(d11); // Compute pivot
         __syncthreads();
         
         // Apply pivot
         int idx = tx + ty*TILE_SIZE;
         if (idx < ld_swork) {
         // for (int idx = tx + ty*TILE_SIZE; idx < ld_swork; idx += TILE_SIZE*TILE_SIZE)
            swork[idx + k*ld_swork] /= d11;

            // T u = 1e-5; // Threshold
            // T aik = swork[idx + k*ld_swork];
            // if (isnan(aik) || isinf(aik)) {
            //    continue;
            //    // printf("[dev_llt_block] NaN enrty detected, aik = %.3e\n", aik);
            // }
            // else if (fabs(aik) > (1/u)) {
            //    printf("[dev_llt_block] large enrty detected, aik = %.3e\n", aik);
            // }

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

   // #if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)

   // Ugly but partial function specialization does not seems to be
   // possible. Other solution would be to specialize dev_llt_bcol()
   // for half prec.
   template<>
   __device__ void
   dev_llt_block<__half, BLOCK_SIZE>(
         unsigned int bx,
         int m, int n,
         __half const *const d, int ldd,
         __half *const l, int ldl,
         int *const stat // Info parameter
         ) {

      // No-op
      // printf("TETTETETETET\n");

      // printf("[dev_llt_block] m = %d, n = %d, lda = %d, TILE_SIZE = %d\n", m, n, ldl, TILE_SIZE);
      
      // Dynamically allocated shared memory
      // T * swork = (T*) SharedMemory; // Contains 2 tile i.e. dimensions (2*TILE_SIZE,TILE_SIZE) 
      // extern __shared__ __align__(sizeof(T)) unsigned char SharedMemory[];
      __half *swork = reinterpret_cast<__half*>(SharedMemory); // Contains 2 tile i.e. dimensions (2*TILE_SIZE,TILE_SIZE)
 
      int ld_swork = 2*BLOCK_SIZE;
      
      // Load A (A_kk and A_ik) into shared memory workspace W
      dev_block_load<__half, BLOCK_SIZE>(bx, m, n, d, ldd, l, ldl, swork);
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

         __half d11 = swork[k+ld_swork*k];
         // if ( d11 <=  0.0 ) {
         if(__hle(d11, 0.0)) {
            // zero or negative pivot detected , stop factorization
            // and record column index
            if ((bx == 0) && (ty == 0) && (tx == 0)) {
               printf("[dev_llt_block] Zero or negative pivot detected, d11 = %.3e\n", d11);
               stat[0] = k;
            }
            return;
         }

         // if (__hisnan(d11)) {
         //    if ((bx == 0) && (ty == 0) && (tx == 0)) {
         //       printf("[dev_llt_block] NaN detected\n");
         //       // printf("[dev_llt_block] m = %d, n = %d\n", m, n);
         //    }
         //    return;
         // }
         // if(__hisinf(d11)) {
         //    if ((bx == 0) && (ty == 0) && (tx == 0)) {
         //       printf("[dev_llt_block] Inf detected\n");
         //    }
         //    return;
         // }
         
         d11 = hsqrt(d11); // Compute pivot
         __syncthreads();
         
         // Apply pivot
         int idx = tx + ty*BLOCK_SIZE;
         if (idx < ld_swork) {
            // for (int idx = tx + ty*TILE_SIZE; idx < ld_swork; idx += TILE_SIZE*TILE_SIZE)
            swork[idx + k*ld_swork] /= d11;

            // T u = 1e-5; // Threshold
            // T aik = swork[idx + k*ld_swork];
            // if (isnan(aik) || isinf(aik)) {
            //    continue;
            //    // printf("[dev_llt_block] NaN enrty detected, aik = %.3e\n", aik);
            // }
            // else if (fabs(aik) > (1/u)) {
            //    printf("[dev_llt_block] large enrty detected, aik = %.3e\n", aik);
            // }

            // if(__hisinf(swork[idx + k*ld_swork])) {
            //    printf("[dev_llt_block] Inf detected\n");
            //    return;
            // }
         }
         __syncthreads();
         
         // Update trailing submatrix
         if ((ty > k)) {
            // Update A_kk
            if (tx > k) {
               swork[tx + ty*ld_swork] -= swork[tx + k*ld_swork]*swork[ty + k*ld_swork];
               // if(__hisinf(swork[tx + ty*ld_swork])) {
               //    if (bx == 0) {
               //       // printf("[dev_llt_block] Inf detected\n");
               //       printf("[dev_llt_block] Inf detected, aik = %e, ajk = %e, aij = %e, d11 = %e\n",
               //              swork[tx + k*ld_swork], swork[ty + k*ld_swork], swork[tx + ty*ld_swork], d11);
               //       printf("[dev_llt_block] Inf detected, i = %d, j = %d, k = %d\n", tx, ty, k);

               //    }
               //    return;
               // }
            }
            // Update A_ik
            int sdata_x = tx + BLOCK_SIZE; // Row index in sdata
            swork[sdata_x + ty*ld_swork] -= swork[sdata_x + k*ld_swork]*swork[ty + k*ld_swork];
         }
         __syncthreads();
         
      }

      // We manage to eliminate all columns
      if ((bx == 0) && (threadIdx.x == 0) && (threadIdx.y == 0))
         stat[0] = n;
      
      // Store W into A (A_ik)
      dev_block_store<__half, BLOCK_SIZE>(bx, m, n, swork, l, ldl);
      // dev_save_chol_fact<T, TILE_SIZE, 2>(bx, m, n, swork, l, ldl);
      __syncthreads();

   }
   
   // Perform the Cholesky factorization of a block-column matrix size
   // m x n with m >= n and n <= TILE_SIZE
   // TODO introduce input and output data buffers?
   template<typename T,
            int TILE_SIZE>
   __global__ void
   dev_llt_bcol(
         int m, int n,
         T const *const d, int ldd,
         T *const l, int ldl,
         int *const stat // Info parameter
         ) {

      unsigned int bx = blockIdx.x;
      // printf("[dev_llt_bcol] bx = %d\n", bx);
      dev_llt_block<T, TILE_SIZE>(bx, m, n, d, ldd, l, ldl, stat);
      // dev_block_chol<T, TILE_SIZE, 2>(bx, m, n, l, ldl, l, ldl, stat);
   }
   
}

namespace sylver {
namespace spldlt {
namespace gpu {

   ////////////////////////////////////////

   // template<>
   // void factor_bcol<float>(
   //       const cudaStream_t stream,
   //       int m, int n,
   //       float const *const d, int ldd,
   //       float *const a, int lda,
   //       int *const stat) {

   //    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
   //    // std::cout << "[factor] blockDim.x = " << threads.x << ", blockDim.y = " << threads.y << std::endl; 
   //    // dim3 grid((m + threads.x - 1) / threads.x,
   //              // (n + threads.y -1) / threads.y);
   //    dim3 grid((m + threads.x - 1) / threads.x);
   //    // std::cout << "[factor] gridDim.x = " << grid.x << std::endl; 

   //    // Calculate the size of the shared memory workspace per thread
   //    // blocks
   //    size_t smsize = 2*BLOCK_SIZE*BLOCK_SIZE*sizeof(float); // 2 tiles per blocks
      
   //    dev_llt_bcol
   //       <float, BLOCK_SIZE>
   //       <<<grid, threads, smsize, stream>>>
   //       (m, n, d, ldd, a, lda, stat);
   // }

   template<typename T>
   void factor_bcol(
         const cudaStream_t stream,
         int m, int n,
         T const *const d, int ldd,
         T *const a, int lda,
         int *const stat) {

      dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
      // dim3 grid((m + threads.x - 1) / threads.x,
      // (n + threads.y -1) / threads.y);
      dim3 grid((m + threads.x - 1) / threads.x);
      // std::cout << "[factor] gridDim.x = " << grid.x << std::endl; 

      // Calculate the size of the shared memory workspace per thread
      // blocks
      size_t smsize = 2*BLOCK_SIZE*BLOCK_SIZE*sizeof(T); // 2 tiles per blocks
      
      dev_llt_bcol
         <T, BLOCK_SIZE>
         <<<grid, threads, smsize, stream>>>
         (m, n, d, ldd, a, lda, stat);
   }

   // Half precision
   template void factor_bcol<sylver::gpu::half>(
         const cudaStream_t stream, int m, int n, sylver::gpu::half const *const d, int ldd,
         sylver::gpu::half *const a, int lda, int *const stat);
   // Single precision
   template void factor_bcol<float>(
         const cudaStream_t stream, int m, int n, float const *const d, int ldd,
         float *const a, int lda, int *const stat);
   // Double precision
   template void factor_bcol<double>(
         const cudaStream_t stream, int m, int n, double const *const d, int ldd,
         double *const a, int lda, int *const stat);

   ////////////////////////////////////////

   // Using compute type 32F factoring panel
   template<>
   void factor_ll_hp<float>(
         const cublasHandle_t cuhandle, 
         int m, // Number of rows 
         int n, // Number of columns
         float *const d_a, // Matrix pointer on device 
         int ldda, // Matrix leadind dim on device
         inform_t& inform, // Info host
         int *d_info // Info device
         ) {
      
      // Error handling
      std::string context = "spldlt::gpu::factor_ll_hp";
      cudaError_t cuerr; // CUDA error
      cublasStatus_t custat; // CuBLAS status
      
      int const ib = BLOCK_SIZE; // Inner block size
      // int const nb = BLOCK_SIZE; // Outer block size 
      int const nb = OUTER_BLOCK_SIZE; // Outer block size         
      // Number of block columns
      int const nc = (n-1) / nb +1;

      // Timers
      std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

      std::cout << "[" << context << "]" << std::endl;
      // std::cout << "[" << context << "]" << " nc = " << nc << std::endl;

      // Retreive CUDA stream from cuBLAS handle
      cudaStream_t stream = NULL; // CUDA Stream
      // custat = cublasSetStream(cuhandle, NULL);
      custat = cublasGetStream(cuhandle, &stream);
      sylver::gpu::cublas_check_error(custat, context, inform);

      // Set Math mode
      // custat = cublasSetMathMode(cuhandle, CUBLAS_DEFAULT_MATH);
      // custat = cublasSetMathMode(cuhandle, CUBLAS_TENSOR_OP_MATH);
      // sylver::gpu::cublas_check_error(custat, context, inform);
      cublasMath_t mode;
      cublasGetMathMode(cuhandle, &mode);
      
      // Allocate memory to accomodate half prec representation of
      // matrix A
      sylver::gpu::half *d_a_hp = nullptr;
      start = std::chrono::high_resolution_clock::now();
      cuerr = cudaMalloc((void**)&d_a_hp, m*ldda*sizeof(sylver::gpu::half));      
      sylver::gpu::cuda_check_error(cuerr, context, inform);
      end = std::chrono::high_resolution_clock::now();
      long talloc =
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (end-start).count();
      std::cout << "[" << context << "] "
                << "Time to allocate hp matrix (s) = " 
                << std::scientific << 1e-9*talloc 
                << std::endl;

      // Copy matrix into buffer and convert to half prec  
      start = std::chrono::high_resolution_clock::now();
      sylver::gpu::convert(stream, m, n, d_a, ldda, d_a_hp, ldda);
      end = std::chrono::high_resolution_clock::now();
      // Calculate time to convert
      long tconv =  
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (end-start).count();
      std::cout << "[" << context << "] "
                << "Time to convert sp to hp (s) = " << 1e-9*tconv 
                << std::endl;

      // Allocate buffer for computing the panel
      float *d_a_tmp = nullptr;
      cuerr = cudaMalloc((void**)&d_a_tmp, nb*ldda*sizeof(float));      
      sylver::gpu::cuda_check_error(cuerr, context, inform);      
      
      // Workspace holding the diagonal tile
      float *d_d = nullptr;
      int lddd = ib;
      cuerr = cudaMalloc((void**)&d_d, lddd*ib*sizeof(float));
      sylver::gpu::cuda_check_error(cuerr, context, inform);
      
      // // CuSOLVER for debug purpose
      // cusolverStatus_t cusolstat;
      // cusolverDnHandle_t cusolhandle;
      // cusolstat = cusolverDnCreate(&cusolhandle);
      // cusolstat = cusolverDnSetStream(cusolhandle, stream);
      // int worksz; // Workspace size
      // sylver::gpu::dev_potrf_buffersize(cusolhandle, CUBLAS_FILL_MODE_LOWER, m, d_a, ldda, &worksz);
      // float *d_work = nullptr;
      // cuerr = cudaMalloc((void**)&d_work, worksz*sizeof(float)); 

      float alpha = -1.0, beta = 1.0;
      sylver::gpu::half alpha_hp = -1.0, beta_hp = 1.0;
      // We use a 2 level blocking for maximizing the performance of
      // the GEMM operaion on the GPU

      for (int kk = 0; kk < nc; ++kk) {

         int ofs = kk*nb;
         int in = std::min(n-ofs, nb);
         int inc = (in-1) / ib + 1; 
         int updm = m-ofs;

         // Copy panel into temporary buffer
         // start = std::chrono::high_resolution_clock::now();
         // sylver::gpu::convert(stream, updm, in, &d_a_hp[ofs+ofs*ldda], ldda, d_a_tmp, ldda);
         // cudaStreamSynchronize(stream);
         // end = std::chrono::high_resolution_clock::now();
         // tconv =  
         // std::chrono::duration_cast<std::chrono::nanoseconds>
            // (end-start).count();
         // std::cout << "[" << context << "] "
                   // << "Time to convert hp to sp into panel (s) = " << 1e-9*tconv 
                   // << std::endl;

         // std::cout << "[" << context << "]" << " im =  " << updm << ", in = " << in << std::endl;

         // Update trailing outer block in a left-looking fashion
         if (ofs > 0) {

            // std::cout << "[spldlt::gpu::factor] updm = " << updm << ", updn = " << in << ", k = " << ofs << std::endl;

            // custat = cublasHgemm(cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
            //                      updm, in, ofs,
            //                      &alpha_hp,
            //                      &d_a_hp[ofs], ldda,
            //                      &d_a_hp[ofs], ldda,
            //                      &beta_hp,
            //                      &d_a_hp[ofs+ofs*ldda], ldda);
            
            custat = cublasGemmEx(cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                  updm, in, ofs, 
                                  &alpha_hp,
                                  // &alpha,
                                  &d_a_hp[ofs], CUDA_R_16F, ldda,
                                  &d_a_hp[ofs], CUDA_R_16F, ldda,
                                  &beta_hp,
                                  // &beta,
                                  // &d_a[ofs+ofs*ldda], CUDA_R_16F, ldda,
                                  // d_a_tmp, CUDA_R_32F, ldda,
                                  &d_a_hp[ofs+ofs*ldda], CUDA_R_16F, ldda,
                                  // CUDA_R_32F,
                                  CUDA_R_16F,
                                  (mode==CUBLAS_TENSOR_OP_MATH) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT
                                  // CUBLAS_GEMM_DEFAULT
                                  // CUBLAS_GEMM_ALGO0
                                  // CUBLAS_GEMM_ALGO1
                                  // CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                  // CUBLAS_GEMM_ALGO0_TENSOR_OP
                  );
            // custat = sylver::gpu::dev_gemm(
            //       cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
            //       updm, in, ofs, &alpha,
            //       &d_a[ofs], ldda,
            //       &d_a[ofs], ldda,
            //       &beta, &d_a[ofs+ofs*ldda], ldda);
            // cudaStreamSynchronize(stream);
            sylver::gpu::cublas_check_error(custat, context, inform, "Failed to launch cublasGemmEx");
            // cudaStreamSynchronize(stream);
         }

         sylver::gpu::convert(stream, updm, in, &d_a_hp[ofs+ofs*ldda], ldda, d_a_tmp, ldda);

         // Factor outer block
         for (int k = 0; k < inc; ++k) {
         
            // std::cout << "[spldlt::gpu::factor] k = " << k << std::endl;
            // Factor kth block column
            int iofs = k*ib; // Number of eliminated columns
            int cblkm = m-ofs-iofs; // Block column height
            int cblkn = std::min(in-iofs, ib); // Block column width

            // Copy diagonal tile into workspace d_d
            cuerr = cudaMemcpy2DAsync(
                  d_d, lddd*sizeof(float),
                  &d_a_tmp[iofs+iofs*ldda], ldda*sizeof(float),
                  cblkn*sizeof(float), cblkn,
                  cudaMemcpyDeviceToDevice,
                  stream);
            // cudaStreamSynchronize(stream);
            sylver::gpu::cuda_check_error(cuerr, context, inform, "Failed to copy diag tile into buffer");
            
            factor_bcol(
                  stream, cblkm, cblkn,
                  d_d, lddd,
                  &d_a_tmp[iofs+iofs*ldda], ldda,
                  d_info);
            // cudaStreamSynchronize(stream);
            
            // sylver::gpu::dev_potrf(
            //       cusolhandle, CUBLAS_FILL_MODE_LOWER,
            //       cblkn,
            //       &d_a_tmp[iofs+iofs*ldda], ldda,
            //       d_work, worksz, d_info);

            // factor_bcol(
            //       stream, cblkn, cblkn,
            //       d_d, lddd,
            //       &d_a_tmp[iofs+iofs*ldda], ldda,
            //       d_info);

            // float alp = 1.0;
            // sylver::gpu::dev_trsm(
            //       cuhandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            //       cblkm-cblkn, cblkn, &alp,
            //       &d_a_tmp[iofs+iofs*ldda], ldda,
            //       &d_a_tmp[iofs+cblkn+iofs*ldda], ldda);
            
            // cudaStreamSynchronize(stream);
            // int *info;
            // cudaMallocHost(&info, sizeof(int));
            // cuerr = cudaMemcpyAsync(info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
            // cudaStreamSynchronize(stream);
            // if (*info < cblkn) { // Not positive-definite
            //    std::cout << "[spldlt::gpu::factor][error] kk = " << kk << std::endl;
            //    std::cout << "[spldlt::gpu::factor][error] negative or null pivot, info = " << *info << std::endl;
            //    inform.flag = ERROR_NOT_POS_DEF;
            //    exit(1);
            // }
         
            // Update trailing submatrix
            int iofst = (k+1)*ib; // Offset to trailing submatrix in outer block
            int tblkm = updm-iofst; // Width of trailing submatrix in outer block
            int tblkn = in-iofst; // Width of trailing submatrix in outer block
            if (tblkn>0) {
            
               // Update trailing submatrix (inner & outer)
               // cubstat = sylver::gpu::dev_syrk(
               //       cuhandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
               //       tblkn, ib, 
               //       &alpha, &d_a[ofs+iofst+ (ofs+iofs)*ldda], ldda, 
               //       &beta,  &d_a[ofs+iofst+(ofs+iofst)*ldda], ldda);

               // Update trailing submatrix (inner & outer)
               // cubstat = sylver::gpu::dev_gemm(
               //       cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
               //       tblkn, tblkn, ib, &alpha,
               //       &d_a[ofst+ofs*ldda], ldda,
               //       &d_a[ofst+ofs*ldda], ldda,
               //       &beta, &d_a[ofst+ofst*ldda], ldda);

               // Update trailing submatrix (inner only)
               // std::cout << "[spldlt::gpu::factor] cblkm = " << cblkm << ", tblkn = " << tblkn << ", inc = " << inc << std::endl;
               custat = sylver::gpu::dev_gemm(
                     cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
                     tblkm, tblkn, ib, &alpha,
                     &d_a_tmp[iofst + iofs  * ldda], ldda,
                     &d_a_tmp[iofst + iofs  * ldda], ldda,
                     &beta,
                     &d_a_tmp[iofst + iofst * ldda], ldda);
               // cudaStreamSynchronize(stream);
               sylver::gpu::cublas_check_error(custat, context, inform, "Failed to launch inner block update");
                              
               // cudaStreamSynchronize(stream);
               // std::cout << "[spldlt::gpu::factor] cubstat = " << cubstat << std::endl;
            }

         }

         // Copy computed panel into half prec matrix representation
         sylver::gpu::convert(stream, updm, in, d_a_tmp, ldda, &d_a_hp[ofs+ofs*ldda], ldda);

      }
      // Copy matrix back to its original location and convert to full
      // prec
      start = std::chrono::high_resolution_clock::now();
      sylver::gpu::convert(stream, m, n, d_a_hp, ldda, d_a, ldda);
      end = std::chrono::high_resolution_clock::now();
      tconv =  
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (end-start).count();
      std::cout << "[" << context << "] "
                << "Time to convert hp to sp (s) = " << 1e-9*tconv 
                << std::endl;

      // Wait for completion
      cuerr = cudaStreamSynchronize(stream);
      sylver::gpu::cuda_check_error(cuerr, context, inform, "Failed to synchonize stream");

      // Cleanup memory
      cuerr =  cudaFree(d_a_tmp);
      sylver::gpu::cuda_check_error(cuerr, context, inform);
      // cublasDestroy(cuhandle);
      cuerr = cudaFree(d_d);
      sylver::gpu::cuda_check_error(cuerr, context, inform);
      cuerr = cudaFree(d_a_hp);
      sylver::gpu::cuda_check_error(cuerr, context, inform);

   }

   ////////////////////////////////////////

   // Using compute type 32F updating and factoring panel
   template<>
   void factor_ll_hp_u32<float>(
         const cublasHandle_t cuhandle, 
         int m, // Number of rows 
         int n, // Number of columns
         float *const d_a, // Matrix pointer on device 
         int ldda, // Matrix leadind dim on device
         inform_t& inform, // Info host
         int *d_info // Info device
         ) {
      
      // Error handling
      std::string context = "spldlt::gpu::factor_ll_hp_u32";
      cudaError_t cuerr; // CUDA error
      cublasStatus_t custat; // CuBLAS status
      
      int const ib = BLOCK_SIZE; // Inner block size
      // int const nb = BLOCK_SIZE; // Outer block size 
      int const nb = OUTER_BLOCK_SIZE; // Outer block size         
      // Number of block columns
      int const nc = (n-1) / nb +1;

      // Timers
      std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

      std::cout << "[" << context << "]" << std::endl;
      // std::cout << "[" << context << "]" << " nc = " << nc << std::endl;

      // Retreive CUDA stream from cuBLAS handle
      cudaStream_t stream = NULL; // CUDA Stream
      // custat = cublasSetStream(cuhandle, NULL);
      custat = cublasGetStream(cuhandle, &stream);
      sylver::gpu::cublas_check_error(custat, context, inform);

      // Set Math mode
      // custat = cublasSetMathMode(cuhandle, CUBLAS_DEFAULT_MATH);
      // custat = cublasSetMathMode(cuhandle, CUBLAS_TENSOR_OP_MATH);
      // sylver::gpu::cublas_check_error(custat, context, inform);
      cublasMath_t mode;
      cublasGetMathMode(cuhandle, &mode);
      
      // Allocate memory to accomodate half prec representation of
      // matrix A
      sylver::gpu::half *d_a_hp = nullptr;
      start = std::chrono::high_resolution_clock::now();
      cuerr = cudaMalloc((void**)&d_a_hp, m*ldda*sizeof(sylver::gpu::half));      
      sylver::gpu::cuda_check_error(cuerr, context, inform);
      end = std::chrono::high_resolution_clock::now();
      long talloc =
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (end-start).count();
      std::cout << "[" << context << "] "
                << "Time to allocate hp matrix (s) = " 
                << std::scientific << 1e-9*talloc 
                << std::endl;

      // Copy matrix into buffer and convert to half prec  
      start = std::chrono::high_resolution_clock::now();
      sylver::gpu::convert(stream, m, n, d_a, ldda, d_a_hp, ldda);
      end = std::chrono::high_resolution_clock::now();
      // Calculate time to convert
      long tconv =  
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (end-start).count();
      std::cout << "[" << context << "] "
                << "Time to convert sp to hp (s) = " << 1e-9*tconv 
                << std::endl;

      // Allocate buffer for computing the panel
      float *d_a_tmp = nullptr;
      cuerr = cudaMalloc((void**)&d_a_tmp, nb*ldda*sizeof(float));      
      sylver::gpu::cuda_check_error(cuerr, context, inform);      
      
      // Workspace holding the diagonal tile
      float *d_d = nullptr;
      int lddd = ib;
      cuerr = cudaMalloc((void**)&d_d, lddd*ib*sizeof(float));
      sylver::gpu::cuda_check_error(cuerr, context, inform);
      
      // // CuSOLVER for debug purpose
      // cusolverStatus_t cusolstat;
      // cusolverDnHandle_t cusolhandle;
      // cusolstat = cusolverDnCreate(&cusolhandle);
      // cusolstat = cusolverDnSetStream(cusolhandle, stream);
      // int worksz; // Workspace size
      // sylver::gpu::dev_potrf_buffersize(cusolhandle, CUBLAS_FILL_MODE_LOWER, m, d_a, ldda, &worksz);
      // float *d_work = nullptr;
      // cuerr = cudaMalloc((void**)&d_work, worksz*sizeof(float)); 

      float alpha = -1.0, beta = 1.0;
      sylver::gpu::half alpha_hp = -1.0, beta_hp = 1.0;
      // We use a 2 level blocking for maximizing the performance of
      // the GEMM operaion on the GPU

      for (int kk = 0; kk < nc; ++kk) {

         int ofs = kk*nb;
         int in = std::min(n-ofs, nb);
         int inc = (in-1) / ib + 1; 
         int updm = m-ofs;

         // Copy panel into temporary buffer
         // start = std::chrono::high_resolution_clock::now();
         sylver::gpu::convert(stream, updm, in, &d_a_hp[ofs+ofs*ldda], ldda, d_a_tmp, ldda);
         // cudaStreamSynchronize(stream);
         // end = std::chrono::high_resolution_clock::now();
         // tconv =  
         // std::chrono::duration_cast<std::chrono::nanoseconds>
            // (end-start).count();
         // std::cout << "[" << context << "] "
                   // << "Time to convert hp to sp into panel (s) = " << 1e-9*tconv 
                   // << std::endl;

         // std::cout << "[" << context << "]" << " im =  " << updm << ", in = " << in << std::endl;

         // Update trailing outer block in a left-looking fashion
         if (ofs > 0) {

            // std::cout << "[spldlt::gpu::factor] updm = " << updm << ", updn = " << in << ", k = " << ofs << std::endl;

            // custat = cublasHgemm(cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
            //                      updm, in, ofs,
            //                      &alpha_hp,
            //                      &d_a_hp[ofs], ldda,
            //                      &d_a_hp[ofs], ldda,
            //                      &beta_hp,
            //                      &d_a_hp[ofs+ofs*ldda], ldda);
            
            custat = cublasGemmEx(cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                  updm, in, ofs, 
                                  // &alpha_hp,
                                  &alpha,
                                  &d_a_hp[ofs], CUDA_R_16F, ldda,
                                  &d_a_hp[ofs], CUDA_R_16F, ldda,
                                  // &beta_hp,
                                  &beta,
                                  // &d_a[ofs+ofs*ldda], CUDA_R_16F, ldda,
                                  d_a_tmp, CUDA_R_32F, ldda,
                                  // &d_a_hp[ofs+ofs*ldda], CUDA_R_16F, ldda,
                                  CUDA_R_32F,
                                  // CUDA_R_16F,
                                  (mode==CUBLAS_TENSOR_OP_MATH) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT
                                  // CUBLAS_GEMM_DEFAULT
                                  // CUBLAS_GEMM_ALGO0
                                  // CUBLAS_GEMM_ALGO1
                                  // CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                  // CUBLAS_GEMM_ALGO0_TENSOR_OP
                  );
            // custat = sylver::gpu::dev_gemm(
            //       cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
            //       updm, in, ofs, &alpha,
            //       &d_a[ofs], ldda,
            //       &d_a[ofs], ldda,
            //       &beta, &d_a[ofs+ofs*ldda], ldda);
            // cudaStreamSynchronize(stream);
            sylver::gpu::cublas_check_error(custat, context, inform, "Failed to launch cublasGemmEx");
            // cudaStreamSynchronize(stream);
         }

         // sylver::gpu::convert(stream, updm, in, &d_a_hp[ofs+ofs*ldda], ldda, d_a_tmp, ldda);

         // Factor outer block
         for (int k = 0; k < inc; ++k) {
         
            // std::cout << "[spldlt::gpu::factor] k = " << k << std::endl;
            // Factor kth block column
            int iofs = k*ib; // Number of eliminated columns
            int cblkm = m-ofs-iofs; // Block column height
            int cblkn = std::min(in-iofs, ib); // Block column width

            // Copy diagonal tile into workspace d_d
            cuerr = cudaMemcpy2DAsync(
                  d_d, lddd*sizeof(float),
                  &d_a_tmp[iofs+iofs*ldda], ldda*sizeof(float),
                  cblkn*sizeof(float), cblkn,
                  cudaMemcpyDeviceToDevice,
                  stream);
            // cudaStreamSynchronize(stream);
            sylver::gpu::cuda_check_error(cuerr, context, inform, "Failed to copy diag tile into buffer");
            
            factor_bcol(
                  stream, cblkm, cblkn,
                  d_d, lddd,
                  &d_a_tmp[iofs+iofs*ldda], ldda,
                  d_info);
            // cudaStreamSynchronize(stream);
            
            // sylver::gpu::dev_potrf(
            //       cusolhandle, CUBLAS_FILL_MODE_LOWER,
            //       cblkn,
            //       &d_a_tmp[iofs+iofs*ldda], ldda,
            //       d_work, worksz, d_info);

            // factor_bcol(
            //       stream, cblkn, cblkn,
            //       d_d, lddd,
            //       &d_a_tmp[iofs+iofs*ldda], ldda,
            //       d_info);

            // float alp = 1.0;
            // sylver::gpu::dev_trsm(
            //       cuhandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            //       cblkm-cblkn, cblkn, &alp,
            //       &d_a_tmp[iofs+iofs*ldda], ldda,
            //       &d_a_tmp[iofs+cblkn+iofs*ldda], ldda);
            
            // cudaStreamSynchronize(stream);
            // int *info;
            // cudaMallocHost(&info, sizeof(int));
            // cuerr = cudaMemcpyAsync(info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
            // cudaStreamSynchronize(stream);
            // if (*info < cblkn) { // Not positive-definite
            //    std::cout << "[spldlt::gpu::factor][error] kk = " << kk << std::endl;
            //    std::cout << "[spldlt::gpu::factor][error] negative or null pivot, info = " << *info << std::endl;
            //    inform.flag = ERROR_NOT_POS_DEF;
            //    exit(1);
            // }
         
            // Update trailing submatrix
            int iofst = (k+1)*ib; // Offset to trailing submatrix in outer block
            int tblkm = updm-iofst; // Width of trailing submatrix in outer block
            int tblkn = in-iofst; // Width of trailing submatrix in outer block
            if (tblkn>0) {
            
               // Update trailing submatrix (inner & outer)
               // cubstat = sylver::gpu::dev_syrk(
               //       cuhandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
               //       tblkn, ib, 
               //       &alpha, &d_a[ofs+iofst+ (ofs+iofs)*ldda], ldda, 
               //       &beta,  &d_a[ofs+iofst+(ofs+iofst)*ldda], ldda);

               // Update trailing submatrix (inner & outer)
               // cubstat = sylver::gpu::dev_gemm(
               //       cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
               //       tblkn, tblkn, ib, &alpha,
               //       &d_a[ofst+ofs*ldda], ldda,
               //       &d_a[ofst+ofs*ldda], ldda,
               //       &beta, &d_a[ofst+ofst*ldda], ldda);

               // Update trailing submatrix (inner only)
               // std::cout << "[spldlt::gpu::factor] cblkm = " << cblkm << ", tblkn = " << tblkn << ", inc = " << inc << std::endl;
               custat = sylver::gpu::dev_gemm(
                     cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
                     tblkm, tblkn, ib, &alpha,
                     &d_a_tmp[iofst + iofs  * ldda], ldda,
                     &d_a_tmp[iofst + iofs  * ldda], ldda,
                     &beta,
                     &d_a_tmp[iofst + iofst * ldda], ldda);
               // cudaStreamSynchronize(stream);
               sylver::gpu::cublas_check_error(custat, context, inform, "Failed to launch inner block update");
                              
               // cudaStreamSynchronize(stream);
               // std::cout << "[spldlt::gpu::factor] cubstat = " << cubstat << std::endl;
            }

         }

         // Copy computed panel into half prec matrix representation
         sylver::gpu::convert(stream, updm, in, d_a_tmp, ldda, &d_a_hp[ofs+ofs*ldda], ldda);

      }
      // Copy matrix back to its original location and convert to full
      // prec
      start = std::chrono::high_resolution_clock::now();
      sylver::gpu::convert(stream, m, n, d_a_hp, ldda, d_a, ldda);
      end = std::chrono::high_resolution_clock::now();
      tconv =  
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (end-start).count();
      std::cout << "[" << context << "] "
                << "Time to convert hp to sp (s) = " << 1e-9*tconv 
                << std::endl;

      // Wait for completion
      cuerr = cudaStreamSynchronize(stream);
      sylver::gpu::cuda_check_error(cuerr, context, inform, "Failed to synchonize stream");

      // Cleanup memory
      cuerr =  cudaFree(d_a_tmp);
      sylver::gpu::cuda_check_error(cuerr, context, inform);
      // cublasDestroy(cuhandle);
      cuerr = cudaFree(d_d);
      sylver::gpu::cuda_check_error(cuerr, context, inform);
      cuerr = cudaFree(d_a_hp);
      sylver::gpu::cuda_check_error(cuerr, context, inform);

   }

   ////////////////////////////////////////

   // Using compute type 16F for updating panel
   template<>
   void factor_ll_hp_c16<float>(
         const cublasHandle_t cuhandle, 
         int m, // Number of rows 
         int n, // Number of columns
         float *const d_a, // Matrix pointer on device 
         int ldda, // Matrix leadind dim on device
         inform_t& inform, // Info host
         int *d_info // Info device
         ) {
      
      // Error handling
      std::string context = "spldlt::gpu::factor_ll_hp_c16";
      cudaError_t cuerr; // CUDA error
      cublasStatus_t custat; // CuBLAS status
      
      int const ib = BLOCK_SIZE; // Inner block size
      // int const nb = BLOCK_SIZE; // Outer block size 
      int const nb = OUTER_BLOCK_SIZE; // Outer block size         
      // Number of block columns
      int const nc = (n-1) / nb +1;

      // Timers
      std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

      std::cout << "[" << context << "]" << std::endl;

      // Retreive CUDA stream from cuBLAS handle
      cudaStream_t stream; // CUDA Stream
      custat = cublasGetStream(cuhandle, &stream);
      sylver::gpu::cublas_check_error(custat, context, inform);

      // Set Math mode
      // custat = cublasSetMathMode(cuhandle, CUBLAS_DEFAULT_MATH);
      // custat = cublasSetMathMode(cuhandle, CUBLAS_TENSOR_OP_MATH);
      // sylver::gpu::cublas_check_error(custat, context, inform);
      
      // Allocate memory to accomodate half prec representation of
      // matrix A
      sylver::gpu::half *d_a_hp = nullptr;
      start = std::chrono::high_resolution_clock::now();
      cuerr = cudaMalloc((void**)&d_a_hp, m*ldda*sizeof(sylver::gpu::half));      
      sylver::gpu::cuda_check_error(cuerr, context, inform);
      end = std::chrono::high_resolution_clock::now();
      long talloc =
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (end-start).count();
      std::cout << "[" << context << "] "
                << "Time to allocate hp matrix (s) = " 
                << std::scientific << 1e-9*talloc 
                << std::endl;

      // Copy matrix into buffer and convert to half prec  
      start = std::chrono::high_resolution_clock::now();
      sylver::gpu::convert(stream, m, n, d_a, ldda, d_a_hp, ldda);
      end = std::chrono::high_resolution_clock::now();
      // Calculate time to convert
      long tconv =  
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (end-start).count();
      std::cout << "[" << context << "] "
                << "Time to convert sp to hp (s) = " << 1e-9*tconv 
                << std::endl;

      // Allocate buffer for computing the panel
      // float *d_a_tmp = nullptr;
      // cuerr = cudaMalloc((void**)&d_a_tmp, nb*ldda*sizeof(float)); // nb columns wide panel      
      // cuerr = cudaMalloc((void**)&d_a_tmp, ib*ldda*sizeof(float)); // ib columns wide panel
      // sylver::gpu::cuda_check_error(cuerr, context, inform);      
      
      // Workspace holding the diagonal tile
      // float *d_d = nullptr;
      sylver::gpu::half *d_d = nullptr;
      int lddd = ib;
      // cuerr = cudaMalloc((void**)&d_d, lddd*ib*sizeof(float));
      cuerr = cudaMalloc((void**)&d_d, lddd*ib*sizeof(sylver::gpu::half));
      sylver::gpu::cuda_check_error(cuerr, context, inform);
      
      // float alpha = -1.0, beta = 1.0;
      sylver::gpu::half alpha_hp = -1.0, beta_hp = 1.0;
      // We use a 2 level blocking for maximizing the performance of
      // the GEMM operaion on the GPU

      for (int kk = 0; kk < nc; ++kk) {

         int ofs = kk*nb;
         int in = std::min(n-ofs, nb);
         int inc = (in-1) / ib + 1; 
         int updm = m-ofs;

         // Copy panel into temporary buffer
         // start = std::chrono::high_resolution_clock::now();
         // sylver::gpu::convert(stream, updm, in, &d_a_hp[ofs+ofs*ldda], ldda, d_a_tmp, ldda);
         // cudaStreamSynchronize(stream);
         // end = std::chrono::high_resolution_clock::now();
         // tconv =  
         // std::chrono::duration_cast<std::chrono::nanoseconds>
            // (end-start).count();
         // std::cout << "[" << context << "] "
                   // << "Time to convert hp to sp into panel (s) = " << 1e-9*tconv 
                   // << std::endl;

         // std::cout << "[" << context << "]" << " im =  " << updm << ", in = " << in << std::endl;

         // Update trailing outer block in a left-looking fashion
         if (ofs > 0) {

            // std::cout << "[spldlt::gpu::factor] updm = " << updm << ", updn = " << in << ", k = " << ofs << std::endl;
            custat = cublasHgemm(cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 updm, in, ofs,
                                 &alpha_hp,
                                 &d_a_hp[ofs], ldda,
                                 &d_a_hp[ofs], ldda,
                                 &beta_hp,
                                 &d_a_hp[ofs+ofs*ldda], ldda);
            
            // cublasGemmEx(cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
            //              updm, in, ofs, &alpha_hp,
            //              &d_a_hp[ofs], CUDA_R_16F, ldda,
            //              &d_a_hp[ofs], CUDA_R_16F, ldda,
            //              &beta_hp,
            //              &d_a_hp[ofs+ofs*ldda], CUDA_R_16F, ldda,
            //              // d_a_tmp, CUDA_R_32F, ldda,
            //              CUDA_R_16F,
            //              // CUDA_R_32F,
            //              // CUBLAS_GEMM_DEFAULT
            //              CUBLAS_GEMM_DEFAULT_TENSOR_OP
            //       );
            // custat = sylver::gpu::dev_gemm(
            //       cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
            //       updm, in, ofs, &alpha,
            //       &d_a[ofs], ldda,
            //       &d_a[ofs], ldda,
            //       &beta, &d_a[ofs+ofs*ldda], ldda);
            // cudaStreamSynchronize(stream);
            sylver::gpu::cublas_check_error(custat, context, inform, "Failed to launch cublasGemmEx");
            // cudaStreamSynchronize(stream);
         }

         // Factor outer block
         for (int k = 0; k < inc; ++k) {
         
            // std::cout << "[spldlt::gpu::factor] k = " << k << std::endl;
            // Factor kth block column
            int iofs = k*ib; // Number of eliminated columns
            int cblkm = m-ofs-iofs; // Block column height
            int cblkn = std::min(in-iofs, ib); // Block column width

            // sylver::gpu::convert(stream, updm, ib, &d_a_hp[ofs+iofs+(ofs+iofs)*ldda], ldda, d_a_tmp, ldda);
            // cudaStreamSynchronize(stream);

            // Copy diagonal tile into workspace d_d
            // cuerr = cudaMemcpy2DAsync(
            //       d_d, lddd*sizeof(float),
            //       // &d_a_tmp[iofs+iofs*ldda], ldda*sizeof(float),
            //       d_a_tmp, ldda*sizeof(float),
            //       cblkn*sizeof(float), cblkn,
            //       cudaMemcpyDeviceToDevice,
            //       stream);
            cuerr = cudaMemcpy2DAsync(
                  d_d, lddd*sizeof(sylver::gpu::half),
                  // &d_a_tmp[iofs+iofs*ldda], ldda*sizeof(float),
                  // d_a_tmp, ldda*sizeof(sylver::gpu::half),
                  &d_a_hp[ofs+iofs+(ofs+iofs)*ldda], ldda*sizeof(sylver::gpu::half),
                  cblkn*sizeof(sylver::gpu::half), cblkn,
                  cudaMemcpyDeviceToDevice,
                  stream);
            // cudaStreamSynchronize(stream);
            sylver::gpu::cuda_check_error(cuerr, context, inform, "Failed to copy diag tile into buffer");
            
            // factor_bcol(
            //       stream, cblkm, cblkn,
            //       d_d, lddd,
            //       // &d_a_tmp[iofs+iofs*ldda], ldda,
            //       d_a_tmp, ldda,
            //       d_info);
            factor_bcol(
                  stream, cblkm, cblkn,
                  d_d, lddd,
                  &d_a_hp[ofs+iofs+(ofs+iofs)*ldda], ldda,
                  d_info);
            // cudaStreamSynchronize(stream);

            // sylver::gpu::convert(stream, updm, ib, d_a_tmp, ldda, &d_a_hp[ofs+iofs+(ofs+iofs)*ldda], ldda);
            // cudaStreamSynchronize(stream);

            // sylver::gpu::dev_potrf(
            //       cusolhandle, CUBLAS_FILL_MODE_LOWER, cblkn, &d_a[ofs+iofs+(ofs+iofs)*ldda], ldda,
            //       d_work, worksz, d_info);

            // factor_bcol(
            //       stream, cblkn, cblkn,
            //       &d_a[ofs+iofs+(ofs+iofs)*ldda], ldda,
            //       d_info);

            // T alp = 1.0;
            // sylver::gpu::dev_trsm(
            //       cuhandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            //       cblkm-cblkn, cblkn, &alp,
            //       &d_a[ofs+iofs+(ofs+iofs)*ldda], ldda,
            //       &d_a[ofs+iofs+cblkn+(ofs+iofs)*ldda], ldda);
            
            // cudaStreamSynchronize(stream);
            // int *info;
            // cudaMallocHost(&info, sizeof(int));
            // cuerr = cudaMemcpyAsync(info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
            // cudaStreamSynchronize(stream);
            // if (*info < cblkn) { // Not positive-definite
            //    std::cout << "[spldlt::gpu::factor][error] kk = " << kk << std::endl;
            //    std::cout << "[spldlt::gpu::factor][error] negative or null pivot, info = " << *info << std::endl;
            //    inform.flag = ERROR_NOT_POS_DEF;
            //    exit(1);
            // }
         
            // Update trailing submatrix
            int iofst = (k+1)*ib; // Offset to trailing submatrix in outer block
            int tblkm = updm-iofst; // Width of trailing submatrix in outer block
            int tblkn = in-iofst; // Width of trailing submatrix in outer block
            if (tblkn>0) {
            
               // Update trailing submatrix (inner & outer)
               // cubstat = sylver::gpu::dev_syrk(
               //       cuhandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
               //       tblkn, ib, 
               //       &alpha, &d_a[ofs+iofst+ (ofs+iofs)*ldda], ldda, 
               //       &beta,  &d_a[ofs+iofst+(ofs+iofst)*ldda], ldda);

               // Update trailing submatrix (inner & outer)
               // cubstat = sylver::gpu::dev_gemm(
               //       cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
               //       tblkn, tblkn, ib, &alpha,
               //       &d_a[ofst+ofs*ldda], ldda,
               //       &d_a[ofst+ofs*ldda], ldda,
               //       &beta, &d_a[ofst+ofst*ldda], ldda);

               // Update trailing submatrix (inner only)
               // std::cout << "[spldlt::gpu::factor] cblkm = " << cblkm << ", tblkn = " << tblkn << ", inc = " << inc << std::endl;
               // custat = sylver::gpu::dev_gemm(
               //       cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
               //       tblkm, tblkn, ib, &alpha,
               //       &d_a_tmp[iofst + iofs  * ldda], ldda,
               //       &d_a_tmp[iofst + iofs  * ldda], ldda,
               //       &beta,
               //       &d_a_tmp[iofst + iofst * ldda], ldda);
               // cudaStreamSynchronize(stream);
               // custat = cublasGemmEx(cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
               //                       tblkm, tblkn, ib, &alpha_hp,
               //                       &d_a_hp[ofs+iofst+(ofs+iofs)*ldda], CUDA_R_16F, ldda,
               //                       &d_a_hp[ofs+iofst+(ofs+iofs)*ldda], CUDA_R_16F, ldda,                            
               //                       &beta_hp,
               //                       // &d_a[ofs+ofs*ldda], CUDA_R_16F, ldda,
               //              &d_a_hp[ofs+iofst+(ofs+iofst)*ldda], CUDA_R_16F, ldda,
               //                       CUDA_R_16F,
               //                       // CUBLAS_GEMM_DEFAULT
               //                       CUBLAS_GEMM_DEFAULT_TENSOR_OP
               //       );

               custat = cublasHgemm(cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
                           tblkm, tblkn, ib, 
                           &alpha_hp,
                           &d_a_hp[ofs+iofst+(ofs+iofs)*ldda], ldda,
                           &d_a_hp[ofs+iofst+(ofs+iofs)*ldda], ldda,
                           &beta_hp,
                           &d_a_hp[ofs+iofst+(ofs+iofst)*ldda], ldda);            

               sylver::gpu::cublas_check_error(custat, context, inform, "Failed to launch inner block update");
                              
               // cudaStreamSynchronize(stream);
               // std::cout << "[spldlt::gpu::factor] cubstat = " << cubstat << std::endl;
            }

         }

         // Copy computed panel into half prec matrix representation
         // sylver::gpu::convert(stream, updm, in, d_a_tmp, ldda, &d_a_hp[ofs+ofs*ldda], ldda);

      }
      // Copy matrix back to its original location and convert to full
      // prec
      start = std::chrono::high_resolution_clock::now();
      sylver::gpu::convert(stream, m, n, d_a_hp, ldda, d_a, ldda);
      end = std::chrono::high_resolution_clock::now();
      tconv =  
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (end-start).count();
      std::cout << "[" << context << "] "
                << "Time to convert hp to sp (s) = " << 1e-9*tconv 
                << std::endl;

      // Wait for completion
      cuerr = cudaStreamSynchronize(stream);
      sylver::gpu::cuda_check_error(cuerr, context, inform, "Failed to synchonize stream");

      // Cleanup memory
      // cuerr =  cudaFree(d_a_tmp);
      // sylver::gpu::cuda_check_error(cuerr, context, inform);
      // cublasDestroy(cuhandle);
      cuerr = cudaFree(d_d);
      sylver::gpu::cuda_check_error(cuerr, context, inform);
      cuerr = cudaFree(d_a_hp);
      sylver::gpu::cuda_check_error(cuerr, context, inform);

   }

}}} // End of namespace sylver::spldlt::gpu
