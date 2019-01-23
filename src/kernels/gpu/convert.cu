/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#include "kernels/gpu/convert.cuh"
//STD
#include <cassert>
#include <iostream>
// CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// MAGMA see magmablas/hlaconvert.cu
const int max_blocks = 65535;

// MAGMA see magmablas/hlaconvert.cu
#define BLK_X 64
#define BLK_Y BLK_X

namespace sylver {
namespace gpu {

   // MAGMA routine see magmablas/hlaconvert.cu
   static __device__
   void convert_sp2hp_device(
         int m, int n,
         const float  *dA, int ldda,
         sylver::gpu::half *dB, int lddb )
   {
      int ind = blockIdx.x*BLK_X + threadIdx.x;
      int iby = blockIdx.y*BLK_Y;
      /* check if full block-column */
      bool full = (iby + BLK_Y <= n);
      /* do only rows inside matrix */
      if ( ind < m ) {
         dA += ind + iby*ldda;
         dB += ind + iby*lddb;
         if ( full ) {
            // full block-column
#pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
               dB[j*lddb] = __float2half( dA[j*ldda] );
            }
         }
         else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
               dB[j*lddb] = __float2half( dA[j*ldda] );
            }
         }
      }
   }

   // MAGMA routine see magmablas/hlaconvert.cu
   static __device__
   void convert_hp2sp_device(
         int m, int n,
         const sylver::gpu::half *dA, int ldda,
         float  *dB, int lddb )
   {
      int ind = blockIdx.x*BLK_X + threadIdx.x;
      int iby = blockIdx.y*BLK_Y;
      /* check if full block-column */
      bool full = (iby + BLK_Y <= n);
      /* do only rows inside matrix */
      if ( ind < m ) {
         dA += ind + iby*ldda;
         dB += ind + iby*lddb;
         if ( full ) {
            // full block-column
#pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
               dB[j*lddb] = __half2float( dA[j*ldda] );
            }
         }
         else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
               dB[j*lddb] = __half2float( dA[j*ldda] );
            }
         }
      }
   }

   // MAGMA routine see magmablas/hlaconvert.cu
   __global__
   void convert_sp2hp_kernel(
         int m, int n,
         const float  *dA, int ldda,
         sylver::gpu::half *dB, int lddb )
   {
#if CUDA_VERSION >= 7500
      convert_sp2hp_device(m, n, dA, ldda, dB, lddb);
#endif
   }

   // MAGMA routine see magmablas/hlaconvert.cu
   __global__
   void convert_hp2sp_kernel(
         int m, int n,
         const sylver::gpu::half *dA, int ldda,
         float  *dB, int lddb )
   {
#if CUDA_VERSION >= 7500
      convert_hp2sp_device(m, n, dA, ldda, dB, lddb);
#endif
   }

   template<typename TA, typename TAO> 
   __global__
   void convert_kernel(int m, int n, const TA  *dA, int ldda, TAO *dB, int lddb );

   // Template specialization
   template<>
   __global__
   void convert_kernel<float, sylver::gpu::half>(
         int m, int n,
         const float  *dA, int ldda,
         sylver::gpu::half *dB, int lddb ) {
      convert_sp2hp_device(m, n, dA, ldda, dB, lddb);
   }
   template<>
   __global__
   void convert_kernel<sylver::gpu::half, float>(
         int m, int n,
         const sylver::gpu::half  *dA, int ldda,
         float *dB, int lddb ) {
      convert_hp2sp_device(m, n, dA, ldda, dB, lddb);
   }

   // @brief Convert matrix a of type float into half prec and put
   // result in aout
   template<typename TA, typename TAO> 
   void convert(
         cudaStream_t const stream,
         int m, int n,
         TA *const a, int lda, 
         TAO *const aout, int ldaout) {
      
      std::cout << "[convert]"
                << " m = " << m << ", n = " << n
                << " lda = " << lda << ", ldaout = " << ldaout
                << std::endl;
      
      assert( BLK_X == BLK_Y );
      const int super_NB = max_blocks*BLK_X;
      dim3 super_grid(
            (m + super_NB - 1) / super_NB, 
            (n + super_NB - 1) / super_NB);
    
      dim3 threads( BLK_X, 1 );
      dim3 grid;
    
      int mm, nn;
      for( unsigned int i=0; i < super_grid.x; ++i ) {
         mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
         grid.x = (mm + super_NB - 1) / BLK_X;
         for( unsigned int j=0; j < super_grid.y; ++j ) {  // full row
            nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
            grid.y = (nn + super_NB - 1) / BLK_Y;
            convert_kernel 
               <<< grid, threads, 0, stream >>>
               (mm, nn, &a[i*super_NB + j*super_NB*lda], lda, &aout[i*super_NB + j*super_NB*ldaout], ldaout);
         }
      }
   }
   
   template void convert<float, sylver::gpu::half>(
         cudaStream_t const stream, int m, int n, float *const a, int lda, 
         sylver::gpu::half *const aout, int ldaout);

   template void convert<sylver::gpu::half, float>(
         cudaStream_t const stream, int m, int n, sylver::gpu::half *const a, int lda, 
         float *const aout, int ldaout);

}} // End of namespace sykver::gpu
