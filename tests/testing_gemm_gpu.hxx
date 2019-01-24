/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "kernels/gpu/common.hxx"
#include "kernels/gpu/wrappers.hxx"

// STD
#include <iostream>
#include <string>
#include <chrono>

namespace sylver {
namespace tests {   

   template<typename T>
   int gemm_test(int m, int n, int k, enum algo algo) {

      std::string context = "gemm_test";
      bool failed = false;
      cudaError_t cuerr; // CUDA error
      cublasStatus_t custat; // cuBLAS error

      // Leading dimensions
      int lda = spral::ssids::cpu::align_lda<T>(m);
      int ldb = spral::ssids::cpu::align_lda<T>(k);

      std::cout << "[gemm_test] m = " << m << ", n = " << n << ", k = " << k << std::endl;
      std::cout << "[gemm_test] lda = " << lda << ", ldb = " << ldb << std::endl;

      // CUDA stream
      cudaStream_t stream;
      cuerr = cudaStreamCreate(&stream);
      sylver::gpu::cuda_check_error(cuerr, context);

      // Timers
      std::chrono::time_point<std::chrono::high_resolution_clock> sa, en;

      // Generate matrices
      T *a = nullptr;
      T *b = nullptr;
      T *c = nullptr;
         
      T alpha = -1.0, beta = 1.0;

      cudaMallocHost((void**)&a, lda*k*sizeof(T));
      cudaMallocHost((void**)&b, ldb*n*sizeof(T));
      cudaMallocHost((void**)&c, lda*n*sizeof(T));

      // Genrate random matrices
      ::spldlt::tests::gen_mat(m, k, a, lda);
      ::spldlt::tests::gen_mat(k, n, b, ldb);
      ::spldlt::tests::gen_mat(m, n, c, lda);

      // Send matrices to GPU
      T *d_a = nullptr;
      T *d_b = nullptr;
      T *d_c = nullptr;

      // Leading dimensions on the device
      int ldda = lda;
      int lddb = ldb;

      // Allocate memory for matrices on the device
      cuerr = cudaMalloc((void**)&d_a, ldda*k*sizeof(T));
      sylver::gpu::cuda_check_error(cuerr, context);
      cuerr = cudaMalloc((void**)&d_b, lddb*n*sizeof(T));
      sylver::gpu::cuda_check_error(cuerr, context);
      cuerr = cudaMalloc((void**)&d_c, ldda*n*sizeof(T));
      sylver::gpu::cuda_check_error(cuerr, context);
      
      // Send matrices to the device
      custat = cublasSetMatrix(m, k, sizeof(T), a, lda, d_a, ldda);
      sylver::gpu::cublas_check_error(custat, context);
      custat = cublasSetMatrix(k, n, sizeof(T), b, ldb, d_b, lddb);
      sylver::gpu::cublas_check_error(custat, context);
      custat = cublasSetMatrix(m, n, sizeof(T), c, lda, d_c, ldda);
      sylver::gpu::cublas_check_error(custat, context);

      cudaDeviceSynchronize();

      if (algo == sylver::tests::cuSOLVER) {

         std::cout << context << " cuSOLVER" << std::endl;

         // Setup cuBLAS handle
         cublasHandle_t cuhandle;
         custat = cublasCreate(&cuhandle);
         sylver::gpu::cublas_check_error(custat, context);
         custat = cublasSetStream(cuhandle, stream);
         sylver::gpu::cublas_check_error(custat, context);

         // Perfom GEMM
         sa = std::chrono::high_resolution_clock::now();
         custat = sylver::gpu::dev_gemm(
               cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
               m, n, k, &alpha,
               d_a, ldda,
               d_b, lddb,
               &beta,
               d_c, ldda);
         // cuerr = cudaStreamSynchronize(stream);
         sylver::gpu::cublas_check_error(custat, context);
         // Wait for completion
         cuerr = cudaStreamSynchronize(stream);
         sylver::gpu::cuda_check_error(cuerr, context);
         en = std::chrono::high_resolution_clock::now();

         // Cleanup cuBLAS handle
         custat = cublasDestroy(cuhandle);
         sylver::gpu::cublas_check_error(custat, context);
         
      }
      else if(algo == sylver::tests::cuSOLVER_HP) {

         std::cout << context << " cuSOLVER half prec" << std::endl;

         // cublasGemmEx(cuhandle, CUBLAS_OP_N, CUBLAS_OP_T,
         //              updm, in, ofs, &alpha,
         //              &d_a_hp[ofs], CUDA_R_16F, ldda,
         //              &d_a_hp[ofs], CUDA_R_16F, ldda,
         //              &beta,
         //              // &d_a[ofs+ofs*ldda], CUDA_R_16F, ldda,
         //              d_a_tmp, CUDA_R_32F, ldda,
         //              CUDA_R_32F,
         //              CUBLAS_GEMM_DEFAULT);
         
      }
      else {
         std::cout << "[chol_test] Algo NOT implemented " << std::endl;
         std::exit(0);
      }
      
      cuerr = cudaDeviceSynchronize();
      sylver::gpu::cuda_check_error(cuerr, context);

      long ttotal = 
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (en-sa).count();
      double flops = (double)2*m*n*k;
      printf("flops = %e\n", flops);
      printf("time (s) = %e\n", 1e-9*ttotal);
      printf("GFlop/s = %.3f\n", flops/(double)ttotal);

      // Cleanup

      // Cleanup device side memory
      cudaFree(d_a);
      cudaFree(d_b);
      cudaFree(d_c);
      // Cleanup host side memory
      cudaFreeHost(a);
      cudaFreeHost(b);
      cudaFreeHost(c);
      // Cleanup cuda structures
      cuerr = cudaStreamDestroy(stream);
      sylver::gpu::cuda_check_error(cuerr, context);

   }

}}
