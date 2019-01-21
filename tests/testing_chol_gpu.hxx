/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER tests
#include "common.hxx"
#include "sylver_ciface.hxx"
#include "kernels/wrappers.hxx"
#include "kernels/gpu/wrappers.hxx"
#include "kernels/gpu/factor.hxx"

// STD
#include <iostream>

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/kernels/cholesky.hxx"
// SSIDS tests
#include "tests/ssids/kernels/framework.hxx"
// CUDA, CuBLAS and CuSOLVER
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "cublas_v2.h"

namespace sylver {
namespace tests {   
      
   template<typename T>
   int chol_test(int m, enum algo algo) {

      bool failed = false;

      T* a = nullptr;
      T* b = nullptr;
      T* l = nullptr;

      // Generate test matrix
      int lda = spral::ssids::cpu::align_lda<T>(m);
      std::cout << "[chol_test] m = " << m << ", lda = " << lda << std::endl;
      a = new T[m*lda];
      sylver::tests::gen_posdef(m, a, lda);

      // Generate a RHS based on x=1, b=Ax
      b = new T[m];
      sylver::tests::gen_rhs(m, a, lda, b);

      // Copy a into l
      l = new T[m*lda];
      memcpy(l, a, lda*m*sizeof(T));
      // ::spldlt::tests::print_mat("%12.3e", m, l, lda);

      cudaError_t cuerr;
      cublasStatus_t custat;

      // Allocate memory on the device
      T *d_l = nullptr;
      cuerr = cudaMalloc((void**)&d_l, m*lda*sizeof(T));
      if (cuerr != cudaSuccess) {
            std::cout << "[chol_test][error] Failed allocate memory on device "
                      << "(" << cudaGetErrorString(cuerr) << ")" << std::endl;
            std::exit(1);
      }
      // Send matrix to device
      auto transfer_sa = std::chrono::high_resolution_clock::now();
      custat = cublasSetMatrix(m, m, sizeof(T), l, lda, d_l, lda);
      // cudaMemcpy(d_l, l, lda*m*sizeof(T), cudaMemcpyHostToDevice);
      if (custat != CUBLAS_STATUS_SUCCESS) {    
         printf("[chol_test][error] CuBLAS SetMatrix failed\n");
         return 1;
      }   
      auto transfer_en = std::chrono::high_resolution_clock::now();
      long t_transfer = 
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (transfer_en-transfer_sa).count();

      cudaStream_t stream;
      cuerr = cudaStreamCreate(&stream);
      if (cuerr != cudaSuccess) {
         std::cout << "[chol_test][error] Failed to create stream "
                   << "(" << cudaGetErrorString(cuerr) << ")" << std::endl;
         std::exit(1);
      }

      std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

      if (algo == sylver::tests::cuSOLVER) {

         // Initialize cuSOLVER
         cusolverStatus_t cusolstat;
         cusolverDnHandle_t cusolhandle;

         cusolstat = cusolverDnCreate(&cusolhandle);
         cusolverDnSetStream(cusolhandle, stream);
         T *d_work = nullptr;
         int worksz = 0; // Workspace size
         cusolstat = sylver::gpu::dev_potrf_buffersize(
               cusolhandle, CUBLAS_FILL_MODE_LOWER, m, d_l, lda, &worksz);
         std::cout << "[chol_test] dev_potrf_buffersize cusolstat = " << cusolstat << std::endl;
         std::cout << "[chol_test] work size = " << worksz << std::endl;
         cuerr = cudaMalloc((void**)&d_work, worksz*sizeof(T)); 
         int *d_info = nullptr;
         cudaMalloc((void**)&d_info, sizeof(int));

         start = std::chrono::high_resolution_clock::now();
         // Launch cuSOLVER potrf
         cusolstat = sylver::gpu::dev_potrf(
               cusolhandle, CUBLAS_FILL_MODE_LOWER, 
               m,
               d_l, lda, 
               d_work, worksz,
               d_info);
         if (cusolstat != CUSOLVER_STATUS_SUCCESS) {
            printf("[chol_test][error] Failed to launch cuSOLVER potrf\n");
            return 1;
         }
         // Wait for completion
         cuerr = cudaStreamSynchronize(stream);
         if (cuerr != cudaSuccess) {
            std::cout << "[chol_test][error] Failed to synchronize stream "
                      << "(" << cudaGetErrorString(cuerr) << ")" << std::endl;
            std::exit(1);
         }

         end = std::chrono::high_resolution_clock::now();

         int info;
         // Retrieve info value on the device
         cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

         std::cout << "[chol_test] cusolstat = " << cusolstat << std::endl;
         std::cout << "[chol_test] info = " << info << std::endl;

         // Cleanup memory
         cusolstat = cusolverDnDestroy(cusolhandle);

      }
      else if (algo == sylver::tests::SyLVER) { 

         cublasStatus_t custat; // CuBLAS status
         cublasHandle_t cuhandle;
         inform_t inform;

         // Initialize factrization
         custat = cublasCreate(&cuhandle);
         if (custat != CUBLAS_STATUS_SUCCESS) {    
            std::cout << "[chol_test][error] Failed to create cuBLAS handle "
                      << "(" << custat << ")" << std::endl;
            std::exit(1);
         }

         cublasSetStream(cuhandle, stream);
      
         start = std::chrono::high_resolution_clock::now();

         sylver::spldlt::gpu::factor(cuhandle, m, m, d_l, lda, inform);
         cuerr = cudaStreamSynchronize(stream);
         if (cuerr != cudaSuccess) {
            std::cout << "[chol_test][error] Failed to synchronize stream"
                      << "(" << cudaGetErrorString(cuerr) << ")" << std::endl;
            std::exit(1);
         }
      
         end = std::chrono::high_resolution_clock::now();

         std::cout << "[chol_test] flag = " << inform.flag << std::endl;

         cublasDestroy(cuhandle);

      }
      else {
         std::cout << "[chol_test] Algo NOT implemented " << std::endl;
         return 1;
      }

      // Calculate walltime
      long ttotal =  
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (end-start).count();

      // Get matrix into host memory      
      custat = cublasGetMatrix(m, m, sizeof(T), d_l, lda, l, lda);
      // cudaMemcpy(l, d_l, lda*m*sizeof(T), cudaMemcpyDeviceToHost);

      cudaDeviceSynchronize();

      // ::spldlt::tests::print_mat_unsym("%12.3e", m, l, lda);

      // Check result
      int nrhs = 1;
      int ldsoln = m;
      T *soln = new T[nrhs*ldsoln];
      for(int r=0; r<nrhs; ++r)
         memcpy(&soln[r*ldsoln], b, m*sizeof(T));

      // spral::ssids::cpu::cholesky_solve_fwd(m, m, l, lda, nrhs, soln, ldsoln);
      // host_trsm<T>(SIDE_LEFT, FILL_MODE_LWR, OP_N, DIAG_NON_UNIT, m-n, nrhs, 1.0, &l[n*lda+n], lda, &soln[n], ldsoln);
      // host_trsm<T>(SIDE_LEFT, FILL_MODE_LWR, OP_T, DIAG_NON_UNIT, m-n, nrhs, 1.0, &l[n*lda+n], lda, &soln[n], ldsoln);
      // spral::ssids::cpu::cholesky_solve_bwd(m, m, l, lda, nrhs, soln, ldsoln);

      // Fwd substitution
      ::spldlt::host_trsm(
            ::spldlt::SIDE_LEFT, ::spldlt::FILL_MODE_LWR,
            ::spldlt::OP_N, ::spldlt::DIAG_NON_UNIT,
            m, nrhs,
            (T) 1.0,
            l, lda,
            soln, ldsoln);

      // Bwd substitution
      ::spldlt::host_trsm(
            ::spldlt::SIDE_LEFT, ::spldlt::FILL_MODE_LWR,
            ::spldlt::OP_T, ::spldlt::DIAG_NON_UNIT,
            m, nrhs,
            (T) 1.0,
            l, lda,
            soln, ldsoln);

      // Print useful info
      printf("Host to Device matrix transfer (s) = %e\n", 1e-9*t_transfer);
      
      T bwderr = sylver::tests::backward_error(m, a, lda, b, 1, soln, m);
      printf("bwderr = %le\n", bwderr);
 
      double flops = ((double)m*m*m)/3.0;
      printf("factor time (s) = %e\n", 1e-9*ttotal);
      printf("GFlop/s = %.3f\n", flops/(double)ttotal);

      // Cleanup memory

      // cudaStreamDestroy(stream);

      // cudaFree(d_work);
      cudaFree(d_l);
      
      delete[] a;
      delete[] b;
      delete[] l;

      return failed ? 1 : 0;

   }
   
}}
