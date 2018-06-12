#pragma once

// SpLDLT
#include "common.hxx"
#include "kernels/wrappers.hxx"

// STD
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <chrono>

// SSIDS
#include "ssids/cpu/kernels/common.hxx"
#include "ssids/cpu/kernels/ldlt_tpp.hxx"
#include "ssids/cpu/kernels/calc_ld.hxx"
// SSIDS tests
#include "tests/ssids/kernels/framework.hxx"
// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
// cuBLAS
#include "cublas_v2.h"


namespace spldlt { namespace tests {

      template<typename T, bool debug = false>
      int update_block_test(
            int m, int n, int k,
            int test=0, int seed=0) {

         bool failed = false;

         T u = 0.01;
         T small = 1e-20;
         
         printf("[update_block_gpu_test] m = %d, n = %d, k = %d\n", m, n, k);

         // Create matrix for generating D
         int ldc = k;
         T* c = new T[k*ldc];

         bool posdef = true;
         if (posdef) gen_posdef(k, c, ldc);
         else        gen_sym_indef(k, c, ldc);


         T* d = new T[2*k];

         T *tmp = new T[2*k];
         int* perm = new int[k];
         for(int i=0; i<k; i++) perm[i] = i;

         int nelim = 0;

         nelim += spral::ssids::cpu::ldlt_tpp_factor(
               k, k, perm, c, ldc, d, tmp, k, true, u, small, 0, c, k);

         delete[] tmp;
         delete[] perm;
         
         // Create matrix L_ij
         int ld_lij = m;
         T* l_ij = new T[ld_lij*n];         
         T* upd = new T[ld_lij*n]; // Copy of Lij         
         // Create matrix L_ik
         int ld_lik = m;
         T* l_ik = new T[ld_lik*k];
         // Create matrix L_jk
         int ld_ljk = n;
         T* l_jk = new T[ld_ljk*k];

         // Generate coefficients in blocks
         gen_mat(m, n, l_ij, ld_lij);
         memcpy(upd, l_ij, ld_lij*n*sizeof(T));
         gen_mat(m, k, l_ik, ld_lik);
         gen_mat(n, k, l_jk, ld_ljk);

         ////////////////////////////////////////
         // Compute udpate_block on CPU for checking
         T ralpha = -1.0;
         T rbeta = 1.0;

         // Compute L_ik D_k
         int ldld = m;
         T* ld = new T[ldld*k];
         spral::ssids::cpu::calcLD<spral::ssids::cpu::OP_N>(
               m, k, l_ik, ld_lik, d, ld, ldld);

         auto start_cpu = std::chrono::high_resolution_clock::now();
         // Compute U = U - W L^{T}
         host_gemm(
               OP_N, OP_T, m, n, k,
               // -1.0, ljk, ld_ljk, ld, ldld,
               -1.0, ld, ldld, l_jk, ld_ljk,
               rbeta, l_ij, ld_lij);
         auto end_cpu = std::chrono::high_resolution_clock::now();
         long ttotal_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu-start_cpu).count();


         ////////////////////////////////////////
         // Init GPU
         int dev = 0;

         auto start_init = std::chrono::high_resolution_clock::now();

         // Set device where to run kernel
         cudaSetDevice(dev);
         // Force initialization of GPU
         cudaFree(0);

         auto end_init = std::chrono::high_resolution_clock::now();
         long t_init = std::chrono::duration_cast<std::chrono::nanoseconds>(end_init-start_init).count();

         ////////////////////////////////////////
         // GPU data
         T* d_upd;
         T* d_l_ik;
         T* d_l_jk;

         T* d_d;
         T* d_ld;

         // Allocate memory on GPU
         cudaError_t cerr;
         cerr = cudaMalloc((void **) &d_upd, ld_lij*n*sizeof(T));
         cerr = cudaMalloc((void **) &d_l_jk, ld_ljk*k*sizeof(T));
         cerr = cudaMalloc((void **) &d_ld, ldld*k*sizeof(T));
         cerr = cudaMalloc((void **) &d_d, 2*k*sizeof(T));

         // Send data to the GPU
         cudaMemcpy(d_upd, upd, ld_lij*n*sizeof(T), cudaMemcpyHostToDevice);
         cudaMemcpy(d_l_jk, l_jk, ld_ljk*k*sizeof(T), cudaMemcpyHostToDevice);
         // cudaMemcpy(d_ld, ld, ldld*k*sizeof(T), cudaMemcpyHostToDevice);
         cudaMemcpy(d_d, d, 2*k*sizeof(T), cudaMemcpyHostToDevice);

         // Perform update on the GPU
         cublasHandle_t handle;
         cublasCreate(&handle);

         auto start = std::chrono::high_resolution_clock::now();

         

         cublasDgemm(
               handle,
               CUBLAS_OP_N, CUBLAS_OP_T,
               m, n, k,
               &ralpha, 
               d_ld, ldld, d_l_jk, ld_ljk,
               &rbeta,
               d_upd, ld_lij);               

         auto end = std::chrono::high_resolution_clock::now();
         long ttotal = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();

         cublasDestroy(handle);

         // Get result back from the GPU
         cudaMemcpy(upd, d_upd, m*n*sizeof(T), cudaMemcpyDeviceToHost);

         ////////////////////////////////////////
         // Check results

         T lij_norm = host_lange(spldlt::NORM_FRO, m, n, l_ij, ld_lij);
         host_axpy(ld_lij*n, -1.0, upd, 1.0, l_ij, 1.0);
         // host_axpy(ld_lij*n, -1.0, l_ij, 1.0, l_ij, 1.0);
         T err_norm = host_lange(spldlt::NORM_FRO, m, n, l_ij, ld_lij);

         // printf("[update_block_gpu_test] lij_norm = : %e\n", lij_norm);
         // printf("[update_block_gpu_test] err_norm = : %e\n", err_norm);
         printf("[update_block_gpu_test] rel err = %e\n", err_norm/lij_norm);

         ////////////////////////////////////////
         // Print results

         printf("[update_block_gpu_test] init = %e\n", 1e-9*t_init);
         printf("[update_block_gpu_test] t_gpu = %e\n", 1e-9*ttotal);
         printf("[update_block_gpu_test] t_cpu = %e\n", 1e-9*ttotal_cpu);
         
         ////////////////////////////////////////
         // Cleanup memory

         delete[] ld;

         cerr = cudaFree((void*)d_upd);
         cerr = cudaFree((void*)d_l_jk);
         cerr = cudaFree((void*)d_ld);

         delete[] c;
         delete[] l_ij;
         delete[] l_ik;
         delete[] l_jk;

         return failed ? -1 : 0;
      }

}} // namespace spldlt::tests
