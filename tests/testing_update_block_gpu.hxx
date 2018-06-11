#pragma once

// STD
#include <cstdio>

// SSIDS
#include "ssids/cpu/kernels/ldlt_tpp.hxx"
// SSIDS tests
#include "tests/ssids/kernels/framework.hxx"
// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

namespace spldlt { namespace tests {

      template<typename T, bool debug = false>
      int update_block_test(
            int m, int n, int k,
            int test=0, int seed=0) {

         bool failed = false;

         T u = 0.01;
         T small = 1e-20;
         
         printf("[update_block_test] m = %d, n = %d, k = %d\n", m, n, k);

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
         T* l_ij = new T[m*n];         
         // Create matrix L_ij
         T* l_ik = new T[m*k];
         // Create matrix L_ij
         T* l_jk = new T[n*k];

         ////////////////////////////////////////
         // Init GPU
         int dev = 0;
         // Set device where to run kernel
         cudaSetDevice(dev);
         // Force initialization of GPU
         cudaFree(0);

         ////////////////////////////////////////
         // GPU data
         
         T* d_l_ij;
         T* d_l_ik;
         T* d_l_jk;

         T* d_ld;

         // Allocate memory on GPU
         cudaError_t cerr;
         cerr = cudaMalloc((void **) &d_l_ij, m*n*sizeof(T));
         cerr = cudaMalloc((void **) &d_l_ik, m*k*sizeof(T));
         cerr = cudaMalloc((void **) &d_l_jk, n*k*sizeof(T));

         // Send data to the GPU
         cudaMemcpy(d_l_ij, l_ij, m*n*sizeof(T), cudaMemcpyHostToDevice);
         cudaMemcpy(d_l_ik, l_ik, m*k*sizeof(T), cudaMemcpyHostToDevice);
         cudaMemcpy(d_l_jk, l_jk, n*k*sizeof(T), cudaMemcpyHostToDevice);

         // Perform update on the GPU
         
         
         // Get result back from the GPU
         cudaMemcpy(l_ij, d_l_ij, m*n*sizeof(T), cudaMemcpyDeviceToHost);
         
         ////////////////////////////////////////
         // Cleanup memory

         cerr = cudaFree((void*)d_l_ij);
         cerr = cudaFree((void*)d_l_ik);
         cerr = cudaFree((void*)d_l_jk);

         delete[] c;
         delete[] l_ij;
         delete[] l_ik;
         delete[] l_jk;

         return failed ? -1 : 0;
      }

}} // namespace spldlt::tests
