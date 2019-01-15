/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER tests
#include "common.hxx"
#include "kernels/wrappers.hxx"
#include "kernels/gpu/wrappers.hxx"

// STD
#include <iostream>

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/kernels/cholesky.hxx"
// SSIDS tests
#include "tests/ssids/kernels/framework.hxx"
// CUDA, CuBLAS and CuSOLVER
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusolverDn.h>

namespace sylver {
namespace tests {

   
   
   
   // template<>
   // cusolverStatus_t cusolverDnSpotrf_bufferSize(
   //       cusolverDnHandle_t handle,
   //       cublasFillMode_t uplo,
   //       int n,
   //       float *A,
   //       int lda,
   //       int *Lwork );
   
   template<typename T>
   int chol_test(int m) {

      bool failed = false;

      T* a = nullptr;
      T* b = nullptr;
      T* l = nullptr;

      // Generate test matrix
      int lda = spral::ssids::cpu::align_lda<T>(m);
      a = new T[m*lda];
      sylver::tests::gen_posdef(m, a, lda);

      // Generate a RHS based on x=1, b=Ax
      b = new T[m];
      sylver::tests::gen_rhs(m, a, lda, b);

      // Copy a into l
      l = new T[m*lda];
      memcpy(l, a, lda*m*sizeof(T));

      cudaError_t cuerr;
      cublasStatus_t custat;

      T *d_l = nullptr;
      cuerr = cudaMalloc((void**)&d_l, m*lda*sizeof(T));
      if (cuerr != cudaSuccess) {
         printf("[chol_test] CUDA memory allocation error\n");
         return -1;
      }

      cusolverStatus_t cusolstat;
      cusolverDnHandle_t cusolhandle;
      cusolstat = cusolverDnCreate(&cusolhandle);
      T *d_work;
      int worksz; // Workspace size
      cusolstat = dev_potrf_buffersize(cusolhandle, CUBLAS_FILL_MODE_LOWER, m, d_l, lda, &worksz);
      std::cout << "[chol_test] work size = " << worksz << std::endl;
      cuerr = cudaMalloc((void**)&d_work, worksz*sizeof(T));
 
      // Send matrix to device
      custat = cublasSetMatrix(m, m, sizeof(T), l, lda, d_l, lda);

      int info;
      cusolstat = dev_potrf(
            cusolhandle, CUBLAS_FILL_MODE_LOWER, 
            m, 
            d_l, lda, 
            d_work, worksz,
            &info);
      
      custat = cublasGetMatrix(m, m, sizeof(T), d_l, lda, l, lda);

      cusolstat = cusolverDnDestroy(cusolhandle);
      
      // Check results

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
      spldlt::host_trsm(
            spldlt::SIDE_LEFT, spldlt::FILL_MODE_LWR,
            spldlt::OP_N, spldlt::DIAG_NON_UNIT,
            m, nrhs,
            (T) 1.0,
            l, lda,
            soln, ldsoln);

      // Bwd substitution
      spldlt::host_trsm(
            spldlt::SIDE_LEFT, spldlt::FILL_MODE_LWR,
            spldlt::OP_T, spldlt::DIAG_NON_UNIT,
            m, nrhs,
            (T) 1.0,
            l, lda,
            soln, ldsoln);

      T bwderr = sylver::tests::backward_error(m, a, lda, b, 1, soln, m);
      printf("bwderr = %le\n", bwderr);
      
      // Cleanup memory

      cudaFree(d_work);
      cudaFree(d_l);
      
      delete[] a;
      delete[] b;
      delete[] l;

      return failed ? -1 : 0;

   }
   
}}
