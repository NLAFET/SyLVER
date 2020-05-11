/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Jonathan Hogg
/// @author Florent Lopez
#pragma once

#include "kernels/gpu/wrappers.hxx"
#include "kernels/gpu/common.hxx"

namespace sylver {
namespace spldlt {
namespace cuda {

template<typename T>
class Chol {

public:
   
   static void solve(
         cublasHandle_t cuhandle,
         int m, int n,
         T const* akk, int ld_akk,
         T* aik, int ld_aik) {

      std::string context = "Chol::solve";
      cublasStatus_t custat;

      T alpha = T(1.0);
      
      custat = sylver::gpu::dev_trsm(
         cuhandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
         CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
         m, n,
         &alpha,
         akk, ld_akk,
         aik, ld_aik);

      sylver::gpu::cublas_check_error(custat, context);
   }
};
   
}}}
