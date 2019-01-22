/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "sylver_ciface.hxx"

// STD
#include <iostream>
#include <string>

// CUDA
#include <cuda.h>
#include "cublas_v2.h"

namespace sylver {
namespace gpu {

   // @brief Check CUDA error, set inform flag and exit if error is
   // detected
   inline void cuda_check_error(cudaError_t cuerr, std::string fname, inform_t& inform) {
      if (cuerr != cudaSuccess) {
         std::cout << "[" << fname << "][CUDA error] "
                   << "(" << cudaGetErrorString(cuerr) << ")" << std::endl;
         inform.flag = ERROR_CUDA_UNKNOWN;
         std::exit(1);
      }
   }

   // @brief Check CUDA error and exit if error is detected
   inline void cuda_check_error(cudaError_t cuerr, std::string fname) {
      if (cuerr != cudaSuccess) {
         std::cout << "[" << fname << "][CUDA error] "
                   << "(" << cudaGetErrorString(cuerr) << ")" << std::endl;
         std::exit(1);
      }
   }

   // @brief Check cuBLAS error and exit if error is detected
   inline void cublas_check_error(cublasStatus_t custat, std::string fname, inform_t& inform) {
      if (custat != CUBLAS_STATUS_SUCCESS) {    
         std::cout << "[" << fname << "][cuBLAS error] "
                   << "(" << custat << ")" << std::endl;
         inform.flag = ERROR_CUBLAS_UNKNOWN;
         std::exit(1);
      }
   }

   // @brief Check cuBLAS error and exit if error is detected
   inline void cublas_check_error(cublasStatus_t custat, std::string fname) {
      if (custat != CUBLAS_STATUS_SUCCESS) {    
         std::cout << "[" << fname << "][cuBLAS error] "
                   << "(" << custat << ")" << std::endl;
         std::exit(1);
      }
   }

}} // End of namespace sylver::gpu
