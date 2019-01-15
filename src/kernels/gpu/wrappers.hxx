/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author    Jonathan Hogg

#pragma once

// CuSOLVER
#include <cusolverDn.h>

namespace sylver {

   // _POTRF BufferSize
   template <typename T> 
   cusolverStatus_t dev_potrf_buffersize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T *a, int lda, int *lwork);

   // _POTRF
   template <typename T>
   cusolverStatus_t dev_potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T *a, int lda, T *work, int lwork, int *info);

}
