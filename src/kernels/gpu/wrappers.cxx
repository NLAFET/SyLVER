#include "kernels/gpu/wrappers.hxx"

// CuSOLVER
#include <cusolverDn.h>

namespace sylver {

   // _POTRF BufferSize
   template<>
   cusolverStatus_t dev_potrf_buffersize<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *a, int lda, int *lwork) {
      return cusolverDnSpotrf_bufferSize(handle, uplo, n, a, lda, lwork);
   }
   template<>
   cusolverStatus_t dev_potrf_buffersize<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *a, int lda, int *lwork) {
      return cusolverDnDpotrf_bufferSize(handle, uplo, n, a, lda, lwork);
   }

   // _POTRF
   template<>
   cusolverStatus_t dev_potrf<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *a, int lda, float *work, int lwork, int *info) {
      return cusolverDnSpotrf(handle, uplo, n, a, lda, work, lwork, info);
   }
   template<>
   cusolverStatus_t dev_potrf<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *a, int lda, double *work, int lwork, int *info) {
      return cusolverDnDpotrf(handle, uplo, n, a, lda, work, lwork, info);
   }
   
}
