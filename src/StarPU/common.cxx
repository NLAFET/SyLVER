/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

// SyLVER
#include "common.hxx"
#if defined(SPLDLT_USE_GPU)
#include "kernels/gpu/common.hxx"
#endif
// STD
#include <string>
// STARPU
#include <starpu.h>
#if defined(SPLDLT_USE_GPU)
#include <cublas_v2.h>
#include <starpu_cublas_v2.h>
#endif

namespace sylver {
namespace starpu {

#if defined(SPLDLT_USE_GPU)

   // @brief Disable tensor cores on CUDA worker
   void disable_tc_cuworker(void *args) {
      
      std::string context = "disable_tc_cuworker";
      cublasStatus_t custat;
      cublasHandle_t cuhandle = starpu_cublas_get_local_handle();
      custat = cublasSetMathMode(cuhandle, CUBLAS_DEFAULT_MATH);
      sylver::gpu::cublas_check_error(custat, context);

   }

   // @brief Enable tensor cores on CUDA worker
   void enable_tc_cuworker(void *args) {
      
      std::string context = "disable_tc_cuworker";
      cublasStatus_t custat;
      cublasHandle_t cuhandle = starpu_cublas_get_local_handle();
      custat = cublasSetMathMode(cuhandle, CUBLAS_TENSOR_OP_MATH);
      sylver::gpu::cublas_check_error(custat, context);

   }

   // @brief Disable tensor cores for all CUDA workers
   void disable_tc() {
      // Disable TC on each CUDA worker via associated cuBLAS handle
      starpu_execute_on_each_worker(disable_tc_cuworker, nullptr, STARPU_CUDA);
   }

   // @brief Enable tensor cores for all CUDA workers
   void enable_tc() {
      // Disable TC on each CUDA worker via associated cuBLAS handle
      starpu_execute_on_each_worker(enable_tc_cuworker, nullptr, STARPU_CUDA);
   }

#endif

}} // End of namespace sylver::starpu
