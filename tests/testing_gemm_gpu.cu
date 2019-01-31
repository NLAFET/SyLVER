/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

// STD
#include <iostream>
#include <string>
// CUTLASS
#if defined(HAVE_CUTLASS)
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/sgemm_traits.h"
#include "cutlass/gemm/wmma_gemm_traits.h"
#endif

namespace sylver {
namespace tests {   

   template<typename T>
   cudaError_t cutlass_gemm_test(
         cudaStream_t stream,
         int m, int n, int k, T alpha,
         T *a, int lda, T *b, int ldb, T beta,
         T *c, int ldc) {
      std::string context = "cutlass_gemm_test"; 
      std::cout << "[" << context << "] NOT implemented for working prec" << std::endl;
   }

   template<>
   cudaError_t cutlass_gemm_test<float>(
         cudaStream_t stream,
         int m, int n, int k, float alpha,
         float *a, int lda, float *b, int ldb, float beta, float *c, int ldc) {
      std::string context = "cutlass_gemm_test"; 
      std::cout << "[" << context << "]" << std::endl;

      cudaError_t cuerr;

      // typedef cutlass::gemm::SgemmTraits<
      //    cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
      //    cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
      //    cutlass::Shape<8, 128, 128>            // threadblock tile size
      //    >
      //    GemmTraits;

      // typedef cutlass::gemm::WmmaGemmTraits<
      //    cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
      //    cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
      //    // cutlass::Shape<8, 128, 128>,            // threadblock tile size
      //    // cutlass::Shape<64, 128, 128>,            // threadblock tile size
      //    cutlass::Shape<128, 128, 128>,            // threadblock tile size
      //    float, float, float,
      //    cutlass::gemm::LinearScaling<float>,
      //    float, // Accumulator type
      //    cutlass::Shape<128, 32, 32>, // Shap of warp-level GEMM (K-by-N-by-M)
      //    cutlass::Shape<16, 16, 16>, // The shape of the WMMA instruction
      //    1, // scalars every time a thread loads from A
      //    1  // scalars every time a thread loads from B
      //    >
      //    GemmTraits;

      typedef cutlass::gemm::WmmaGemmTraits<
         cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
         cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
         cutlass::Shape<128, 128, 128>,         // threadblock tile size
         signed char,                           // A type
         signed char,                           // B type
         int,                                   // D type
         cutlass::gemm::LinearScaling<int>,     // functor to do the math in the epilogue
         int,                                   // accumulator type
         cutlass::Shape<128, 32, 32>,           // warp tile size
         cutlass::Shape<16, 16, 16>,            // WMMA instruction tile size
         16,                                    // scalars every time a thread loads from A
         16                                     // scalars every time a thread loads from B
         >
         GemmTraits;
      
      typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

      // Gemm operation parameters 
      // typename Gemm::Params params;

      // int result = params.initialize(
      //       m,     // GEMM M dimension
      //       n,     // GEMM N dimension
      //       k,     // GEMM K dimension
      //       alpha, // scalar alpha
      //       a,     // matrix A operand
      //       lda,
      //       b,     // matrix B operand
      //       ldb,
      //       beta,  // scalar beta
      //       c,     // source matrix C
      //       ldc,
      //       c,     // destination matrix C (may be different memory than source C matrix)
      //       ldc
      //       );

      // if (result) {
      //    std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
      //    return cudaErrorInvalidValue;
      // }

      // // Launch kernel
      // cuerr = Gemm::launch(params, stream);

      return cuerr;
   }
   
}} // End of namespace sylver::tests
