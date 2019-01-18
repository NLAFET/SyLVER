/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author    Florent Lopez

namespace sylver {
namespace spldlt {
namespace gpu {

   // Perform the Cholesky factorization of a block-column matrix size
   // m x n with m >= n and n <= TILE_SIZE
   template<typename T>
   void factor_bcol(
         const cudaStream_t stream, int m, int n, T *const a, int lda, int *const stat);


   // Perform the Cholesky factorization of a matrix size m x n such
   // that m >= n
   template<typename T>
   void factor(
         const cudaStream_t stream, int m, int n, T *const a, int lda) {

      

   }
   
}}} // End of sylver::spldlt::gpu
