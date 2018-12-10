#pragma once

// SyLVER
#include "kernels/wrappers.hxx"

namespace spldlt {

   /// @brief Perform LU factorization of a block using partial pivoting
   template<typename T>
   factor_block_lu_pp(int m, int n, T *a, int lda, int *perm) {
      
      host_getrf(m, n, a, lda, perm);
   }


} // end of namespace spldlt
