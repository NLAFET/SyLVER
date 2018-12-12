#pragma once

#include "Block.hxx"
#include "kernels/factor_unsym.hxx"

namespace spldlt {

   template <typename T>
   void factor_block_lu_pp_task(BlockUnsym<T>& dblk, int *perm) {
      
      factor_block_lu_pp(
            dblk.m, dblk.n, perm, dblk.a, dblk.lda, dblk.b, dblk.ldb);
   }
}
