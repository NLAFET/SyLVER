#pragma once

#include "kernels/factor_indef.hxx"

#if defined(SPLDLT_USE_STARPU)
#include "StarPU/kernels_indef.hxx"
#endif
namespace spldlt {

   template <typename T, typename PoolAlloc>
   void update_contrib_indef_task(
         SymbolicFront const& snode,
         NumericFront<T, PoolAlloc> &node,
         int blk, int iblk, int jblk,
         int blksz, int prio
         ) {
      
#if defined(SPLDLT_USE_STARPU)


      int nrow = snode.nrow + node.ndelay_in;
      int ncol = snode.ncol + node.ndelay_in;
      int rsa = ncol / blksz; // index of first block in contribution blocks
      int nr = (nrow-1) / blksz + 1; // number of block rows
      int ncontrib = nr-rsa;

      insert_udpate_contrib_block_indef(
            node.contrib_blocks[(jblk-rsa)*ncontrib+(iblk-rsa)].hdl,
            snode.handles[blk*nr+iblk], snode.handles[blk*nr+jblk],
            &node, blk, iblk, jblk, blksz, prio);

#else

      int nrow = snode.nrow + node.ndelay_in;
      int ncol = snode.ncol + node.ndelay_in;
      int rsa = ncol / blksz; // index of first block in contribution blocks
      T *lcol = node.lcol;
      int ldl = align_lda<T>(nrow);
      T *d = &lcol[ncol*ldl];
      T *dk = &d[2*blk*blksz];
      int nelim = std::min(blksz, node.nelim - blk*blksz);
      int nr = (nrow-1) / blksz + 1; // number of block rows
      int ncontrib = nr-rsa;

      spldlt::Block<T, PoolAlloc>& upd =
         node.contrib_blocks[(jblk-rsa)*ncontrib+(iblk-rsa)];

      int ljk_first_row = std::max(jblk*blksz, ncol);
      T *ljk = &lcol[(blk*blksz)*ldl+ljk_first_row];
      
      int lik_first_row = std::max(iblk*blksz, ncol);
      T *lik = &lcol[(blk*blksz)*ldl+lik_first_row];

      int ldld = spral::ssids::cpu::align_lda<T>(blksz);
      T *ld = new T[blksz*ldld];

      udpate_contrib_block(
            upd.m, upd.n, upd.a, upd.lda,  
            nelim, lik, ldl, ljk, ldl,
            (blk == 0), dk, ld, ldld);

      delete[] ld;

#endif
   }

}
