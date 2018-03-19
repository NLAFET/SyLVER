#pragma once

#include "ssids/cpu/cpu_iface.hxx"

#include "kernels/ldlt_app.hxx"
#include "kernels/factor_indef.hxx"

#if defined(SPLDLT_USE_STARPU)
#include "StarPU/kernels_indef.hxx"
#include "StarPU/factor_indef.hxx"
#endif

namespace spldlt {

   ////////////////////////////////////////////////////////////////////////////////

   template <typename T, typename PoolAlloc, typename BlockSpec>
   void update_contrib_block_app_task(
         BlockSpec isrc, BlockSpec jsrc,
         Tile<T, PoolAlloc>& upd, 
         NumericFront<T, PoolAlloc> &node,
         int blk, int iblk, int jblk,
         std::vector<spral::ssids::cpu::Workspace> &workspaces,
         int blksz, int prio
         ) {
      
#if defined(SPLDLT_USE_STARPU)

      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> IntAlloc;

      int ncol = node.get_ncol();
      int rsa = ncol / blksz; // index of first block in contribution blocks
      int nr = node.get_nr(); // number of block rows
      int ncontrib = nr-rsa;

      spldlt::ldlt_app_internal::ColumnData<T, IntAlloc> &cdata = *node.cdata;
      int const nblk = node.get_nc(); // number of block columns in factors      
      
      spldlt::starpu::insert_update_contrib_block_app(
            upd.hdl, isrc.get_hdl(), jsrc.get_hdl(),
            // node.contrib_blocks[(jblk-rsa)*ncontrib+(iblk-rsa)].hdl,
            // snode.handles[blk*nr+iblk], snode.handles[blk*nr+jblk],
            cdata[nblk-1].get_hdl(), // make sure col has been processed
            node.contrib_hdl, // For synchronization purpose
            &node, blk, iblk, jblk, 
            &workspaces,
            blksz, prio);

#else

      int nrow = node.get_nrow();
      int ncol = node.get_ncol();
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
      T *ld = new T[blksz*ldld]; // TODO: Use workspaces

      update_contrib_block(
            upd.m, upd.n, upd.a, upd.lda,  
            nelim, lik, ldl, ljk, ldl,
            (blk == 0), dk, ld, ldld);

      delete[] ld;

#endif
   }


   ////////////////////////////////////////////////////////////////////////////////
   // factor_front_indef_secondpass_nocontrib_task

   template <typename T, typename PoolAlloc>
   void factor_front_indef_failed_task(
         NumericFront<T, PoolAlloc> &node,
         std::vector<spral::ssids::cpu::Workspace> &workspaces,
         struct cpu_factor_options& options,
         std::vector<ThreadStats>& worker_stats
         ) {

#if defined(SPLDLT_USE_STARPU)

      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> IntAlloc;
      int blksz = options.cpu_block_size;

      spldlt::ldlt_app_internal::ColumnData<T, IntAlloc> &cdata = *node.cdata;
      int n = node.get_ncol();
      int const nblk = node.get_nc(); // Number of block-columns
      
      spldlt::starpu::insert_factor_front_indef_failed(
            cdata[nblk-1].get_hdl(), // node.get_hdl(),
            &node, &workspaces, &options, &worker_stats
            );

#else
      ThreadStats& stats = worker_stats[0];
      factor_front_indef_failed(node, work, options, stats);
#endif
      
   }
   
} // namespace spldlt
