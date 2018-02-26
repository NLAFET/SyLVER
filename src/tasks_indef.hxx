#pragma once

#include "ssids/cpu/cpu_iface.hxx"

#include "kernels/factor_indef.hxx"

#if defined(SPLDLT_USE_STARPU)
#include "StarPU/kernels_indef.hxx"
#include "StarPU/factor_indef.hxx"
#endif

namespace spldlt {

   ////////////////////////////////////////////////////////////////////////////////

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
      T *ld = new T[blksz*ldld];

      udpate_contrib_block(
            upd.m, upd.n, upd.a, upd.lda,  
            nelim, lik, ldl, ljk, ldl,
            (blk == 0), dk, ld, ldld);

      delete[] ld;

#endif
   }


   ////////////////////////////////////////////////////////////////////////////////
   // factor_front_indef_secondpass_nocontrib_task

   template <typename T, typename PoolAlloc>
   void factor_front_indef_secondpass_nocontrib_task(
         NumericFront<T, PoolAlloc> &node,
         std::vector<spral::ssids::cpu::Workspace> &workspaces,
         struct cpu_factor_options& options
         ) {

#if defined(SPLDLT_USE_STARPU)

      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> IntAlloc;
      int blksz = options.cpu_block_size;

      ColumnData<T, IntAlloc> &cdata = *node.cdata;
      int n = node.get_ncol();
      int const nblk = calc_nblk(n, blksz);
      
      insert_factor_front_indef_secondpass_nocontrib(
            cdata[nblk-1].get_hdl(), node.get_hdl(),
            &node, &workspaces, &options
            );

#else
      
      int m = node.get_nrow();
      int n = node.get_ncol();
      size_t ldl = align_lda<T>(m);
      T *lcol = node.lcol;
      T *d = &node.lcol[n*ldl];
      int *perm = node.perm;

      int nelim = 0;

      // Try to eliminate the columns uneliminated at first pass
      if (node.nelim < n) {
         // Use TPP factor to eliminate the remaining columns in the following cases:
         // 1) options.pivot_method is set to tpp;
         // 2) We are at a root node;
         // 3) options.failed_pivot_method is set to tpp.
         if (m==n || options.pivot_method==PivotMethod::tpp ||
             options.failed_pivot_method==FailedPivotMethod::tpp) {
            nelim = node.nelim;
            T *ld = new T[2*(m-nelim)]; // TODO: workspace
            node.nelim += ldlt_tpp_factor(
                  m-nelim, n-nelim, &perm[nelim], &lcol[nelim*(ldl+1)], ldl, 
                  &d[2*nelim], ld, m-nelim, options.action, options.u, options.small, 
                  nelim, &lcol[nelim], ldl);
            delete[] ld;
            printf("[factor_front_indef_nocontrib] second pass = %d out of %d\n", node.nelim, n);

         }
      }

      // Update number of delayed columns
      node.ndelay_out = n - node.nelim;
#endif
      
   }
   
} // namespace spldlt
