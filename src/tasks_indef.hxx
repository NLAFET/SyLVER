#pragma once

#include <assert.h>
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
         NumericFront<T, PoolAlloc>& node,
         int blk, int iblk, int jblk,
         std::vector<spral::ssids::cpu::Workspace>& workspaces,
         int blksz, int prio
         ) {

      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> IntAlloc;
      
#if defined(SPLDLT_USE_STARPU)

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
            cdata[blk].get_d_hdl(),
            cdata[nblk-1].get_hdl(), // make sure col has been processed
            node.contrib_hdl, // For synchronization purpose
            &node, blk, iblk, jblk, 
            &workspaces, prio);

#else

      spral::ssids::cpu::Workspace& work = workspaces[0];
      
      update_contrib_block_app<T, IntAlloc, PoolAlloc>(
            node, blk, iblk, jblk,
            isrc.get_a(), isrc.get_lda(),
            jsrc.get_a(), jsrc.get_lda(),
            upd.m, upd.n, upd.a, upd.lda, 
            work);

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
            cdata[nblk-1].get_hdl(), node.contrib_hdl,
            &node, &workspaces, &options, &worker_stats
            );

#else
      spral::ssids::cpu::Workspace& work = workspaces[0];
      ThreadStats& stats = worker_stats[0];
      factor_front_indef_failed(node, work, options, stats);
#endif
      
   }

   ////////////////////////////////////////////////////////////
   // Assemble delays task

   /// @brief Submit task for assembling delays from one node to its
   /// parent.
   template <typename T, typename PoolAlloc>
   void assemble_delays_task(
         NumericFront<T,PoolAlloc>& cnode,
         int delay_col,
         NumericFront<T,PoolAlloc>& node) {

      if (cnode.ndelay_out <= 0) return; // No task to submit if no delays

#if defined(SPLDLT_USE_STARPU)
      // Node info
      SymbolicFront &snode = node.symb; // Child symbolic node
      int blksz = node.blksz;
      int ncol = node.get_ncol();
      int nr = node.get_nr(); // Number of block-rows in destination node
      int nc = node.get_nc(); // Number of block-columns in destination node
      starpu_data_handle_t *hdls = new starpu_data_handle_t[nr*nc];
      int nh = 0; // Number of handles/blocks in node
      
      // Child node info
      SymbolicFront &csnode = cnode.symb; // Child symbolic node
      int cncol = cnode.get_ncol();
      int cnrow = cnode.get_nrow();
      int cnr = cnode.get_nr(); // Number of block-rows in child node
      int cnc = cnode.get_nc(); // Number of block-columns in child node
      starpu_data_handle_t *chdls = new starpu_data_handle_t[cnr*cnc];
      int nch = 0; // Number of handles/blocks in the child node

      int csa = cnode.nelim / blksz; // First block-column in 

      // Add block handles in the child node
      //
      // Note: add blocks for both fully summed and non-fully summed
      // rows which must be delayed in the parent
      for (int jj=csa; jj<cnc; jj++) {
         for (int ii=jj; ii<cnr; ii++) {
            chdls[nch] = cnode.blocks[jj*cnr+ii].get_hdl();
            nch++;
         }
      }

      // Add blocks in node
      int rr = -1;
      int cc = -1;
      for(int j = 0; j < cnode.ndelay_out; j++) {

         int c = delay_col+j; // Destination column
         if (cc == (c/blksz)) continue;
         cc = c/blksz; // Destination block-column
         rr = -1;

         for (int i = cnode.nelim+j; i < cnrow; i++) {

            int r = (i < cncol) ? delay_col+i-cnode.nelim : csnode.map[i-cncol];
            
            if (rr == (r/blksz)) continue;
            rr = r / blksz; // Destination block-row
            
            if (r < ncol) {
               hdls[nh] = node.blocks[rr*nr+cc].get_hdl();
            }
            else {
               hdls[nh] = node.blocks[cc*nr+rr].get_hdl();
            }
            nh++;
            
         }
      }
      
      assert(nch > 0); // Make sure the set is not empty
      assert(nh > 0);

      spldlt::starpu::insert_assemble_delays(
            chdls, nch, hdls, nh, &cnode, delay_col, &node);
      
      delete[] chdls;
      delete[] hdls;
#else

      assemble_delays(cnode, delay_col, node);
            
#endif
   }
   
} // namespace spldlt
