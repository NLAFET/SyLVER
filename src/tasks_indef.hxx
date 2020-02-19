/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "factor_failed.hxx"
#include "kernels/ldlt_app.hxx"
#include "kernels/factor_indef.hxx"
#include "Tile.hxx"
// STD
#include <assert.h>
// StarPU
#if defined(SPLDLT_USE_STARPU)
#include "StarPU/kernels_indef.hxx"
#include "StarPU/factor_indef.hxx"
#include "StarPU/factor_failed.hxx"
#endif
// SSIDS 
#include "ssids/cpu/cpu_iface.hxx"

namespace spldlt {

   ////////////////////////////////////////////////////////////

   template <typename T, typename PoolAlloc, typename BlockSpec>
   void update_contrib_block_app_task(
         BlockSpec& isrc, BlockSpec& jsrc,
         sylver::Tile<T, PoolAlloc>& upd,
         NumericFront<T, PoolAlloc>& node,
         int blk, int iblk, int jblk,
         std::vector<spral::ssids::cpu::Workspace>& workspaces,
         int blksz, int prio
         ) {

      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> IntAlloc;
      
#if defined(SPLDLT_USE_STARPU)

      // int ncol = node.get_ncol();
      // int rsa = ncol / blksz; // index of first block in contribution blocks
      // int nr = node.get_nr(); // number of block rows
      // int ncontrib = nr-rsa;

      spldlt::ldlt_app_internal::ColumnData<T, IntAlloc> &cdata = *node.cdata;
      int const nblk = node.get_nc(); // number of block columns in factors      
      
      spldlt::starpu::insert_update_contrib_block_app(
            upd.hdl, isrc.get_hdl(), jsrc.get_hdl(),
            // node.contrib_blocks[(jblk-rsa)*ncontrib+(iblk-rsa)].hdl,
            // snode.handles[blk*nr+iblk], snode.handles[blk*nr+jblk],
            cdata[blk].get_d_hdl(),
            cdata[nblk-1].get_hdl(), // make sure col has been processed
            node.contrib_hdl(), // For synchronization purpose
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

   ////////////////////////////////////////////////////////////
   // Assemble subtree delays task

   /// @brief Submit task for assembling delays from one subtree to
   /// its parent.
   template <typename T, typename PoolAlloc>
   void assemble_delays_subtree_task(
         NumericFront<T,PoolAlloc>& node, // Destination node 
         sylver::SymbolicFront &csnode, // Root of the subtree
         void** child_contrib, 
         int contrib_idx, // Index of subtree to assemble
         int delay_col) {

#if defined(SPLDLT_USE_STARPU)

      // Node info
      int blksz = node.blksz();
      int ncol = node.get_ncol(); // Number of block-rows in destination node
      int nr = node.get_nr(); // Number of block-rows in destination node
      int nc = node.get_nc(); // Number of block-columns in destination node
      starpu_data_handle_t *hdls = new starpu_data_handle_t[nr*nc];
      int nh = 0; // Number of handles/blocks in node

      // Retreive contribution block from subtrees
      int cn, ldcontrib, ndelay, lddelay;
      double const *cval, *delay_val;
      int const *crlist, *delay_perm;
      spral_ssids_contrib_get_data(
            child_contrib[csnode.contrib_idx], &cn, &cval, &ldcontrib, &crlist,
            &ndelay, &delay_perm, &delay_val, &lddelay
            );
      
      if (ndelay <= 0) return;
         
      // Add blocks in node
      int rr = -1;
      int cc = -1;
      for(int j = 0; j < ndelay; j++) {

         int c = delay_col+j; // Destination column
         if (cc == (c/blksz)) continue;
         cc = c/blksz; // Destination block-column
         rr = -1; // Reset block-row index 

         int r;

         for(int i=0; i<ndelay-j; i++) {
            r = delay_col+i;
            if (rr == (r/blksz)) continue;
            rr = r/blksz; // Destination block-row
            hdls[nh] = node.blocks[cc*nr+rr].get_hdl();
            nh++;
         }
         for(int i=0; i<cn; i++) {
            int r = csnode.map[i];
            if (rr == (r/blksz)) continue;
            rr = r/blksz; // Destination block-row
            if (r < ncol) hdls[nh] = node.blocks[rr*nr+cc].get_hdl();
            else          hdls[nh] = node.blocks[cc*nr+rr].get_hdl();
         }
      }
      
      assert(nh > 0);

      if (nh > 0)
         spldlt::starpu::insert_assemble_delays_subtree(
               hdls, nh, csnode.hdl, &node, &csnode, child_contrib, contrib_idx,
               delay_col);

      delete[] hdls;
#else

      assemble_delays_subtree(
            node, csnode, child_contrib, contrib_idx, delay_col);

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

      // printf("[assemble_delays_task] idx = %d, cidx = %d, cnode.ndelay_out = %d\n", 
      //        node.symb.idx+1 , cnode.symb.idx+1 ,cnode.ndelay_out);
      if (cnode.ndelay_out() <= 0) return; // No task to submit if no delays

#if defined(SPLDLT_USE_STARPU)
      // Node info
      sylver::SymbolicFront &snode = node.symb(); // Child symbolic node
      int blksz = node.blksz();
      int ncol = node.get_ncol();
      int nr = node.get_nr(); // Number of block-rows in destination node
      int nc = node.get_nc(); // Number of block-columns in destination node
      // starpu_data_handle_t *hdls = new starpu_data_handle_t[nr*nc];
      // FIXME: check upper bound on number of StarPU handles for this array
      starpu_data_handle_t *hdls = new starpu_data_handle_t[nr*nr];
      int nh = 0; // Number of handles/blocks in node
      
      // Child node info
      sylver::SymbolicFront &csnode = cnode.symb(); // Child symbolic node
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
            assert(nch < cnr*cnc);
            chdls[nch] = cnode.blocks[jj*cnr+ii].get_hdl();
            nch++;
         }
      }

      // Add blocks in node
      int rr = -1;
      int cc = -1;
      for(int j = 0; j < cnode.ndelay_out(); j++) {

         int c = delay_col+j; // Destination column
         if (cc == (c/blksz)) continue;
         cc = c/blksz; // Destination block-column
         rr = -1;

         for (int i = cnode.nelim+j; i < cnrow; i++) {

            int r = (i < cncol) ? delay_col+i-cnode.nelim : csnode.map[i-cncol];
            
            if (rr == (r/blksz)) continue;
            rr = r / blksz; // Destination block-row
            
            if (r < snode.ncol) {
            // if (r < ncol) {
               assert(cc >= rr);
               assert(cc <= nr && rr <= nc);
               assert(nh < nr*nc);
               hdls[nh] = node.blocks[rr*nr+cc].get_hdl();
            }
            else {
               assert(rr >= cc);
               assert(rr <= nr && cc <= nc);
               // assert(nh < nr*nc);
               assert(nh < nr*nr);
               hdls[nh] = node.blocks[cc*nr+rr].get_hdl();
            }
            nh++;
            
         }
      }
      
      assert(nch > 0); // Make sure the set is not empty
      assert(nh > 0);

      if ((nh > 0) && (nch > 0))
         spldlt::starpu::insert_assemble_delays(
               chdls, nch, hdls, nh, &cnode, delay_col, &node);
      
      delete[] chdls;
      delete[] hdls;
#else

      assemble_delays(cnode, delay_col, node);
            
#endif
   }
   
} // namespace spldlt
