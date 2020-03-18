/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "kernels/assemble.hxx"
#include "kernels/factor_indef.hxx"
#include "kernels/ldlt_app.hxx"
#include "Tile.hxx"
// STD
#include <assert.h>
// StarPU
#if defined(SPLDLT_USE_STARPU)
#include "StarPU/kernels_indef.hxx"
#include "StarPU/factor_indef.hxx"
#endif
// SSIDS 
#include "ssids/cpu/cpu_iface.hxx"

namespace sylver {
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
      int const nblk = node.nc(); // number of block columns in factors      
      
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
      int const blksz = node.blksz();
      int const ncol = node.ncol(); // Number of block-rows in destination node
      int const nr = node.nr(); // Number of block-rows in destination node
      int const nc = node.nc(); // Number of block-columns in destination node
      starpu_data_handle_t *hdls = new starpu_data_handle_t[nr*nc];
      int nh = 0; // Number of handles/blocks in node

      // Retreive contribution block from subtrees
      int cn, ldcontrib, ndelay, lddelay;
      double const *cval, *delay_val;
      int const *crlist, *delay_perm;
      spral_ssids_contrib_get_data(
            child_contrib[csnode.contrib_idx], &cn, &cval, &ldcontrib, &crlist,
            &ndelay, &delay_perm, &delay_val, &lddelay);
      
      if (ndelay <= 0) return;
         
      // Figure out blocks involved in the assembly so that we can
      // give this information to StarPU.
      int rr = -1;
      int cc = -1;

      // Here we collect the StarPU handles for all blocks for rows
      // and columns comprised between `delay_col` and `delay_col +
      // cnode.ndelay_out() - 1`. As a result, we might select more
      // blocks than necessary.
      
      // Row and Column index of first destination block for this
      // assembly
      int sa = delay_col / node.blksz();
      // Row and Column index of last destination block for this
      // assembly      
      int en = (delay_col+ndelay-1) / node.blksz();  
         
      for (int cc = 0; cc <= en; ++cc) {
         int br_sa = std::max(sa, cc);
         int br_en = -1;
         if (((cc+1)*blksz) < delay_col) {
            // We are in the fully-summed part of the node
            br_en = en;
         }
         else {
            // We are in both the fully-summed and non fully-summed
            // part of the node
            br_en = node.nr()-1; // Block row index
         }

         for (int rr = br_sa; rr <= br_en; ++rr) {
            assert(nh <= nr*nc);
            // hdls[nh] = node.blocks[cc*nr+rr].get_hdl();
            hdls[nh] = node.block_hdl(rr,cc);
            nh++;            
         }
      }

      // for(int j = 0; j < ndelay; j++) {

      //    int c = delay_col+j; // Destination column
      //    if (cc == (c/blksz)) continue;
      //    cc = c/blksz; // Destination block-column
      //    rr = -1; // Reset block-row index 

      //    int r;

      //    // loop over fully-summed rows
      //    // for(int i=0; i<ndelay-j; i++) {
      //    for(int i=j; i<ndelay; i++) {
      //       r = delay_col+i;
      //       if (rr == (r/blksz)) continue;
      //       rr = r/blksz; // Destination block-row
      //       assert(r >= c); // Make sure the coefficient is below
      //                         // the diagonal
      //       assert(rr >= cc); // Make sure the tile is below the
      //                         // diagonal
      //       hdls[nh] = node.blocks[cc*nr+rr].get_hdl();
      //       assert(hdls[nh] != nullptr); // Make sure the handle has
      //                                    // been registered
      //       nh++;
      //    }
      //    // loop over non fully-summed rows
      //    for(int i=0; i<cn; i++) {
      //       int r = csnode.map[i];
      //       if (rr == (r/blksz)) continue;
      //       rr = r/blksz; // Destination block-row index
      //       if (r < ncol) hdls[nh] = node.blocks[rr*nr+cc].get_hdl();
      //       else          hdls[nh] = node.blocks[cc*nr+rr].get_hdl();
      //       assert(hdls[nh] != nullptr);
      //    }
      // }
      
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

      if (cnode.ndelay_out() <= 0) return; // No task to submit if no delays

#if defined(SPLDLT_USE_STARPU)
      // Node info
      sylver::SymbolicFront &snode = node.symb(); // Child symbolic node
      int blksz = node.blksz();
      int ncol = node.ncol();
      int nr = node.nr(); // Number of block-rows in destination node
      int nc = node.nc(); // Number of block-columns in destination node
      // starpu_data_handle_t *hdls = new starpu_data_handle_t[nr*nc];
      // FIXME: check upper bound on number of StarPU handles for this array
      starpu_data_handle_t *hdls = new starpu_data_handle_t[nr*nr];
      int nh = 0; // Number of handles/blocks in node
      
      // Child node info
      sylver::SymbolicFront &csnode = cnode.symb(); // Child symbolic node
      int cnrow = cnode.nrow();
      int cnr = cnode.nr(); // Number of block-rows in child node
      int cnc = cnode.nc(); // Number of block-columns in child node
      starpu_data_handle_t *chdls = new starpu_data_handle_t[cnr*cnc];
      int nch = 0; // Number of handles/blocks in the child node

      int csa = cnode.nelim() / blksz; // Row and Column index for the
                                     // first block in delayed part of
                                     // children node

      // Add StarPU handles of sources blocks in the child node for
      // delayed columns
      //
      // Note: add blocks for both fully summed and non-fully summed
      // rows which must be delayed in the parent
      for (int jj=csa; jj<cnc; jj++) {
         for (int ii=jj; ii<cnr; ii++) {
            assert(nch <= cnr*cnc);
            chdls[nch] = cnode.blocks[jj*cnr+ii].get_hdl();
            nch++;
         }
      }

      // Add StarPU handles of destination blocks in node for delayed
      // columns
      int rr = -1;
      int cc = -1;

      // Here we gather the handles for all blocks for rows and
      // columns comprised between delay_col and
      // delay_col+cnode.ndelay_out()-1. As a result, we might select
      // more blocks than necessary.
      
      // Row and Column index of first destination block for this
      // assembly
      int sa = delay_col / node.blksz();
      // Row and Column index of last destination block for this
      // assembly      
      int en = (delay_col+cnode.ndelay_out()-1) / node.blksz();  
         
      for (int cc = 0; cc <= en; ++cc) {
         int br_sa = std::max(sa, cc);
         int br_en = -1;
         if (((cc+1)*blksz) < delay_col) {
            // We are in the fully-summed part of the node
            br_en = en;
         }
         else {
            // We are in both the fully-summed and non fully-summed
            // part of the node
            br_en = node.nr()-1; // Block row index
         }

         for (int rr = br_sa; rr <= br_en; ++rr) {
            assert(nh <= nr*nc);
            // hdls[nh] = node.blocks[cc*nr+rr].get_hdl();
            hdls[nh] = node.block_hdl(rr,cc);
            nh++;            
         }
      }
      
      // // Loop over columns in children node to be delayed to the
      // // parent node
      // for(int j = 0; j < cnode.ndelay_out(); j++) {

      //    int c = delay_col+j; // Destination column
      //    if (cc == (c/blksz)) continue;
      //    cc = c/blksz; // Column index for destination block
      //    rr = -1;

      //    // Loop over row coefficients of the j-th delayed column
      //    for (int i = cnode.nelim+j; i < cnode.nrow(); i++) {

      //       // Determine destination row index of the i-th row
      //       // coefficient of the j-th delayed column
      //       int r = -1;
      //       if (i < cnode.ncol()) {
      //          // i-th coefficient is in the fully-summed part of the
      //          // frontal matrix
      //          r = delay_col+i-cnode.nelim;
      //       }
      //       else {
      //          // i-th row coefficient is in the non fully-summed part
      //          // of the frontal matrix: use row mapping to determine
      //          // its index in the parent's node
      //          r = csnode.map[i-cnode.ncol()];
      //       }
      //       // int r = (i < cncol) ? delay_col+i-cnode.nelim : csnode.map[i-cncol];
            
      //       if (rr == (r/blksz)) continue;
      //       rr = r / blksz; // Row index for destination block
            
      //       if (r < delay_col) {
      //       // if (r < snode.ncol) {
      //       // if (r < node.ncol()) {
      //          assert(cc >= rr);
      //          assert(cc <= nr && rr <= nc);
      //          // assert(nh <= nr*nr);
      //          hdls[nh] = node.blocks[rr*nr+cc].get_hdl();
      //       }
      //       else {
      //          assert(rr >= cc);
      //          assert(rr <= nr && cc <= nc);
      //          // assert(nh <= nr*nr);
      //          hdls[nh] = node.blocks[cc*nr+rr].get_hdl();
      //       }
      //       nh++;            
      //    }
      // }
      
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
   
}} // namespace sylver::spldlt
