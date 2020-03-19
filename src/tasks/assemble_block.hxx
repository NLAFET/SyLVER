/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

namespace sylver {
namespace spldlt {

   ////////////////////////////////////////////////////////////
   // assemble_subtree_block_task

   template <typename T, typename PoolAlloc>   
   void assemble_subtree_block_task(
         NumericFront<T,PoolAlloc>& front, // Destination node 
         sylver::SymbolicFront const& csfront, // Root of the subtree
         int ii, int jj,
         void** child_contrib, 
         int contrib_idx, // Index of subtree to assemble
         int prio) {

      assert((ii >= 0) && (jj >= 0));
      assert(nullptr != csfront.map);
      
#if defined(SPLDLT_USE_STARPU)

      // Destination node info
      //
      int const blksz = front.blksz();
      int const ncol = front.ncol();
      // Number of block-rows in destination node
      int const nr = front.nr();
      // Number of block-columns in destination node
      int const nc = front.nc();

      // Block column index in destination node
      int cc = -1;
      // Block row index in destination node
      int rr = -1;

      // Source node info
      //
      // Contrib block dimensions 
      int const cn = csfront.nrow - csfront.ncol;
      
      // Array of StarPU handles for destination blocks
      starpu_data_handle_t *hdls = new starpu_data_handle_t[nr*nc];
      // Number of destination blocks involved in assembly operation
      int nh = 0;

      // Loop over columns of block-column `jj` 
      for(int j = jj*blksz;
          j < std::min((jj+1)*blksz,cn); ++j) {               

         int c = csfront.map[ j ]; // Destination column

         if (cc == (c/blksz)) continue;
         // Make sure we stay in fully-summed coefficients
         if (c < front.symb().ncol) {

            cc = c / blksz;
            rr = -1;

            for (int i = std::max(ii*blksz, j) ;
                 i < std::min((ii+1)*blksz,cn); ++i) {

               // Row mapping index (in parent front) of the i-th rows
               // in contribution block
               int r = csfront.map[ i ];
               
               if (rr==(r/blksz)) continue;
               rr = r/blksz;

               hdls[nh] = front.block_hdl(rr,cc);
               nh++;
               
            }

         }
      }

      if (nh>0) {
         sylver::spldlt::starpu::insert_subtree_assemble_block(
               &front, &csfront, front.hdl(), csfront.hdl, hdls, nh,
               child_contrib, contrib_idx, ii, jj);
      }

      delete[] hdls;

#else      

      assemble_subtree_block(
            front, csfront, child_contrib, contrib_idx, ii, jj);
#endif
      
   }

   ////////////////////////////////////////////////////////////
   // assemble_contrib_subtree_block_task

   // Assemble coefficient in contrib block `(ii, jj)` into parent
   // nodes
   template <typename T, typename PoolAlloc>
   void assemble_contrib_subtree_block_task(
         NumericFront<T,PoolAlloc>& node, // Destination node
         sylver::SymbolicFront const& csnode, // Root of the subtree
         int ii, int jj,
         void** child_contrib, 
         int contrib_idx, // Index of subtree to assemble
         int prio) {

#if defined(SYLVER_HAVE_STARPU)
         
      int const blksz = node.blksz();

      // Destination node info
      //
      int const ncol = node.ncol();
      int const nr = node.nr();
      int const rsa = ncol / blksz; // Rows/Cols index of first block in contrib 
      int const ncontrib = nr-rsa; // Number of block rows/cols in contrib

      starpu_data_handle_t *hdls = new starpu_data_handle_t[ncontrib*ncontrib];
      int nh = 0;

      // Source node info
      //

      // Contrib block dimensions
      int const cn = csnode.nrow - csnode.ncol;
      
      // Block column index in destination node
      int cc = -1;
      // Block row index in destination node
      int rr = -1;

      // Loop over columns of block-column `jj` 
      for(int j = jj*blksz;
          j < std::min((jj+1)*blksz,cn); ++j) {               

         int c = csnode.map[ j ]; // Destination column

         // Make sure column is whitin the nodes dimensions
         assert((c >= 0) && (c < node.nrow()));
         // Make sure column is not in the delays
         assert((c < node.symb().ncol) || (c >= node.ncol()));

         if (cc == (c/blksz)) continue;

         // Make sure destination column is in the contribution block
         if (c >= node.ncol()) {

            cc = c / blksz; // Destination block column
            rr = -1;

            // Loop over rows of block-row `ii`
            for (int i = std::max(ii*blksz, j) ;
                 i < std::min((ii+1)*blksz,cn); ++i) {

               int r = csnode.map[ i ]; // Destination row in parent front
               if (rr == (r/blksz)) continue;
               rr = r / blksz; // Destination block row

               hdls[nh] = node.contrib_block(rr, cc).hdl;
               nh++;               
            }
         }
      }
      
      // Insert assembly tasks if there are any contributions
      if (nh > 0) {
         sylver::spldlt::starpu::insert_subtree_assemble_contrib_block(
               &node, &csnode, ii, jj, node.hdl(), node.contrib_hdl(),
               csnode.hdl, hdls, nh, child_contrib, contrib_idx, prio);

      }
      
      delete[] hdls;

#else

      assemble_contrib_subtree_block(
            node, csnode, child_contrib, contrib_idx, ii, jj);

#endif

   }

   ////////////////////////////////////////////////////////////
   // @brief Launch task for assembling block (ii,jj) in contrib block
   // of cnode into node
   //
   // @param ii Row index of block in cnode
   // @param jj Column index of block in cnode
   template <typename T, typename PoolAlloc>   
   void assemble_block_task(
         NumericFront<T,PoolAlloc>& node, 
         NumericFront<T,PoolAlloc>& cnode, 
         int ii, int jj, int *cmap, int prio) {

#if defined(SPLDLT_USE_STARPU)

      // Node info
      sylver::SymbolicFront const& snode = node.symb();
      int const blksz = node.blksz();
      int const ncol = node.ncol();
      // Number of block-rows in destination node
      int const nr = node.nr();
      // Number of block-columns in destination node
      int const nc = node.nc();

      // StarPU handle array holding destination blocks handles
      starpu_data_handle_t *hdls = new starpu_data_handle_t[nr*nc]; // Upperbound nr*nc handles 
      int nh = 0;

      // Children node info
      // sylver::SymbolicFront const& csnode = cnode.symb;
      int cnrow = cnode.nrow();
      int cncol = cnode.ncol();
      int cm = cnrow-cncol;

      // colum indexes
      // int c_sa = (csnode.ncol > jj*blksz) ? 0 : (jj*blksz-csnode.ncol); // first col in block
      // int c_en = std::min((jj+1)*blksz-csnode.ncol, cm); // last col in block
      int c_sa = (cncol > jj*blksz) ? 0 : (jj*blksz-cncol); // first col in block
      int c_en = std::min((jj+1)*blksz-cncol, cm); // last col in block
      // row indexes
      // int r_en = std::min((ii+1)*blksz-csnode.ncol, cm); // last row in block
      int r_en = std::min((ii+1)*blksz-cncol, cm); // last row in block

      int cc = -1; // Block column index in destination node
      int rr = -1; // Block row index in destination node

      // loop over column in block
      for (int j=c_sa; j<c_en; ++j) {
         
         // Column index in parent node.
         int c = cmap[ j ];

         if (cc == (c/blksz)) continue;

         if (c < node.symb().ncol) {

            cc = c/blksz;
            rr = -1;

            // int r_sa = (ii == jj) ? j : (ii*blksz-csnode.ncol); // first row in block
            int r_sa = (ii == jj) ? j : (ii*blksz-cncol); // first row in block

            for (int i=r_sa; i<r_en; ++i) {

               // int r = map[ csnode.rlist[csnode.ncol+i] ];
               int r = cmap[ i ];
               if (rr == (r / blksz)) continue;
               rr = r/blksz;

               assert(rr < nr);
               assert(cc < nc);
               assert(nh < (nr*nc));

               // hdls[nh] = snode.handles[cc*nr+rr];
               // hdls[nh] = node.blocks[cc*nr+rr].get_hdl();
               hdls[nh] = node.block_hdl(rr, cc);

               nh++;
            }
         }
      }

      // Insert assembly tasks if there are contributions
      if (nh > 0) {
         
         // Contrib block to be assembled into current node
         sylver::Tile<T, PoolAlloc>& cb = cnode.contrib_block(ii, jj);

         spldlt::starpu::insert_assemble_block(
               &node, &cnode, ii, jj, cmap,
               cb.hdl,
               hdls, nh,
               node.hdl(), cnode.hdl(),
               prio);
      }

      delete[] hdls;

      // assemble_block(node, cnode, ii, jj, cmap, blksz);
#else

      assemble_block(node, cnode, ii, jj, cmap);

#endif
   }

   ////////////////////////////////////////////////////////////
   /// @brief Launch task for assembling block (ii,jj) in cnode into
   /// the contrib block of node
   ///
   /// @param ii Row index in front cnode
   /// @param jj Column index in front cnode
   template <typename T, typename PoolAlloc>   
   void assemble_contrib_block_task(
         NumericFront<T,PoolAlloc>& node, 
         NumericFront<T,PoolAlloc>& cnode, 
         int ii, int jj, int *cmap,
         std::vector<spral::ssids::cpu::Workspace>& workspaces,
         int prio) {

      int blksz = node.blksz();

#if defined(SPLDLT_USE_STARPU)

      // Node info
      sylver::SymbolicFront const& snode = node.symb();   
      int const nrow = node.nrow();
      int const ncol = node.ncol();
      int const nr = node.nr(); // Number of block-rows in destination node
      int const nc = node.nc(); // Number of block-columns in destination node
      int const rsa = ncol/blksz; // rows/cols index of first block in contrib 
      int const ncontrib = nr-rsa; // number of block rows/cols in contrib

      // StarPU handle array holding handles of destination block in parent front
      starpu_data_handle_t *hdls = new starpu_data_handle_t[ncontrib*ncontrib];
      int nh = 0;

      // Children node info
      // sylver::SymbolicFront const& csnode = cnode.symb;
      int const cnrow = cnode.nrow();
      int const cncol = cnode.ncol();

      int const cm = cnrow-cncol;

      // colum indexes
      // int c_sa = (csnode.ncol > jj*blksz) ? 0 : (jj*blksz-csnode.ncol); // first col in block
      // int c_en = std::min((jj+1)*blksz-csnode.ncol, cm); // last col in block
      int c_sa = (cncol > jj*blksz) ? 0 : (jj*blksz-cncol); // First col in block
      int c_en = std::min((jj+1)*blksz-cncol, cm); // Last col in block
      // row indexes
      // int r_en = std::min((ii+1)*blksz-csnode.ncol, cm); // last row in block
      int r_en = std::min((ii+1)*blksz-cncol, cm); // Last row in block

      int cc = -1;
      int rr = -1;

      // loop over column in block
      for (int j = c_sa; j < c_en; j++) {

         // Column index in parent node
         int c = cmap[ j ];

         if (cc == (c/blksz)) continue;

         if (c >= ncol) {

            cc = c/blksz;
            rr = -1;

            int r_sa = (ii == jj) ? j : (ii*blksz-cncol); // first row in block

            for (int i = r_sa; i < r_en; i++) {

               int r = cmap[ i ];
               if (rr == (r/blksz)) continue;
               rr = r/blksz;

               assert(nh < (ncontrib*ncontrib));
               
               // hdls[nh] = snode.contrib_handles[(cc-rsa)*ncontrib+(rr-rsa)];
               // hdls[nh] = node.contrib_blocks[(cc-rsa)*ncontrib+(rr-rsa)].hdl;
               hdls[nh] = node.contrib_block(rr, cc).hdl;
               nh++;
            }
         }
      }

      // Insert assembly tasks if there are contribution
      if (nh>0) {
         
         // Contrib block to assembled into node
         auto& cb = cnode.contrib_block(ii, jj);

         spldlt::starpu::insert_assemble_contrib_block(
               &node, &cnode, ii, jj, cmap, 
               cb.hdl,
               hdls, nh,
               node.hdl(), node.contrib_hdl(),
               cnode.hdl(),
               &workspaces,
               prio);

      }

      delete[] hdls;

#else

#if defined(MEMLAYOUT_1D)
      assemble_contrib_block_1d(node, cnode, ii, jj, cmap);
#else
      assemble_contrib_block(node, cnode, ii, jj, cmap);
#endif
      
#endif
   } 

}} // End of namespace sylver::spldlt
