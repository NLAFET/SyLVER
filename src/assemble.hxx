/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author    Florent Lopez
#pragma once

#include "SymbolicFront.hxx"
#include "NumericFront.hxx"
#include "tasks.hxx"

namespace spldlt {

   ////////////////////////////////////////////////////////////////////////////////   
   // fini_cnodes

   template <typename T, typename PoolAlloc>
   void fini_cnodes(NumericFront<T,PoolAlloc>& node) {

      // Deactivate children fronts
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {

         SymbolicFront const& csnode = child->symb;

         if (csnode.exec_loc == -1) {
            // fini_node(*child);
            fini_node_task(*child);
            // #if defined(SPLDLT_USE_STARPU)
            //             starpu_task_wait_for_all();
            // #endif
         }
#if defined(SPLDLT_USE_STARPU)
         // Unregister symbolic handle on child node
         starpu_data_unregister_submit(csnode.hdl);
#endif
      } // Loop over child nodes

   }

   ////////////////////////////////////////////////////////////////////////////////   
   // assemble_contrib
   //
   // Assemble contributions from children node and subtrees into the
   // contribution blocks of node
   template <typename T, typename PoolAlloc>
   void assemble_contrib(
         NumericFront<T,PoolAlloc>& node,
         void** child_contrib,
         std::vector<spral::ssids::cpu::Workspace>& workspaces
         ) {

      // printf("[assemble_contrib]\n");

      int blksz = node.blksz;

      // Assemble front: non fully-summed columns i.e. contribution block 
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {

         SymbolicFront& child_sfront = child->symb;
         // SymbolicFront &child_sfront = symb_[child->symb.idx];

         int ldcontrib = child_sfront.nrow - child_sfront.ncol;
         // Handle expected contributions (only if something there)
         // if (child->contrib) {
         if (ldcontrib>0) {
            // Skip iteration if child node is in a subtree
            if (child_sfront.exec_loc != -1) {                     
               // Assemble contribution block from subtrees into non
               // fully-summed coefficients
               // assemble_contrib_subtree(
               //       node, child_sfront, child_contrib, 
               //       child_sfront.contrib_idx, blksz);
// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif
               assemble_contrib_subtree_task(
                     node, child_sfront, child_contrib,
                     child_sfront.contrib_idx, child_sfront.map,
                     ASSEMBLE_PRIO);
// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif

            }
            else {                     

               int cncol = child->get_ncol();
               int cnrow = child->get_nrow();

               int csa = cncol / blksz;
               // Number of block rows in child node
               int cnr = child->get_nr();
               // Lopp over blocks in contribution blocks
               for (int jj = csa; jj < cnr; ++jj) {                     
                  for (int ii = jj; ii < cnr; ++ii) {
                     // assemble_contrib_block(
                     //       node, *child, ii, jj, child_sfront.map, 
                     //       blksz);

                     assemble_contrib_block_task(
                           node, *child, ii, jj, 
                           child_sfront.map, workspaces,
                           ASSEMBLE_PRIO);
// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif

                  }
               }
            }
         }
      } // Loop over child nodes

// #if defined(SPLDLT_USE_STARPU)
//       // Synchronize assembly tasks
//       spldlt::starpu::insert_assemble_contrib_sync(
//             node.contrib_hdl, node.symb.idx);
// #endif

   } // assemble_contrib

   ///////////////////////////////////////////////////////////   
   // @brief Assemble contributions from children node and subtrees
   // into the fully-summed columns
   template <typename T, typename PoolAlloc>
   void assemble(
         int n,
         NumericFront<T,PoolAlloc>& node,
         void** child_contrib,
         PoolAlloc const& pool_alloc
         ) {

      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;

      int blksz = node.blksz;
      SymbolicFront snode = node.symb;

      int nrow = node.get_nrow();
      int ncol = node.get_ncol();
      size_t ldl = align_lda<double>(nrow);

      /*
       * Add children
       */
      int delay_col = snode.ncol;

      // printf("[assemble]\n");

      // Allocate mapping array
      // int *map = new int[n+1];
      std::vector<int, PoolAllocInt> map(n+1, PoolAllocInt(pool_alloc));

      // build lookup vector, allowing for insertion of delayed vars
      // Note that while rlist[] is 1-indexed this is fine so long as lookup
      // is also 1-indexed (which it is as it is another node's rlist[]
      for(int i=0; i<snode.ncol; i++)
         map[ snode.rlist[i] ] = i;
      for(int i=snode.ncol; i<snode.nrow; i++)
         map[ snode.rlist[i] ] = i + node.ndelay_in;
      
      // Assemble front: fully-summed columns 
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {

         SymbolicFront &csnode = child->symb; // Children symbolic node

         int cm = csnode.nrow - csnode.ncol;
         csnode.map = new int[cm];
         for (int i=0; i<cm; i++)
            csnode.map[i] = map[ csnode.rlist[csnode.ncol+i] ];

         int ldcontrib = csnode.nrow - csnode.ncol;
         if (csnode.exec_loc == -1) {
            // Assemble contributions from child front

            // printf("[assemble] child->ndelay_out = %d\n", child->ndelay_out);

            /* Handle delays - go to back of node
             * (i.e. become the last rows as in lower triangular format) */
            // for(int i=0; i<child->ndelay_out; i++) {
            //    // Add delayed rows (from delayed cols)
            //    T *dest = &node.lcol[delay_col*(ldl+1)];
            //    int lds = align_lda<T>(csnode.nrow + child->ndelay_in);
            //    T *src = &child->lcol[(child->nelim+i)*(lds+1)];
            //    node.perm[delay_col] = child->perm[child->nelim+i];
            //    for(int j=0; j<child->ndelay_out-i; j++) {
            //       dest[j] = src[j];
            //    }
            //    // Add child's non-fully summed rows (from delayed cols)
            //    dest = node.lcol;
            //    src = &child->lcol[child->nelim*lds + child->ndelay_in +i*lds];
            //    for(int j=csnode.ncol; j<csnode.nrow; j++) {
            //       int r = map[ csnode.rlist[j] ];
            //       // int r = csnode.map[j];
            //       if(r < ncol) dest[r*ldl+delay_col] = src[j];
            //       else         dest[delay_col*ldl+r] = src[j];
            //    }
            //    delay_col++;
            // }

// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif      

            // assemble_delays(*child, delay_col, node);
            assemble_delays_task(*child, delay_col, node);
// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif      
            delay_col += child->ndelay_out;

            // Handle expected contributions (only if something there)
            if (ldcontrib>0) {
               // int *cache = new int[cm];
               // spral::ssids::cpu::assemble_expected(0, cm, node, *child, map, cache);
               // delete cache;

               int cncol = child->get_ncol();

               int csa = cncol / blksz;
               int cnr = child->get_nr(); // number of block rows
               // Loop over blocks in contribution blocks
               for (int jj = csa; jj < cnr; ++jj) {
                  for (int ii = jj; ii < cnr; ++ii) {
                     // assemble_block(node, *child, ii, jj, csnode.map);
                     assemble_block_task(
                           node, *child, ii, jj, csnode.map, ASSEMBLE_PRIO);
// #if defined(SPLDLT_USE_STARPU)
//                      starpu_task_wait_for_all();
// #endif
                  }
               }
// #if defined(SPLDLT_USE_STARPU)
//                starpu_task_wait_for_all();
// #endif
            }

         }
         else {
            // Assemble contributions from subtree

            // printf("[assemble] TETETET");

            // Assemble delays
            // assemble_delays_subtree(
            //       node, csnode, child_contrib, csnode.contrib_idx, delay_col);
            assemble_delays_subtree_task(
                  node, csnode, child_contrib, csnode.contrib_idx, delay_col);
// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif
            
            // Retreive contribution block from subtrees
            int cn, ldcontrib, ndelay, lddelay;
            double const *cval, *delay_val;
            int const *crlist, *delay_perm;
            spral_ssids_contrib_get_data(
                  child_contrib[csnode.contrib_idx], &cn, &cval, &ldcontrib, &crlist,
                  &ndelay, &delay_perm, &delay_val, &lddelay
                  );
            // delay_col += child->ndelay_out;
            delay_col += ndelay;
            // printf("[assemble] ndelay = %d, delay_col = %d\n", ndelay, delay_col);

            // Assemble contribution blocks
            // assemble_subtree(node, csnode, child_contrib, csnode.contrib_idx);
            assemble_subtree_task(
                  node, csnode, child_contrib, csnode.contrib_idx,
                  csnode.map, ASSEMBLE_PRIO);
// #if defined(SPLDLT_USE_STARPU)
//                      starpu_task_wait_for_all();
// #endif      

            // // int *cache = new int[cn];
            // // for(int j=0; j<cn; ++j)
            // //    cache[j] = map[ crlist[j] ];

            // // printf("[assemble] contrib_idx = %d, ndelay = %d\n", csnode.contrib_idx, ndelay);

            // /* Handle delays - go to back of node
            //  * (i.e. become the last rows as in lower triangular format) */
            // for(int i=0; i<ndelay; i++) {
            //    // Add delayed rows (from delayed cols)
            //    T *dest = &node.lcol[delay_col*(ldl+1)];
            //    T const* src = &delay_val[i*(lddelay+1)];
            //    node.perm[delay_col] = delay_perm[i];
            //    for(int j=0; j<ndelay-i; j++) {
            //       dest[j] = src[j];
            //    }
            //    // Add child's non-fully summed rows (from delayed cols)
            //    dest = node.lcol;
            //    src = &delay_val[i*lddelay+ndelay];
            //    for(int j=0; j<cn; j++) {
            //       // int r = cache[j];
            //       int r = csnode.map[j];
            //       if(r < ncol) dest[r*ldl+delay_col] = src[j];
            //       else         dest[delay_col*ldl+r] = src[j];
            //    }
            //    delay_col++;
            // }
            // if(!cval) continue; // child was all delays, nothing more to do
            // /* Handle expected contribution */
            // for(int j = 0; j < cn; ++j) {               
            //    int c = csnode.map[ j ]; // Destination column                  
            //    T const* src = &cval[j*ldcontrib];
            //    if (c < snode.ncol) {
            //       int ldd = node.get_ldl();
            //       T *dest = &node.lcol[c*ldd];

            //       for (int i = j ; i < cn; ++i) {
            //          // Assemble destination block
            //          dest[ csnode.map[ i ]] += src[i];
            //       }
            //    }
            // }

         }
      }
   } // assemble

} // end of namespace spldlt
