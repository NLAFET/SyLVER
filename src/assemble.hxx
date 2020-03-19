/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author    Florent Lopez
#pragma once

// SyLVER
#include "SymbolicFront.hxx"
#include "NumericFront.hxx"
#include "tasks/indef.hxx"
#include "tasks/tasks.hxx"

namespace sylver {
namespace spldlt {

   ////////////////////////////////////////////////////////////////////////////////   
   // fini_cnodes

   template <typename T, typename PoolAlloc>
   void fini_cnodes(
         NumericFront<T,PoolAlloc>& node,
         bool posdef) {

      // Deactivate children fronts
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {

         sylver::SymbolicFront const& csnode = child->symb();

         if (csnode.exec_loc == -1) {
            // fini_node(*child);
            fini_node_task(*child, posdef);
            // #if defined(SPLDLT_USE_STARPU)
            //             starpu_task_wait_for_all();
            // #endif
         }
#if defined(SPLDLT_USE_STARPU)
         // Unregister symbolic handle on child node
         child->unregister_submit_symb();
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

      int blksz = node.blksz();

      // Assemble front: non fully-summed columns i.e. contribution block 
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {

         sylver::SymbolicFront const& child_sfront = child->symb();
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

               // Serial assembly of contrib block into parent fronts.
               // assemble_contrib_subtree_task(
               //       node, child_sfront, child_contrib,
               //       child_sfront.contrib_idx, child_sfront.map,
               //       ASSEMBLE_PRIO);               

               // #if defined(SPLDLT_USE_STARPU)
               //             starpu_task_wait_for_all();
               // #endif

               // Blocked assembly of contrib block into parent
               // fronts.

               int const cn = child->nrow() - child->ncol();
               // Number of blocks in contrib
               int const nblk = ((cn - 1) / child->blksz()) + 1;

               // Loop over block-columns in `child` node contrib
               // contribution
               for (int jj = 0; jj < nblk; ++jj) {
                  // Loop over sub-diag block-rows in `child` node
                  // contributions
                  for (int ii = jj; ii < nblk; ++ii) {
                     assemble_contrib_subtree_block_task(
                           node, child_sfront, ii, jj,
                           child_contrib, child_sfront.contrib_idx,
                           ASSEMBLE_PRIO);
                  }
               }
               
            }
            else {                     

               int cncol = child->ncol();
               int cnrow = child->nrow();

               int csa = cncol / blksz;
               // Number of block rows in child node
               int cnr = child->nr();
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

      int blksz = node.blksz();
      sylver::SymbolicFront snode = node.symb();

      int nrow = node.nrow();
      int ncol = node.ncol();
      size_t ldl = align_lda<double>(nrow);

      /*
       * Add children
       */
      // Initialize index of delayed column in the parent
      int delay_col = snode.ncol;

      // printf("[assemble]\n");

      // Allocate mapping array
      // int *map = new int[n+1];
      // TODO: Use workspace for mapping
      std::vector<int, PoolAllocInt> map(n+1, PoolAllocInt(pool_alloc));

      // build lookup vector, allowing for insertion of delayed vars
      // Note that while rlist[] is 1-indexed this is fine so long as lookup
      // is also 1-indexed (which it is as it is another node's rlist[]
      for(int i=0; i<snode.ncol; i++)
         map[ snode.rlist[i] ] = i;
      for(int i=snode.ncol; i<snode.nrow; i++)
         map[ snode.rlist[i] ] = i + node.ndelay_in();
      
      // Assemble front: fully-summed columns 
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {

         sylver::SymbolicFront &csnode = child->symb(); // Children symbolic node

         int cm = csnode.nrow - csnode.ncol;
         csnode.map = new int[cm];
         for (int i=0; i<cm; i++)
            csnode.map[i] = map[ csnode.rlist[csnode.ncol+i] ];

         int ldcontrib = csnode.nrow - csnode.ncol;
         if (csnode.exec_loc == -1) {
            //
            // Assemble contributions from children front
            //

            // Assemble delays
            
            // assemble_delays(*child, delay_col, node);
            assemble_delays_task(*child, delay_col, node);

            delay_col += child->ndelay_out();

            // Handle expected contributions (only if something there)
            if (ldcontrib>0) {
               // int *cache = new int[cm];
               // spral::ssids::cpu::assemble_expected(0, cm, node, *child, map, cache);
               // delete cache;

               int cncol = child->ncol();

               int csa = cncol / blksz;
               int cnr = child->nr(); // number of block rows
               // Loop over blocks in contribution blocks
               for (int jj = csa; jj < cnr; ++jj) {
                  for (int ii = jj; ii < cnr; ++ii) {

                     // Kernel call (synchronous)
                     // assemble_block(node, *child, ii, jj, csnode.map);

                     // Task call (asynchronous)
                     assemble_block_task(
                           node, *child, ii, jj, csnode.map, ASSEMBLE_PRIO);
                  }
               }
            }

         }
         else {
            //
            // Assemble contributions from subtree
            //
            
            // Assemble delays

            
            // Kernel call (synchronous)
            // assemble_delays_subtree(
            //       node, csnode, child_contrib, csnode.contrib_idx, delay_col);

            // Task call (asynchronous)
            assemble_delays_subtree_task(
                  node, csnode, child_contrib, csnode.contrib_idx, delay_col);
            
            // Retreive contribution block from subtrees
            int cn, ldcontrib, ndelay, lddelay;
            double const *cval, *delay_val;
            int const *crlist, *delay_perm;
            spral_ssids_contrib_get_data(
                  child_contrib[csnode.contrib_idx], &cn, &cval, &ldcontrib, &crlist,
                  &ndelay, &delay_perm, &delay_val, &lddelay
                  );

            // Update position (i.e. column index) of first delayed
            // column from current children node to the parent front
            delay_col += ndelay;

            // Assemble expected contributions

            // Kernel call (synchronous)
            // assemble_subtree(node, csnode, child_contrib, csnode.contrib_idx);

            // Task call (asynchronous)
            // Serial assembly
            assemble_subtree_task(
                  node, csnode, child_contrib, csnode.contrib_idx,
                  csnode.map, ASSEMBLE_PRIO);

         }
// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif         
      } // Loop over children nodes

   } // assemble

}} // End of namespace sylver::spldlt
