#pragma once

#include "SymbolicFront.hxx"
#include "NumericFront.hxx"
#include "tasks.hxx"

namespace spldlt {

   ////////////////////////////////////////////////////////////////////////////////   
   // assemble_contrib
   //
   // Assemble contributions from children node and subtrees into the
   // contribution blocks of node
   template <typename T, typename PoolAlloc>
   void assemble_contrib(
         NumericFront<T,PoolAlloc>& node,
         void** child_contrib) {

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
               int cnr = (cnrow-1) / blksz + 1; 
               // Lopp over blocks in contribution blocks
               for (int jj = csa; jj < cnr; ++jj) {                     
                  for (int ii = jj; ii < cnr; ++ii) {
                     // assemble_contrib_block(
                     //       node, *child, ii, jj, child_sfront.map, 
                     //       blksz);

                     assemble_contrib_block_task(
                           node, *child, ii, jj, 
                           child_sfront.map, ASSEMBLE_PRIO);
// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif

                  }
               }
            }
         }
      } // Loop over child nodes

   } // assemble_contrib

} // end of namespace spldlt
