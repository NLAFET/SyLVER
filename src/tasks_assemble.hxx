/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include <assert.h>

#include "assemble.hxx"

#if defined(SPLDLT_USE_STARPU)
#include "StarPU/assemble.hxx"
#include "StarPU/kernels_indef.hxx"
#endif

namespace spldlt {


   ////////////////////////////////////////////////////////////
   // fini child nodes

   template <typename T, typename PoolAlloc>
   void fini_cnodes_task(NumericFront<T, PoolAlloc>& node) {

#if defined(SPLDLT_USE_STARPU)

      int nchild = 0;
      for (auto* child=node.first_child; child!=NULL; child=child->next_child)
         nchild++;

      starpu_data_handle_t *cnode_hdls = new starpu_data_handle_t[nchild];

      int nc = 0;
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {
         cnode_hdls[nc] = child->symb.hdl;
         ++nc;
      }

      assert(nchild==nc);
      
      spldlt::starpu::insert_fini_cnodes(
            node.get_hdl(), cnode_hdls, nchild, &node);

      delete[] cnode_hdls;
#else

      fini_cnodes(node);

#endif

   }

   ////////////////////////////////////////////////////////////
   // assemble contribution block
   
   template <typename T, typename PoolAlloc>
   void assemble_contrib_task(
         NumericFront<T, PoolAlloc>& node,
         void** child_contrib,
         std::vector<spral::ssids::cpu::Workspace>& workspaces
         ){

#if defined(SPLDLT_USE_STARPU)

      int nchild = 0;
      for (auto* child=node.first_child; child!=NULL; child=child->next_child)
         nchild++;

      starpu_data_handle_t *cnode_hdls = new starpu_data_handle_t[nchild];
      
      int i = 0;
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {
         cnode_hdls[i] = child->symb.hdl;
         ++i;
      }
      // printf("[assemble_contrib_task] node = %d, nchild = %d\n", node.symb.idx+1, nchild);

      spldlt::starpu::insert_assemble_contrib(
            node.get_hdl(), cnode_hdls, nchild, //node.contrib_hdl,
            &node, child_contrib, &workspaces);

      delete[] cnode_hdls;

#else

      assemble_contrib(node, child_contrib, workspaces);

#endif
   }

   ////////////////////////////////////////////////////////////
   // assemble front
   template <typename T, typename PoolAlloc>
   void assemble_task(
         int n,
         NumericFront<T, PoolAlloc>& node,
         void** child_contrib,
         PoolAlloc& pool_alloc
         ) {

#if defined(SPLDLT_USE_STARPU)

      int nchild = 0;
      for (auto* child=node.first_child; child!=NULL; child=child->next_child)
         nchild++;

      starpu_data_handle_t *cnode_hdls = new starpu_data_handle_t[nchild];
      
      int i = 0;
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {
         cnode_hdls[i] = child->symb.hdl;
         ++i;
      }

      assert(nchild==i);
      
      // printf("[assemble_task] node = %d, nchild = %d\n", snode.idx+1, nchild);
      spldlt::starpu::insert_assemble(
            node.get_hdl(), cnode_hdls, nchild,
            n, &node, child_contrib, &pool_alloc);

      delete[] cnode_hdls;
      
#else

      assemble_notask(n, node, child_contrib, pool_alloc);

#endif      
   }

} // end of namespace spldlt
