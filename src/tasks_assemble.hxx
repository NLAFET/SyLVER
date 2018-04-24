#pragma once

#include "assemble.hxx"

#if defined(SPLDLT_USE_STARPU)
#include "StarPU/assemble.hxx"
#endif

namespace spldlt {

   ////////////////////////////////////////////////////////////////////////////////
   // assemble contribution block
   
   template <typename T, typename PoolAlloc>
   void assemble_contrib_task(
         NumericFront<T, PoolAlloc>& node,
         void** child_contrib
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
            &node, child_contrib);

      delete[] cnode_hdls;

#else

      assemble_contrib(node, child_contrib);

#endif
   }

} // end of namespace spldlt
