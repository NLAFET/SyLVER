#pragma once

#include "factor_indef.hxx"
#include "StarPU/factor_indef.hxx"

namespace spldlt {

   ////////////////////////////////////////////////////////////////////////////////
   // factor_front_indef_nocontrib_task

   template <typename T, typename PoolAlloc>
   void factor_front_indef_task(
         NumericFront<T, PoolAlloc> &node,
         std::vector<spral::ssids::cpu::Workspace> &workspaces,
         PoolAlloc& pool_alloc,
         struct cpu_factor_options& options
         ) {

#if defined(SPLDLT_USE_STARPU)

      insert_factor_front_indef(
            node.get_hdl(), &node, &workspaces, &pool_alloc, &options);      

#else

      factor_front_indef(
            node.symb, node, workspaces, pool_alloc, options);

#endif
   }

   ////////////////////////////////////////////////////////////////////////////////
   // factor_front_indef_nocontrib_task
   template <typename T, typename PoolAlloc>
   void factor_front_indef_nocontrib_task(
         NumericFront<T, PoolAlloc> &node,
         std::vector<spral::ssids::cpu::Workspace> &workspaces,
         PoolAlloc& pool_alloc,
         struct cpu_factor_options& options
         ) {

#if defined(SPLDLT_USE_STARPU)

      insert_factor_front_indef_nocontrib(
            node.get_hdl(), &node, &workspaces, &pool_alloc, &options);
      
#else

      factor_front_indef_nocontrib(
            node.symb, node, workspaces, pool_alloc, options);

#endif
   }

   
   ////////////////////////////////////////////////////////////////////////////////

   template <typename T, typename PoolAlloc>
   void form_contrib_front_task(
         NumericFront<T, PoolAlloc> &node,
         int blksz) {

#if defined(SPLDLT_USE_STARPU)


      insert_form_contrib_front(
            node.get_hdl(), &node, blksz);
      
#else

      form_contrib_front(
            node.symb, node, blksz);

#endif

   }
   
} // namespace spldlt