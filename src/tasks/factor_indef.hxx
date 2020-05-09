/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "sylver_ciface.hxx"
#include "factor_indef.hxx"
#if defined(SPLDLT_USE_STARPU)
#include "StarPU/factor_indef.hxx"
#endif

namespace sylver {
namespace spldlt {

   ////////////////////////////////////////////////////////////
   // factor_front_indef_nocontrib_task

   template <typename NumericFrontType, typename PoolAlloc>
   inline void factor_front_indef_task(
         NumericFrontType& node,
         std::vector<spral::ssids::cpu::Workspace>& workspaces,
         PoolAlloc& pool_alloc,
         sylver::options_t & options,
         std::vector<sylver::inform_t>& worker_stats
         ) {

#if defined(SPLDLT_USE_STARPU)

      spldlt::starpu::insert_factor_front_indef(
            node.hdl(), &node, &workspaces, &pool_alloc, &options, 
            &worker_stats);

#else
      
      factor_front_indef(
            node, workspaces, pool_alloc, options, worker_stats);

#endif
   }

   ////////////////////////////////////////////////////////////
   // factor_front_indef_nocontrib_task

   template <typename NumericFrontType, typename PoolAlloc>
   void factor_front_indef_nocontrib_task(
         NumericFrontType& node,
         std::vector<spral::ssids::cpu::Workspace> &workspaces,
         PoolAlloc& pool_alloc,
         sylver::options_t& options
         ) {

#if defined(SPLDLT_USE_STARPU)

      insert_factor_front_indef_nocontrib(
            node.hdl(), &node, &workspaces, &pool_alloc, &options);
      
#else

      factor_front_indef_nocontrib(
            node.symb, node, workspaces, pool_alloc, options);

#endif
   }

   
   ////////////////////////////////////////////////////////////
   // form_contrib_front_task

   template <typename NumericFrontType>
   void form_contrib_front_task(
         NumericFrontType& node,
         int blksz) {

#if defined(SPLDLT_USE_STARPU)

      insert_form_contrib_front(
            node.hdl(), &node, blksz);
      
#else

      form_contrib_front(
            node.symb, node, blksz);

#endif

   }
   
}} // End of namespace sylver::spldlt
