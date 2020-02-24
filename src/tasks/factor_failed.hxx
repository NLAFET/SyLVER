/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "sylver_ciface.hxx"
#include "StarPU/factor_failed.hxx"

// STD
#include <assert.h>

namespace spldlt {

   ////////////////////////////////////////////////////////////
   // factor_front_indef_failed_task

   template <typename T, typename PoolAlloc>
   void factor_front_indef_failed_task(
         NumericFront<T, PoolAlloc> &node,
         std::vector<spral::ssids::cpu::Workspace> &workspaces,
         sylver::options_t& options,
         std::vector<sylver::inform_t>& worker_stats
         ) {

#if defined(SPLDLT_USE_STARPU)

      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> IntAlloc;
      int blksz = node.blksz();
      spldlt::ldlt_app_internal::ColumnData<T, IntAlloc> &cdata = *node.cdata;
      int const nblk = node.nc(); // Number of block-columns
      starpu_data_handle_t *hdls = nullptr;
      int nh = 0;
      int n = node.ncol();
      int m = node.nrow();

      if ((m-n) > 0) {         
         // In case there is a contribution block (non-root nodes)
         
         int rsa = n / blksz; // Index of first block in contribution blocks
         int nr = node.nr(); // Number of block rows
         int ncb = nr-rsa;
         hdls = new starpu_data_handle_t[ncb*ncb];
      
         for (int j=rsa; j<nr; ++j) {
            for (int i=j; i<nr; ++i) {
               hdls[nh] = node.get_contrib_block(i, j).hdl;
               ++nh;
            }
         }
      }
      
      sylver::spldlt::starpu::insert_factor_front_indef_failed(
            cdata[nblk-1].get_hdl(), node.contrib_hdl(),
            hdls, nh,
            &node, &workspaces, &options, &worker_stats
            );

      delete[] hdls;

#else

      // spral::ssids::cpu::Workspace& work = workspaces[0];
      sylver::inform_t& stats = worker_stats[0];
      factor_front_indef_failed(node, workspaces, options, stats);

#endif
      
   }


} // end of namespace spldlt
