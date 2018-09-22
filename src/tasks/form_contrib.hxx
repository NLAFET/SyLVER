#pragma once

#include <assert.h>

// SSIDS
#include "ssids/cpu/Workspace.hxx"

namespace spldlt {

   template <typename T, typename PoolAlloc>
   void form_contrib_task(
         NumericFront<T, PoolAlloc>& node,
         spral::ssids::cpu::Workspace& work,
         int nelim_from, // First column in factors
         int nelim_to // Last column in factors
         ) {

#if defined(SPLDLT_USE_STARPU)

      starpu_data_handle_t *hdls = nullptr;
      int nh = 0;
      int n = node.get_ncol();
      int m = node.get_nrow();

      if ((m-n) > 0) {         
         // In case there is a contribution block (non-root nodes)
         
         int rsa = n / blksz; // Index of first block in contribution blocks
         int nr = node.get_nr(); // Number of block rows
         int ncb = nr-rsa;
         hdls = new starpu_data_handle_t[ncb*ncb];
      
         for (int j=rsa; j<nr; ++j) {
            for (int i=j; i<nr; ++i) {
               hdls[nh] = node.get_contrib_block(i, j).hdl;
               ++nh;
            }
         }
      }

      spldlt::starpu::insert_form_contrib(
            hdls, nh,
            &node, &workspaces, nelim_from, nelim_to);

      delete[] hdls;
      
#else

      form_contrib_notask(node, work, nelim_from, nelim_to);

#endif
   }

   
   
} // end of namespace spldlt
