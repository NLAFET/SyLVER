/** \file
 *  \copyright 2016- The Science and Technology Facilities Council (STFC)
 *  \author    Florent Lopez
 */

#pragma once

namespace sylver {
   namespace splu {
      
      /// @brief Allocate data structure associated with a node
      ///
      /// @param factor_alloc Memory allocator for allocating
      /// fully-summed entries
      template <typename T, typename PoolAlloc, typename FactorAlloc>
      void activate_front_unsym(
            NumericFront<T, PoolAlloc> &front,
            FactorAlloc& factor_alloc
            ) {

         /* Rebind allocators */
         typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<T> FADoubleTraits;
         typename FADoubleTraits::allocator_type factor_alloc(factor_alloc);
         typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<int> FAIntTraits;
         typename FAIntTraits::allocator_type factor_alloc_int(factor_alloc);

         front.ndelay_in = 0; // Init incoming delays 

         int const nrow = front.get_nrow();
         int const numfs = front.get_ncol(); // Number of fully-summed rows/columns
         size_t const ldl = get_ldl(); // L factor 
         size_t const ldu = get_ldu(); // U factor
         
         size_t lenl = ldl*ncol;
         size_t lenu = ldu*(nrow-numfs);
         
         
      }

   }
}
