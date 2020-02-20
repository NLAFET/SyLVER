/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// Sylver
#include "NumericFront.hxx"
#include "BlockUnsym.hxx"

// STD
#include <iostream>

namespace sylver {
   namespace splu {      
      
      /// @brief Allocate data structure associated with a node
      ///
      /// @param factor_alloc Memory allocator for allocating
      /// fully-summed entries
      template <typename T, typename PoolAlloc, typename FactorAlloc>
      void alloc_front_unsym_diagdom(
            spldlt::NumericFront<T, PoolAlloc> &front,
            FactorAlloc& factor_alloc
            ) {

         /* Rebind allocators */
         typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<T> FADoubleTraits;
         typename FADoubleTraits::allocator_type factor_alloc_double(factor_alloc);
         typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<int> FAIntTraits;
         typename FAIntTraits::allocator_type factor_alloc_int(factor_alloc);

         front.ndelay_in(0); // Init incoming delays 

         int const nrow = front.nrow();
         int const numfs = front.ncol(); // Number of fully-summed rows/columns
         size_t const ldl = front.ldl(); // L factor 
         size_t const ldu = front.get_ldu(); // U factor
         
         std::cout << "[alloc_front_unsym_diagdom] ";
         std::cout << " nrow = " << nrow;
         std::cout << " numfs = " << numfs;
         std::cout << std::endl;

         if (nrow <= 0) return; // Front is empty i.e. only symbolic 

         // Allocate contribution block
         front.alloc_contrib_blocks_unsym();

         size_t lenl = ldl*numfs;
         size_t lenu = ldu*(nrow-numfs);
         // Allocate L factor
         front.lcol = FADoubleTraits::allocate(factor_alloc_double, lenl);
         // Allocate U factor
         if (lenu > 0)
            front.ucol = FADoubleTraits::allocate(factor_alloc_double, lenu);
         
         front.alloc_blocks_unsym();
      }

#if defined(SPLDLT_USE_STARPU)
      
   namespace starpu {

      template <typename T, typename PoolAlloc>
      void register_front_unsym(
            spldlt::NumericFront<T, PoolAlloc> &front) {
         
         int const mblk = front.nr();
         int const blksz = front.blksz();
         int const n = front.ncol();

         for(int jblk=0; jblk<mblk; jblk++) {
            
            int first_col = jblk*blksz; // Global index of first column in block

            for(int iblk=0; iblk<mblk; iblk++) {
               
               int first_row = iblk*blksz; // First col in current block
               
               // Loop if we are in the contributution block
               if ((first_col >= n) && (first_row >= n)) continue;
               
               BlockUnsym<T>& blk = front.get_block_unsym(iblk, jblk);
               blk.register_handle();
            }
         }

      }
      
   } // End of namespace starpu

#endif
      /// @param diagdom Must be set to true if matrix is diagonally
      /// dominant and false otherwise. 
      template <typename T, typename PoolAlloc, typename FactorAlloc>
      void activate_front_unsym(
            spldlt::NumericFront<T, PoolAlloc> &front,
            FactorAlloc& factor_alloc,
            bool diagdom
            ) {
         
         if (diagdom) alloc_front_unsym_diagdom(front, factor_alloc);

#if defined(SPLDLT_USE_STARPU)
         starpu::register_front_unsym(front);
#endif

      }

   }
}
