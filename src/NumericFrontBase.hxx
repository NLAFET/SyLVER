/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author    Florent Lopez
#pragma once

#include "SymbolicFront.hxx"

namespace sylver {
   
   template <typename T, typename PoolAllocator>
   class NumericFrontBase {
   public:

      NumericFrontBase(
            SymbolicFront& symb,
            PoolAllocator const& pool_alloc,
            int blksz)
         : symb_(symb), pool_alloc_(pool_alloc), blksz_(blksz)
      {};

      // Return block size
      int blksz() const {return blksz_;}

#if defined(SPLDLT_USE_STARPU)
      /// @brief Return StarPU symbolic handle on front
      starpu_data_handle_t& hdl() {
         return this->symb().hdl;
      }

      /// @brief Return StarPU symbolic handle on front's contribution
      /// blocks
      starpu_data_handle_t& contrib_hdl() {
         return contrib_hdl_; 
      }
#endif

      // Return associated symbolic node
      SymbolicFront& symb() const { return symb_;};

      // Return reference to pool allocator
      PoolAllocator& pool_alloc() { return pool_alloc_; }
      
   protected:
      int blksz_; // Tileing size
#if defined(SPLDLT_USE_STARPU)
      starpu_data_handle_t contrib_hdl_; // Symbolic handle for contribution blocks
#endif
      SymbolicFront& symb_; // Symbolic fronal matrix
      PoolAllocator pool_alloc_; // Pool allocator (for contrib)

   };
}
