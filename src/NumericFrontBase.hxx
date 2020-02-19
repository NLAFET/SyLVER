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
         : symb_(symb), pool_alloc_(pool_alloc), blksz_(blksz),
           contrib_hdl_(nullptr), ndelay_in_(0), ndelay_out_(0)
      {};

      // Return block size
      int blksz() const {return blksz_;}

#if defined(SPLDLT_USE_STARPU)
      /// @brief Return StarPU symbolic handle on front's contribution
      /// blocks
      starpu_data_handle_t contrib_hdl() const {
         return contrib_hdl_; 
      }
      /// @brief Return StarPU symbolic handle on front
      starpu_data_handle_t hdl() const {
         return this->symb().hdl;
      }
#endif

      // Return the number of incoming delays
      int ndelay_in() const { return ndelay_in_; }

      // Set the number of incoming delays to `ndelay`
      void ndelay_in(int ndelay) { ndelay_in_ = ndelay; }

      // Add `ndelay` to the number of incoming delays
      void ndelay_in_add(int ndelay) { ndelay_in_ += ndelay; }

      // Return the number of delays arising to push into parent
      int ndelay_out() const { return ndelay_out_; }

      // Set the number of delays to push into parent to `ndelay`
      void ndelay_out(int ndelay) { ndelay_out_ = ndelay; }

      /// @brief Return the number of columns in the node
      inline int ncol() const {
         return this->symb().ncol + this->ndelay_in_;
      }

      /// @brief Return the number of rows in the node
      inline int nrow() const {
         return this->symb().nrow + this->ndelay_in_;
      }

      /// @brief Return the number of block rows in the node
      inline int nc() const {
         return (this->ncol()-1) / this->blksz() + 1;
      }

      /// @brief Return the number of block rows in the node
      inline int nr() const {
         return (this->nrow()-1) / this->blksz() + 1;
      }

      // Return reference to pool allocator
      PoolAllocator& pool_alloc() { return pool_alloc_; }

#if defined(SPLDLT_USE_STARPU)
      // Register symbolic handle in StarPU
      void register_symb() {
         starpu_void_data_register(&symb_.hdl);
      }

      // Unregister symbolic handle in StarPU when handle is no longer
      // needed
      void unregister_submit_symb() {
         starpu_data_unregister_submit(symb_.hdl);
      }

      // Register symbolic contrib block handle in StarPU
      void register_symb_contrib() {
         
         assert(contrib_hdl_ == nullptr);

         starpu_void_data_register(&contrib_hdl_);
      }

      void unregister_submit_symb_contrib() {

         assert(contrib_hdl_ != nullptr);

         starpu_data_unregister_submit(contrib_hdl_);
      }
#endif

      // Return associated symbolic node
      SymbolicFront& symb() const { return symb_;};
      
   protected:
      int blksz_; // Tileing size
#if defined(SPLDLT_USE_STARPU)
      // Symbolic handle for contribution blocks
      starpu_data_handle_t contrib_hdl_;
#endif
      // Number of delays arising from children
      int ndelay_in_;
      // Number of delays arising to push into parent
      int ndelay_out_;
      // Memory allocator used to manage contribution blocks
      PoolAllocator pool_alloc_;
      // Symbolic frontal matrix
      SymbolicFront& symb_;

   };
}
