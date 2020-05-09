/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author    Florent Lopez
#pragma once

// SyLVER
#include "sylver/SymbolicFront.hxx"
#include "sylver/Tile.hxx"

#include <iostream>

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"


namespace sylver {
   
   template <typename T, typename FactorAllocator, typename PoolAllocator>
   class NumericFrontBase {
   public:
      using ValueType = T;
      using FactorAlloc = FactorAllocator;
      using PoolAlloc = PoolAllocator;
   public:

      NumericFrontBase(
            SymbolicFront& symb,
            FactorAllocator const& factor_alloc,
            PoolAllocator const& pool_alloc,
            int blksz)
         : symb_(symb), factor_alloc_(factor_alloc), pool_alloc_(pool_alloc),
           blksz_(blksz), contrib_hdl_(nullptr), ndelay_in_(0), ndelay_out_(0),
           nelim_first_pass_(0), nelim_(0)
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

      /// @Brief Return block (i,j) in the contribution block
      /// @param i row index of block in the frontal matrix
      /// @param j column index of block in the frontal matrix
      inline sylver::Tile<T, PoolAllocator>& contrib_block(int i, int j) {

         // No bound checks when accessing the element in the
         // contrib_blocks vector
         assert(this->symb().nrow > this->symb().ncol);
         
         int n = this->ncol();
         int sa = n / this->blksz();
         int nr = this->nr();
         int ncontrib = nr-sa;

         assert((i >= sa) && (j >= sa));
            
         return contrib_blocks[(i-sa)+(j-sa)*ncontrib];
      }

      inline sylver::Tile<T, PoolAllocator> const& contrib_block(int i, int j) const {

         // No bound checks when accessing the element in the
         // contrib_blocks vector
         assert(this->symb().nrow > this->symb().ncol);
         
         int n = this->ncol();
         int sa = n / this->blksz();
         int nr = this->nr();
         int ncontrib = nr-sa;

         assert((i >= sa) && (j >= sa));
            
         return contrib_blocks[(i-sa)+(j-sa)*ncontrib];
      }

      /** \brief Return leading dimension of node's lcol member. */
      inline size_t ldl() const {
         return spral::ssids::cpu::align_lda<T>(this->symb().nrow + this->ndelay_in_);
      }

      // Return the number of incoming delays
      inline int ndelay_in() const { return ndelay_in_; }

      // Set the number of incoming delays to `ndelay`
      void ndelay_in(int ndelay) { ndelay_in_ = ndelay; }

      // Add `ndelay` to the number of incoming delays
      void ndelay_in_add(int ndelay) { ndelay_in_ += ndelay; }

      // Return the number of delays arising to push into parent
      inline int ndelay_out() const { return ndelay_out_; }

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
         return (this->ncol() - 1 + this->blksz()) / this->blksz();
      }

      /// @brief Return the number of block rows in the node
      inline int nr() const {
         return (this->nrow() - 1 + this->blksz()) / this->blksz();
      }

      inline int& nelim() {
         return nelim_;
      }

      inline int nelim() const {
         return nelim_;
      }
      
      void nelim(int ne) { nelim_ = ne; };

      void nelim_add(int ne) { nelim_ += ne; };
      
      inline int nelim_first_pass() const {
         return nelim_first_pass_;
      }

      void nelim_first_pass(int ne) { nelim_first_pass_ = ne; };

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
   public:
      // Blocks in the frontal matrix contributions
      std::vector<sylver::Tile<T, PoolAllocator>> contrib_blocks;
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
      // Number of columns succesfully eliminated during first pass
      int nelim_first_pass_;
      // Number of columns succesfully eliminated
      int nelim_;
      // Memory allocator to manage factors
      FactorAllocator factor_alloc_;
      // Memory allocator used to manage contribution blocks
      PoolAllocator pool_alloc_;
      // Symbolic frontal matrix
      SymbolicFront& symb_;

   };
}
