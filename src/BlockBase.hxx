/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// STD
#include <cassert>

#if defined(SPLDLT_USE_STARPU)
#include <starpu.h>
#endif

namespace sylver {

   template<typename T>
   class BlockBase {
   public:

      BlockBase(int i, int j, int m, int n, T* a, int lda)
         : i_(i), j_(j), m_(m), n_(n), a_(a), lda_(lda), hdl_(nullptr)
      {
         // Block indexes must be positive
         assert((i >= 0) && (j >= 0));
         // Block dimensions must be positive
         assert((m >= 0) && (n >= 0));
         // Block leading dimension must be positive
         assert(lda >= 0);         
      }

      BlockBase(int i, int j, int m, int n)
         : i_(i), j_(j), m_(m), n_(n), a_(nullptr), lda_(0), hdl_(nullptr)
      {
         // Block indexes must be positive
         assert((i >= 0) && (j >= 0));
         // Block dimensions must be positive
         assert((m >= 0) && (n >= 0));
      }
         
      T* a() { return a_; }
      T const* a() const { return a_; }

      void a(T const* data) {
         a_ = data;
      }
      
      int i() const { return i_; }
      int j() const { return j_; }

#if defined(SPLDLT_USE_STARPU)
      starpu_data_handle_t hdl() const {
         return hdl_;
      }
#endif
      
      int lda() const { return lda_; }

      void lda(int in_lda) {
         lda_ = in_lda;
      }
      
      int m() const { return m_; }
      int n() const { return n_; }

#if defined(SPLDLT_USE_STARPU)
      // Register handle on block
      void register_handle() {

         assert(a_ != nullptr);
         // Make sure pointer in allocated
         if (!a_) return;
         // Register block in StarPU
         starpu_matrix_data_register(
               &hdl_, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(this->a()),
               this->lda(), this->m(), this->n(), sizeof(T));
      }

      // Unregister handle on block asynchronously
      template<bool async=true>
      void unregister_handle() {

         // Make sure handle has been allocated
         assert(hdl_ != nullptr);
            
         if(async) {
            starpu_data_unregister_submit(hdl_);
         }
         else {
            starpu_data_unregister(hdl_);
         }

         // Nullify handle pointer
         this->hdl_ = nullptr;
      }
#endif

      /// @brief Zero block
      void zero() {
         assert(a_ != nullptr);
         if(!a_) return;
         for (int j = 0; j < this->n(); j++) {
            for (int i = 0; i < this->m(); i++) {
               this->a_[j*this->lda()+i] = static_cast<T>(0.0);
            }
         }
      }

   private:
      T* a_; // pointer to underlying matrix storage
      int i_; // block row index
      int j_; // block column index
#if defined(SPLDLT_USE_STARPU)
      starpu_data_handle_t hdl_;
#endif
      int lda_; // leading dimension of underlying storage
      int m_; // Number of rows in block
      int n_; // Number of columns in block
   };
}
