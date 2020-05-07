/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author    Florent Lopez
#pragma once

namespace sylver {

   // FIXME: Allocator is not used as both factors and contribution
   // blocks are allocated beforehand
   template<typename T, typename PoolAllocator> 
   class Tile {
      typedef std::allocator_traits<PoolAllocator> PATraits;
   public:

      /// \brief Constuctor.
      /// \param i Tile's row index.
      /// \param j Tile's column index.
      /// \param m Number of rows in matrix.
      /// \param n Number of columns in matrix.
      Tile(int i, int j, int m, int n, int lda, 
           PoolAllocator const& pool_alloc)
         : i(i), j(j), m(m), n(n), lda(lda), a(nullptr),
           pool_alloc_(pool_alloc)
      {}
      
      /// \brief Descructor.
      ~Tile() {
         free();
      }

      void alloc() {
         size_t block_sz = lda*n;
         a = (block_sz>0) ? PATraits::allocate(pool_alloc_, block_sz)
            : nullptr;
      }

      void free() {
         if(!a) return;
         size_t block_sz = lda*n;
         PATraits::deallocate(pool_alloc_, a, block_sz);
         a = nullptr;
      }

      /// @brief Zero block
      void zero() {
         if(!a) return;
         for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
               a[j*lda+i] = static_cast<T>(0.0);
            }
         }
      }

#if defined(SPLDLT_USE_STARPU)
      // Register handle on block
      void register_handle() {
         // Make sure pointer in allocated
         if (!a) return;
         // Register block in StarPU
         starpu_matrix_data_register(
               &hdl, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(a),
               lda, m, n, sizeof(T));
      }

      // Unregister handle on block asynchronously
      template<bool async=true>
      void unregister_handle() {
         if(async) starpu_data_unregister_submit(hdl);
         else      starpu_data_unregister(hdl);
         // starpu_data_unregister_submit(hdl);      
      }
#endif

      int i; ///< block's row
      int j; ///< block's column
      int m; ///< number of rows in matrix
      int n; ///< number of columns in matrix
      int lda; ///< leading dimension of underlying storage
      T* a; ///< pointer to underlying matrix storage  
#if defined(SPLDLT_USE_STARPU)
      starpu_data_handle_t hdl;
#endif
   private:
      PoolAllocator pool_alloc_; // Our own version of pool allocator for freeing
   };

} // End of namespace sylver
