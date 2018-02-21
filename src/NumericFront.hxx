/// \file
/// \copyright 2016- The Science and Technology Facilities Council (STFC)
/// \author    Jonathan Hogg
/// \author    Florent Lopez

#pragma once

#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/SymbolicNode.hxx"

namespace spldlt {

   template<typename T, typename PoolAllocator>
   class Block {
      typedef std::allocator_traits<PoolAllocator> PATraits;
   public:

      /// \brief Constuctor.
      // Block()
      //    : m_(0), n_(0), lda_(0), a_(nullptr)
      // {}

      /// \brief Constuctor.
      /// \param i Block's row index.
      /// \param j Block's column index.
      /// \param m Number of rows in matrix.
      /// \param n Number of columns in matrix.
      /// \param a Pointer to matrix data.
      /// \param lda leading dimension associated with data.
      // Block(int i, int j, int m, int n, T* a, int lda)
      //    : i_(i), j_(j), m_(m), n_(n), lda_(lda), a_(a)
      // {}

      /// \brief Constuctor.
      /// \param i Block's row index.
      /// \param j Block's column index.
      /// \param m Number of rows in matrix.
      /// \param n Number of columns in matrix.
      Block(int i, int j, int m, int n, int lda, 
            PoolAllocator const& pool_alloc)
         : i(i), j(j), m(m), n(n), lda(lda), a(nullptr),
           pool_alloc_(pool_alloc)
      {}
      
      /// \brief Descructor.
      ~Block() {
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

#if defined(SPLDLT_USE_STARPU)
      // Register handle on block
      void register_handle() {
         // Register block in StarPU
         starpu_matrix_data_register(
               &hdl, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(a),
               lda, m, n, sizeof(T));
      }

      // Unregister handle on block asynchronously
      void unregister_handle_submit() {
         starpu_data_unregister_submit(hdl);
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

   // class spral::ssids::cpu::SymbolicNode;

   template <typename T, typename PoolAllocator>
   class NumericFront {
      typedef std::allocator_traits<PoolAllocator> PATraits;
   public:
      /**
       * \brief Constructor
       * \param symb Associated symbolic node.
       * \param pool_alloc Pool Allocator to use for contrib allocation.
       */
      NumericFront(
            // spral::ssids::cpu::SymbolicNode const& symb, 
            SymbolicFront &symb,
            PoolAllocator const& pool_alloc, int blksz)
         : symb(symb), contrib(nullptr), pool_alloc_(pool_alloc), blksz(blksz)
      {

         // Note: it is safer to initialize the contrib_blocks array
         // in alloc_contrib as we don't know the number of input
         // delays when instanciating the front

         // int m = symb.nrow;
         // int n = symb.ncol;            
         // size_t contrib_dimn = m-n;
         // if (contrib_dimn>0) {
         //    int nr = (m-1) / blksz + 1; // number of block rows in front amtrix
         //    int rsa = n / blksz; // index of first block in contribution blocks  
         //    int ncontrib = nr-rsa;
         //    contrib_blocks.reserve(ncontrib*ncontrib);
         //    for(int j = rsa; j < nr; j++) {
         //       // First col in contrib block
         //       int first_col = std::max(j*blksz, n);
         //       // Block width
         //       int blkn = std::min((j+1)*blksz, m) - first_col;
         //       for(int i = rsa; i < nr; i++) {
         //          // First col in contrib block
         //          int first_row = std::max(i*blksz, n);
         //          // Block height
         //          int blkm = std::min((i+1)*blksz, m) - first_row;
         //          contrib_blocks.emplace_back(i-rsa, j-rsa, blkm, blkn, blkm, pool_alloc_);
         //       }
         //    }  
         // }
      }
      /**
       * \brief Destructor
       */
      ~NumericFront() {
         free_contrib();
      }

      /// \brief Allocate block structures and memory space for
      /// contrib (below diagonal)
      void alloc_contrib_blocks() {

         int m = symb.nrow + ndelay_in;
         int n = symb.ncol + ndelay_in;
         size_t contrib_dimn = m-n; // Dimension of contribution block
         if (contrib_dimn>0) {
            int nr = (m-1) / blksz + 1; // number of block rows in front amtrix
            int rsa = n / blksz; // index of first block in contribution blocks  
            int ncontrib = nr-rsa;

            contrib_blocks.reserve(ncontrib*ncontrib);
            for(int j = rsa; j < nr; j++) {
               // First col in contrib block
               int first_col = std::max(j*blksz, n);
               // Block width
               int blkn = std::min((j+1)*blksz, m) - first_col;
               for(int i = rsa; i < nr; i++) {
                  // First col in contrib block
                  int first_row = std::max(i*blksz, n);
                  // Block height
                  int blkm = std::min((i+1)*blksz, m) - first_row;
                  contrib_blocks.emplace_back(i-rsa, j-rsa, blkm, blkn, blkm, pool_alloc_);
               }
            }
   
            for(int j = 0; j < ncontrib; j++) {
               for(int i = j; i < ncontrib; i++) {
                  // printf("[alloc_contrib_blocks] i: %d, j: %d\n", i, j);
                  contrib_blocks[j*ncontrib+i].alloc();
               }
            }
         }
      }

      void free_contrib_blocks() {
         int m = symb.nrow + ndelay_in;
         int n = symb.ncol + ndelay_in;            
         size_t contrib_dimn = m-n; // Dimension of contribution block
         if (contrib_dimn>0) {
            int nr = (m-1) / blksz + 1; // number of block rows in front amtrix
            int rsa = n / blksz; // index of first block in contribution blocks  
            int ncontrib = nr-rsa;
            for(int j = 0; j < ncontrib; j++) {
               for(int i = j; i < ncontrib; i++) {
                  contrib_blocks[j*ncontrib+i].free();
               }
            }
         }
      }

      /**
       * \brief Allocate space for contribution block.
       *
       * Note done at construction time, as a major memory commitment that is
       * transitory.
       */
      void alloc_contrib() {
         size_t contrib_dimn = symb.nrow - symb.ncol;
         contrib_dimn = contrib_dimn*contrib_dimn;
         contrib = (contrib_dimn>0) ? PATraits::allocate(pool_alloc_, contrib_dimn)
            : nullptr;
      }

      /** \brief Free space for contribution block (if allocated) */
      void free_contrib() {
         if(!contrib) return;
         size_t contrib_dimn = symb.nrow - symb.ncol;
         contrib_dimn = contrib_dimn*contrib_dimn;
         PATraits::deallocate(pool_alloc_, contrib, contrib_dimn);
         contrib = nullptr;
      }

      int get_nrow() {
         return symb.nrow + ndelay_in;
      }

      int get_ncol() {
         return symb.ncol + ndelay_in;
      }

      /** \brief Return leading dimension of node's lcol member. */
      size_t get_ldl() {
         return spral::ssids::cpu::align_lda<T>(symb.nrow + ndelay_in);
      }

   public:
      /* Symbolic node associate with this one */
      // spral::ssids::cpu::SymbolicNode const& symb;
      SymbolicFront &symb;

      /* Fixed data from analyse */
      NumericFront<T, PoolAllocator>* first_child; // Pointer to our first child
      NumericFront<T, PoolAllocator>* next_child; // Pointer to parent's next child

      /* Data that changes during factorize */
      int ndelay_in; // Number of delays arising from children
      int ndelay_out; // Number of delays arising to push into parent
      int nelim; // Number of columns succesfully eliminated
      T *lcol; // Pointer to start of factor data
      int *perm; // Pointer to permutation
      T *contrib; // Pointer to contribution block
      int blksz; // Blocking size
      std::vector<spldlt::Block<T, PoolAllocator>> contrib_blocks; // Block structures containing contrib
   private:
      PoolAllocator pool_alloc_; // Our own version of pool allocator for freeing
      // contrib
   };

} /* namespaces spldlt */
