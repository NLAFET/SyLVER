/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author    Jonathan Hogg
/// @author    Florent Lopez
#pragma once

// SyLVER
#include "kernels/ldlt_app.hxx"
#include "NumericFrontBase.hxx"
#include "SymbolicFront.hxx"
#include "Tile.hxx"

// STD
#include <cassert>

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"

namespace spldlt {

   template <typename T, typename PoolAllocator>
   class NumericFront : public sylver::NumericFrontBase<T, PoolAllocator> {
   public:
      typedef typename std::allocator_traits<PoolAllocator>::template rebind_alloc<int> IntAlloc;
   private:
      typedef spldlt::ldlt_app_internal::Block<T, INNER_BLOCK_SIZE, IntAlloc> BlockSpec;
      typedef std::allocator_traits<PoolAllocator> PATraits;
   public:
      /**
       * \brief Constructor
       * \param symb Associated symbolic node.
       * \param pool_alloc Pool Allocator to use for contrib allocation.
       */
      NumericFront(
            sylver::SymbolicFront& symb,
            PoolAllocator const& pool_alloc, int blksz)
         :
         sylver::NumericFrontBase<T, PoolAllocator>(symb, pool_alloc, blksz),
         contrib(nullptr), backup(nullptr), cdata(nullptr), lcol(nullptr)
      {}
      
      /**
       * \brief Destructor
       */
      ~NumericFront() {
         free_contrib();
         free_contrib_blocks();
         free_backup();
         free_cdata();
         // TODO Free backups in blocks if necessary
      }

      /// \brief Allocate block structures and memory space for
      /// contrib (below diagonal)
      void alloc_contrib_blocks() {

         int m = this->nrow();
         int n = this->ncol();
         size_t contrib_dimn = m-n; // Dimension of contribution block
         if (contrib_dimn>0) {
            int nr = this->nr();; // number of block rows in front amtrix
            int rsa = n / this->blksz(); // index of first block in contribution blocks  
            int ncontrib = nr-rsa;

            this->contrib_blocks.reserve(ncontrib*ncontrib);
            for(int j = rsa; j < nr; j++) {
               // First col in contrib block
               int first_col = std::max(j*this->blksz(), n);
               // Tile width
               int blkn = std::min((j+1)*this->blksz(), m) - first_col;
               for(int i = rsa; i < nr; i++) {
                  // First col in contrib block
                  int first_row = std::max(i*this->blksz(), n);
                  // Tile height
                  int blkm = std::min((i+1)*this->blksz(), m) - first_row;
                  this->contrib_blocks.emplace_back(i-rsa, j-rsa, blkm, blkn, blkm, this->pool_alloc());
               }
            }

#if defined(MEMLAYOUT_1D)

            // Here we allocate block-columns and let the diagonal
            // block hold the data.
            
            // printf("[alloc_contrib_blocks]\n");

            for(int j = rsa; j < nr; j++) {
               // First col in contrib block
               int first_col = std::max(j*this->blksz(), n);
               // Tile width
               int blk_n = std::min((j+1)*this->blksz(), m) - first_col;
               // Column height
               int col_m = m - first_col;
               int col_ld = col_m;
               size_t col_dimn = col_ld*blk_n;
               T *col = nullptr;

               if (col_dimn>0) {
                  // Allocate block column in memory
                  col = PATraits::allocate(this->pool_alloc(), col_dimn);
#if defined(SPLDLT_USE_STARPU)
#if defined(SPLDLT_USE_GPU)
                  int ret = starpu_memory_pin(col, col_dimn*sizeof(T));
                  STARPU_CHECK_RETURN_VALUE(ret, "starpu_memory_pin");
#endif
#endif
               }

               for(int i = j; i < nr; i++) {

                  int row = std::max(i*this->blksz(), n) - first_col; 
                  assert(row >= 0);
                  this->contrib_block(i, j).a = &col[row];
                  this->contrib_block(i, j).lda = col_ld; 

               }                  
            }

#else
   
            for(int j = 0; j < ncontrib; j++) {
               for(int i = j; i < ncontrib; i++) {
                  // printf("[alloc_contrib_blocks] i: %d, j: %d\n", i, j);
                  this->contrib_blocks[j*ncontrib+i].alloc();
               }
            }
#endif

         }
      }

      /// @brief Cleanup memory associated with contribution blocks
      void free_contrib_blocks() {
         int m = this->nrow();
         int n = this->ncol();           
         int rsa = n / this->blksz(); // index of first block in contribution blocks  
         size_t contrib_dimn = m-n; // Dimension of contribution block
         
         if (contrib_dimn>0 && this->contrib_blocks.size()>0) {
         
            int nr = this->nr(); // number of block rows in front amtrix
            int rsa = n / this->blksz(); // index of first block in contribution blocks  
            int ncontrib = nr-rsa;
            for(int j = 0; j < ncontrib; j++) {

#if defined(MEMLAYOUT_1D)
               // Free data for the block column that are begin held
               // by the diagonal block. Then nullify data pointer in
               // each row tile.

               int jj = j + rsa; // Column index in the global matrix
               int first_col = std::max(jj*this->blksz(), n);
               // Column height
               int col_m = m - first_col;
               int col_ld = col_m;
               // Column width
               int blk_n = std::min((jj+1)*this->blksz(), m) - first_col; // TODO check & test blk_n 
               size_t col_dimn = col_ld*blk_n;

               // Diagonal block holding the data for the block-column
               T *col = this->contrib_blocks[j*(ncontrib+1)].a;
               assert(col != nullptr);

#if defined(SPLDLT_USE_STARPU)
#if defined(SPLDLT_USE_GPU)
                  int ret = starpu_memory_unpin(col, col_dimn*sizeof(T));
                  STARPU_CHECK_RETURN_VALUE(ret, "starpu_memory_unpin");
#endif
#endif
               
               PATraits::deallocate(this->pool_alloc(), col, col_dimn);
               
               for(int i = j; i < ncontrib; i++) {

                  int row = i*this->blksz() - first_col; 
                  // Nullify pointer on block 
                  this->contrib_blocks[j*ncontrib+i].a = nullptr;
               }
               
#else
               for(int i = j; i < ncontrib; i++) {
                  this->contrib_blocks[j*ncontrib+i].free();
               }
#endif
            }

            // TODO: free contrib block structure
            // contrib_blocks.clear();

         }
      }

      /// @brief zero contribution blocks
      /// 
      /// Note: Mainly used for testing purpose as we avoid explictily
      /// zeroing
      void zero_contrib_blocks() {

         int const m = this->nrow();
         int const n = this->ncol();            
         size_t const contrib_dimn = m-n; // Dimension of contribution block
         if (contrib_dimn>0 && this->contrib_blocks.size()>0) {
            int const nr = this->nr(); // number of block rows in front amtrix
            int const rsa = n / this->blksz(); // index of first block in contribution blocks  
            int const ncontrib = nr-rsa;
            for(int j = 0; j < ncontrib; j++) {
               for(int i = j; i < ncontrib; i++) {
                  // this->contrib_blocks[j*ncontrib+i].zero();
                  this->contrib_block(i,j).zero();
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
         size_t contrib_dimn = this->symb().nrow - this->symb().ncol;
         contrib_dimn = contrib_dimn*contrib_dimn;
         contrib = (contrib_dimn>0) ? PATraits::allocate(this->pool_alloc(), contrib_dimn)
            : nullptr;
      }

      /** \brief Free space for contribution block (if allocated) */
      void free_contrib() {
         if(!contrib) return;
         size_t contrib_dimn = this->symb().nrow - this->symb().ncol;
         contrib_dimn = contrib_dimn*contrib_dimn;
         PATraits::deallocate(this->pool_alloc(), contrib, contrib_dimn);
         contrib = nullptr;
      }

      /// @brief Allocate backups to prepare for the LDL^T
      /// facorization using APTP strategy
      void alloc_backup() {
         backup = 
            new spldlt::ldlt_app_internal::CopyBackup<T, PoolAllocator>(
                  this->nrow(), this->ncol(), this->blksz(),
                  this->pool_alloc());
      }
      
      /// @brief Release backup
      void free_backup() {
         if(!backup) return;
         delete backup;
         backup = nullptr;
      }
      
      void alloc_cdata() {
         cdata = 
            new spldlt::ldlt_app_internal::ColumnData<T, IntAlloc> (
                  this->ncol(), this->blksz(), IntAlloc(this->pool_alloc()));
      }
      
      void free_cdata() {
         if (!cdata) return;
         delete cdata;
         cdata = nullptr;
      }

      /// @brief Allocate blocks (structre only)
      /// Note: factor data must be allocated
      /// Note: cdata must be allocated and intialized
      int alloc_blocks() {

         assert(lcol != nullptr);
         assert(cdata != nullptr);

         int ret = sylver::Flag::SUCCESS;
         // Check if factor has been allocated
         if (!lcol) return sylver::Flag::ERROR_UNKNOWN;
         // Check if cdata has been allocated
         if (!cdata) return sylver::Flag::ERROR_UNKNOWN;
         
         int const m = this->nrow();
         int const n = this->ncol();
         int const ldl = this->ldl();
         int const mblk = this->nr();
         int const nblk = this->nc();
         int const num_blocks = nblk*mblk;
         blocks.reserve(num_blocks);

         for(int jblk=0; jblk<nblk; jblk++) {
            for(int iblk=0; iblk<mblk; iblk++) {
               // Create and insert block at the end (column-wise storage)
               blocks.emplace_back(iblk, jblk, m, n, *cdata, &lcol[jblk*this->blksz()*ldl+iblk*this->blksz()], ldl, this->blksz());
               // alternativel store pointer
               // blocks[jblk*mblk + iblk] = new BlockSpec(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
// #if defined(SPLDLT_USE_STARPU)
//                // register handle for block (iblk, jblk)
//                if (iblk >= jblk) blocks[jblk*mblk+iblk].register_handle(); 
// #endif
            }
         }

         return ret;
      }

      /// @Brief Return block (i,j) in the factors
      /// 
      /// @param i Block row index in the front 
      /// @param j Block column index in the front
      inline BlockSpec& get_block(int i, int j) {

         int nr = this->nr();

         assert(i < nr);
         assert(j < this->nc());

         return blocks[i+j*nr];
      }

      /// @Brief Return block (i,j) in the factors similarly to
      /// get_block
      /// 
      /// @param i Block row index in the front 
      /// @param j Block column index in the front
      inline BlockSpec& operator()(int i, int j) {
         return get_block(i,j);
      }

#if defined(SPLDLT_USE_STARPU)
      // Return StarPU handle associated with block (i,j)
      starpu_data_handle_t block_hdl(int i, int j) const {
         assert(i >= j);
         starpu_data_handle_t bhdl = this->blocks[j*this->nr()+i].get_hdl();
         assert(bhdl != nullptr);
         return bhdl;
      }
#endif
      
   public:

      /* Fixed data from analyse */
      NumericFront<T, PoolAllocator>* first_child; // Pointer to our first child
      NumericFront<T, PoolAllocator>* next_child; // Pointer to parent's next child

      /* Data that changes during factorize */
      // Factors
      T *lcol; // Pointer to start of factor data
      // Permutations
      int *perm; // Pointer to permutation
      int *cperm; // Pointer to permutation (column)
      T *contrib; // Pointer to contribution block
      spldlt::ldlt_app_internal::CopyBackup<T, PoolAllocator> *backup; // Stores backups of matrix blocks
      // Structures for indef factor
      spldlt::ldlt_app_internal::ColumnData<T, IntAlloc> *cdata;
      std::vector<BlockSpec> blocks;
   };

} /* namespaces spldlt */
