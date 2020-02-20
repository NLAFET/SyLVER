/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author    Jonathan Hogg
/// @author    Florent Lopez
#pragma once

// SpLDLT
#include "BlockUnsym.hxx"
#include "NumericFrontBase.hxx"
#include "SymbolicFront.hxx"
#include "kernels/ldlt_app.hxx"
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
         contrib(nullptr), backup(nullptr), cdata(nullptr),
         lcol(nullptr), ucol(nullptr), nelim1(0), nelim(0), blocks_unsym_(nullptr)
      {}
      
      /**
       * \brief Destructor
       */
      ~NumericFront() {
         free_contrib();
         free_contrib_blocks();
         free_backup();
         free_cdata();
         free_blocks_unsym();
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

            contrib_blocks.reserve(ncontrib*ncontrib);
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
                  contrib_blocks.emplace_back(i-rsa, j-rsa, blkm, blkn, blkm, this->pool_alloc());
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
               T *col = (col_dimn>0) ? 
                  PATraits::allocate(this->pool_alloc(), col_dimn) : nullptr;  

               for(int i = j; i < nr; i++) {

                  int row = std::max(i*this->blksz(), n) - first_col; 
                  
                  contrib_blocks[(j-rsa)*ncontrib+(i-rsa)].a = &col[row];
                  contrib_blocks[(j-rsa)*ncontrib+(i-rsa)].lda = col_ld; 
                  // contrib_blocks[j*ncontrib+i].a = &col[()]
               }                  
            }

#else
   
            for(int j = 0; j < ncontrib; j++) {
               for(int i = j; i < ncontrib; i++) {
                  // printf("[alloc_contrib_blocks] i: %d, j: %d\n", i, j);
                  contrib_blocks[j*ncontrib+i].alloc();
               }
            }
#endif

         }
      }

      /// @brief Allocate memory for contribution blocks in the
      /// unsymmetric case
      // Use 1D memory layout by default
      void alloc_contrib_blocks_unsym() {

         int m = this->nrow();
         int n = this->ncol();

         size_t contrib_dimn = m-n; // Dimension of contribution block
         if (contrib_dimn>0) {

            int nr = this->nr();

            int rsa = n / this->blksz(); // index of first block in contribution blocks  
            int ncontrib = nr-rsa;

            contrib_blocks.reserve(ncontrib*ncontrib);

            // Setup data structures for contribution block
            for(int j = rsa; j < nr; j++) {
               int first_col = std::max(j*this->blksz(), n); // First col in contrib block
               int blkn = std::min((j+1)*this->blksz(), m) - first_col; // Tile width
               for(int i = rsa; i < nr; i++) {
                  int first_row = std::max(i*this->blksz(), n); // First col in contrib block
                  int blkm = std::min((i+1)*this->blksz(), m) - first_row; // Tile height
                  contrib_blocks.emplace_back(i-rsa, j-rsa, blkm, blkn, blkm, this->pool_alloc());
               }
            }
            
            // Allocate memory using 1D memory layout
            for(int j = rsa; j < nr; j++) {
               // First col in contrib block
               int first_col = std::max(j*this->blksz(), n);
               int blk_n = std::min((j+1)*this->blksz(), m) - first_col; // Tile width
               assert(blk_n > 0);
               int col_m = m-n; // Column height
               int col_ld = col_m;
               size_t col_dimn = col_ld*blk_n;
               T *col = (col_dimn>0) ? 
                  PATraits::allocate(this->pool_alloc(), col_dimn) : nullptr;
#if defined(SPLDLT_USE_STARPU)
#if defined(SPLDLT_USE_GPU)
      starpu_memory_pin(col, col_dimn*sizeof(T));
#endif
#endif
               for(int i = rsa; i < nr; i++) {

                  int row = std::max(i*this->blksz(), n) - n; // First row in block
                  
                  contrib_blocks[(j-rsa)*ncontrib+(i-rsa)].a = &col[row];
                  contrib_blocks[(j-rsa)*ncontrib+(i-rsa)].lda = col_ld; 
               }                  
            }
            
         }         
      }

      /// @brief Cleanup memory associated with contribution blocks
      void free_contrib_blocks() {
         int m = this->nrow();
         int n = this->ncol();           
         int rsa = n / this->blksz(); // index of first block in contribution blocks  
         size_t contrib_dimn = m-n; // Dimension of contribution block
         
         if (contrib_dimn>0 && contrib_blocks.size()>0) {
         
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
               T *col = contrib_blocks[j*(ncontrib+1)].a;
               assert(col != nullptr);
               PATraits::deallocate(this->pool_alloc(), col, col_dimn);

               for(int i = j; i < ncontrib; i++) {

                  int row = i*this->blksz() - first_col; 
                  // Nullify pointer on block 
                  contrib_blocks[j*ncontrib+i].a = nullptr;
               }
               
#else
               for(int i = j; i < ncontrib; i++) {
                  contrib_blocks[j*ncontrib+i].free();
               }
#endif
            }

            // TODO: free contrib block structure
            // contrib_blocks.clear();

         }
      }

      /// @brief Cleanup memory associated with contribution blocks in
      /// non symmetric case
      void free_contrib_blocks_unsym() {

         printf("[free_contrib_blocks_unsym]\n");

         int m = this->nrow();
         int n = this->ncol();           
         int rsa = n / this->blksz(); // index of first block in contribution blocks  
         size_t contrib_dimn = m-n; // Dimension of contribution block

         if (contrib_dimn>0 && contrib_blocks.size()>0) {

            int nr = this->nr(); // number of block rows in front amtrix
            int rsa = n / this->blksz(); // index of first block in contribution blocks  
            int ncontrib = nr-rsa;
            int col_m = m-n; // Column height
            int col_ld = col_m;
            for(int j = 0; j < ncontrib; j++) {
               int jj = j + rsa; // Column index in the global matrix
               int first_col = std::max(jj*this->blksz(), n); // Index of first column in block-column
               int blk_n = std::min((jj+1)*this->blksz(), m) - first_col;
               size_t col_dimn = col_ld*blk_n;
               T *col = contrib_blocks[j*ncontrib].a;
               assert(col != nullptr);
               PATraits::deallocate(this->pool_alloc(), col, col_dimn); // Release memory for this block-column

               for(int i = 0; i < ncontrib; i++) {
                  // Nullify pointer on block
                  contrib_blocks[j*ncontrib+i].a = nullptr;
               }
            }
          
            // Destroy contribution blocks data structures
            contrib_blocks.clear();
         }
      }

      /// @brief zero contribution blocks
      /// 
      /// Note: Mainly used for testing purpose as we avoid explictily
      /// zeroing
      void zero_contrib_blocks() {

         int m = this->nrow();
         int n = this->ncol();            
         size_t contrib_dimn = m-n; // Dimension of contribution block
         if (contrib_dimn>0 && contrib_blocks.size()>0) {
            int nr = this->nr(); // number of block rows in front amtrix
            int rsa = n / this->blksz(); // index of first block in contribution blocks  
            int ncontrib = nr-rsa;
            for(int j = 0; j < ncontrib; j++) {
               for(int i = j; i < ncontrib; i++) {
                  contrib_blocks[j*ncontrib+i].zero();
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

      void alloc_blocks_unsym() {
         
         typedef typename std::allocator_traits<PoolAllocator>::template rebind_traits<sylver::splu::BlockUnsym<T>> BlkAllocTraits;
         typename BlkAllocTraits::allocator_type blkAlloc(this->pool_alloc());
         
         int const m = this->nrow(); // Number of fully summed rows and columns
         int const n = this->ncol(); // Number of fully summed rows and columns
         int const mblk = this->nr();
         // FIXME: overestimation but it might be better to use the
         // same array for fully-summed and contrib blocks
         int const num_blocks = mblk*mblk;
         blocks_unsym_ = BlkAllocTraits::allocate(blkAlloc, num_blocks);
         int const ldl = this->ldl();
         int const ldu = get_ldu();

         printf("[alloc_blocks_unsym] blksz = %d\n", this->blksz());

         for(int jblk=0; jblk<mblk; jblk++) {

            int first_col = jblk*this->blksz(); // Global index of first column in block
            int last_col = std::min((jblk+1)*this->blksz(), m)-1;
            int blk_n = last_col-first_col+1; 
            assert(blk_n > 0);
            for(int iblk=0; iblk<mblk; iblk++) {

               int first_row = iblk*this->blksz(); // First col in current block
               int last_row = std::min((iblk+1)*this->blksz(), m)-1; // First col in current block
               int blk_m = last_row-first_row+1;
               assert(blk_m > 0);
               
               // Loop if we are in the contributution block
               if ((first_col >= n) && (first_row >= n)) continue;

               if (last_col < n) {
                  // Block is in lcol
                  BlkAllocTraits::construct(
                        blkAlloc, &blocks_unsym_[iblk+jblk*mblk],
                        iblk, jblk, blk_m, blk_n,
                        &lcol[first_col*ldl+first_row], ldl);
               }
               else if (first_col >= n) {
                  // Block is in ucol
                  BlkAllocTraits::construct(
                        blkAlloc, &blocks_unsym_[iblk+jblk*mblk],
                        iblk, jblk, blk_m, blk_n,
                        &ucol[(first_col-n)*ldl+first_row], ldu);

               }
               else {
                  // Block is split between lcol and ucol
                  // first_col < n and last_col >= n
                  int mb = std::min((iblk+1)*this->blksz(), n) - first_row;
                  int nb = last_col-n+1;
                  BlkAllocTraits::construct(
                        blkAlloc, &blocks_unsym_[iblk+jblk*mblk],
                        iblk, jblk, blk_m, blk_n,
                        &lcol[first_col*ldl+first_row], ldl,
                        mb, nb, &lcol[first_row], ldu);
               }
               
            }
         }

      }

      /// @brief Release meemory associated with block data structures
      void free_blocks_unsym() {

         if (!blocks_unsym_) return;
         
         typedef typename std::allocator_traits<PoolAllocator>::template rebind_traits<sylver::splu::BlockUnsym<T>> BlkAllocTraits;
         typename BlkAllocTraits::allocator_type blkAlloc(this->pool_alloc());

         int const mblk = this->nr();
         int const num_blocks = mblk*mblk;
         BlkAllocTraits::deallocate(blkAlloc, blocks_unsym_, num_blocks);
         blocks_unsym_ = nullptr;
         
      }

      /// @brief Allocate backups in all blocks for the unsymmetric
      /// case
      void alloc_backup_blocks_unsym() {

         int const n = this->ncol();
         int const nr = this->nr(); 
         int const nc = this->nc();

         int en = (n-1)/this->blksz(); // Last block-row/column in factors

         for (int j = 0; j < nc; ++j) {
            for (int i =  0; i < nr; ++i) {
               // Loop if we are in the cb
               if ((i > en) && (j > en)) continue;
               sylver::splu::BlockUnsym<T>& blk = get_block_unsym(i, j);
               blk.alloc_backup(this->pool_alloc());
            }
         }
      }

      void release_backup_blocks_unsym() {

         int const n = this->ncol();
         int const nr = this->nr(); 
         int const nc = this->nc();

         int en = (n-1)/this->blksz(); // Last block-row/column in factors

         for (int j = 0; j < nc; ++j) {
            for (int i =  0; i < nr; ++i) {
               // Loop if we are in the cb
               if ((i > en) && (j > en)) continue;
               sylver::splu::BlockUnsym<T>& blk = get_block_unsym(i, j);
               blk.release_backup(this->pool_alloc());
            }
         }
         
      }

      /** \brief Return leading dimension of node's ucol member. */
      inline size_t get_ldu() const {
         return spral::ssids::cpu::align_lda<T>(this->symb().ncol + this->ndelay_in_);
      }

      /// @Brief Return block (i,j) in the contribution block
      /// @param i row index of block in the frontal matrix
      /// @param j column index of block in the frontal matrix
      inline sylver::Tile<T, PoolAllocator>& get_contrib_block(int i, int j) {

         // No bound checks when accessing the element in the
         // contrib_blocks vector
         assert(this->symb().nrow > this->symb().ncol);
         
         int n = this->ncol();
         int sa = n / this->blksz();
         int nr = this->nr();
         int ncontrib = nr-sa;

         return contrib_blocks[(i-sa)+(j-sa)*ncontrib];
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

      // TODO Use generic type for sym and unsym blocks
      inline sylver::splu::BlockUnsym<T>& get_block_unsym(int i, int j) {

         int nr = this->nr();
         
         assert(i < nr);
         assert(j < this->nc());

         return blocks_unsym_[i+nr*j];
         
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
      int nelim1; // Number of columns succesfully eliminated during first pass
      int nelim; // Number of columns succesfully eliminated
      // Factors
      T *lcol; // Pointer to start of factor data
      T *ucol; // Factor U
      // Permutations
      int *perm; // Pointer to permutation
      int *cperm; // Pointer to permutation (column)
      T *contrib; // Pointer to contribution block
      std::vector<sylver::Tile<T, PoolAllocator>> contrib_blocks; // Tile structures containing contrib
      spldlt::ldlt_app_internal::CopyBackup<T, PoolAllocator> *backup; // Stores backups of matrix blocks
      // Structures for indef factor
      spldlt::ldlt_app_internal::ColumnData<T, IntAlloc> *cdata;
      std::vector<BlockSpec> blocks;
   private:
      sylver::splu::BlockUnsym<T> *blocks_unsym_; // Block array (unsym case) 
      // PoolAllocator pool_alloc_; // Our own version of pool allocator for freeing
      // contrib
   };

} /* namespaces spldlt */
