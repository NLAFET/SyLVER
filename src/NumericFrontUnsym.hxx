/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author    Florent Lopez
#pragma once

// SyLVER
#include "BlockUnsym.hxx"
#include "sylver/NumericFrontBase.hxx"

namespace sylver {
namespace splu {

template<
   typename T,
   typename FactorAllocator,
   typename PoolAllocator
   >
class NumericFrontUnsym : public sylver::NumericFrontBase<T, FactorAllocator, PoolAllocator> {
public:
   typedef typename std::allocator_traits<PoolAllocator>::template rebind_alloc<int> IntAlloc;
private:
   typedef std::allocator_traits<PoolAllocator> PATraits;

   using NumericFrontType = NumericFrontUnsym<T, FactorAllocator, PoolAllocator>;
public:
   /**
    * \brief Constructor
    * \param symb Associated symbolic node.
    * \param pool_alloc Pool Allocator to use for contrib allocation.
    */
   NumericFrontUnsym(
         sylver::SymbolicFront& symb,
         FactorAllocator const& factor_alloc,
         PoolAllocator const& pool_alloc, int blksz)
      :
      sylver::NumericFrontBase<T, FactorAllocator, PoolAllocator>(symb, factor_alloc, pool_alloc, blksz),
      // cdata(nullptr),
      lcol(nullptr), ucol(nullptr),
      blocks_unsym_(nullptr)
   {}
      
   /**
    * \brief Destructor
    */
   ~NumericFrontUnsym() {
      // free_contrib();
      // free_contrib_blocks();
      // free_backup();
      // free_cdata();
      free_blocks();
   }

   /// @brief Allocate memory for contribution blocks in the
   /// unsymmetric case
   // Use 1D memory layout by default
   void alloc_contrib_blocks() {

      int m = this->nrow();
      int n = this->ncol();

      size_t contrib_dimn = m-n; // Dimension of contribution block
      if (contrib_dimn>0) {

         int nr = this->nr();

         int rsa = n / this->blksz(); // index of first block in contribution blocks  
         int ncontrib = nr-rsa;

         this->contrib_blocks.reserve(ncontrib*ncontrib);

         // Setup data structures for contribution block
         for(int j = rsa; j < nr; j++) {
            int first_col = std::max(j*this->blksz(), n); // First col in contrib block
            int blkn = std::min((j+1)*this->blksz(), m) - first_col; // Tile width
            for(int i = rsa; i < nr; i++) {
               int first_row = std::max(i*this->blksz(), n); // First col in contrib block
               int blkm = std::min((i+1)*this->blksz(), m) - first_row; // Tile height
               this->contrib_blocks.emplace_back(i-rsa, j-rsa, blkm, blkn, blkm, this->pool_alloc());
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
                  
               // this->contrib_blocks[(j-rsa)*ncontrib+(i-rsa)].a = &col[row];
               // this->contrib_blocks[(j-rsa)*ncontrib+(i-rsa)].lda = col_ld; 
               this->contrib_block(i, j).a = &col[row];
               this->contrib_block(i, j).lda = col_ld; 
            }                  
         }
            
      }         
   }

   /// @brief Cleanup memory associated with contribution blocks in
   /// non symmetric case
   void free_contrib_blocks() {

      // printf("[free_contrib_blocks_unsym]\n");

      int m = this->nrow();
      int n = this->ncol();           
      int rsa = n / this->blksz(); // index of first block in contribution blocks  
      size_t contrib_dimn = m-n; // Dimension of contribution block

      if (contrib_dimn>0 && this->contrib_blocks.size()>0) {

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
            T *col = this->contrib_blocks[j*ncontrib].a;
            assert(col != nullptr);
            PATraits::deallocate(this->pool_alloc(), col, col_dimn); // Release memory for this block-column

            for(int i = 0; i < ncontrib; i++) {
               // Nullify pointer on block
               this->contrib_blocks[j*ncontrib+i].a = nullptr;
            }
         }
          
         // Destroy contribution blocks data structures
         this->contrib_blocks.clear();
      }
   }

   void alloc_blocks() {
         
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
   void free_blocks() {

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
   void alloc_backup_blocks() {

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

   void release_backup_blocks() {

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

   // TODO Use generic type for sym and unsym blocks
   inline sylver::splu::BlockUnsym<T>& get_block_unsym(int i, int j) {

      int nr = this->nr();
         
      assert(i < nr);
      assert(j < this->nc());

      return blocks_unsym_[i+nr*j];         
   }

public:
   /* Fixed data from analyse */
   NumericFrontType* first_child; // Pointer to our first child
   NumericFrontType* next_child; // Pointer to parent's next child

   T *lcol; // L factor
   T *ucol; // U factor
      
   int *perm; // Pointer to permutation
   int *cperm; // Pointer to permutation (column)
private:
   sylver::splu::BlockUnsym<T> *blocks_unsym_; // Block array (unsym case) 

};

}} // End of namespace sylver::splu
