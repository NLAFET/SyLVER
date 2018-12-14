/// \file
/// \copyright 2016- The Science and Technology Facilities Council (STFC)
/// \author    Jonathan Hogg
/// \author    Florent Lopez

#pragma once

// SpLDLT
#include "Block.hxx"
#include "SymbolicFront.hxx"
#include "kernels/ldlt_app.hxx"

#include <cassert>

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"

namespace spldlt {

   // static const int INNER_BLOCK_SIZE = 32;

   // FIXME: Allocator is not used as both factors and contribution
   // blocks are allocated beforehand
   template<typename T, typename PoolAllocator> 
   class Tile {
      typedef std::allocator_traits<PoolAllocator> PATraits;
   public:

      /// \brief Constuctor.
      // Tile()
      //    : m_(0), n_(0), lda_(0), a_(nullptr)
      // {}

      /// \brief Constuctor.
      /// \param i Tile's row index.
      /// \param j Tile's column index.
      /// \param m Number of rows in matrix.
      /// \param n Number of columns in matrix.
      /// \param a Pointer to matrix data.
      /// \param lda leading dimension associated with data.
      // Tile(int i, int j, int m, int n, T* a, int lda)
      //    : i_(i), j_(j), m_(m), n_(n), lda_(lda), a_(a)
      // {}

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
               a[j*lda+i] = 0.0;
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

   // class spral::ssids::cpu::SymbolicNode;

   template <typename T, typename PoolAllocator>
   class NumericFront {
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
            SymbolicFront& symb,
            PoolAllocator const& pool_alloc, int blksz)
         : symb(symb), contrib(nullptr), pool_alloc_(pool_alloc), blksz(blksz),
           backup(nullptr), cdata(nullptr), ndelay_in(0), ndelay_out(0),
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

         int m = get_nrow();
         int n = get_ncol();
         size_t contrib_dimn = m-n; // Dimension of contribution block
         if (contrib_dimn>0) {
            int nr = get_nr();; // number of block rows in front amtrix
            int rsa = n / blksz; // index of first block in contribution blocks  
            int ncontrib = nr-rsa;

            contrib_blocks.reserve(ncontrib*ncontrib);
            for(int j = rsa; j < nr; j++) {
               // First col in contrib block
               int first_col = std::max(j*blksz, n);
               // Tile width
               int blkn = std::min((j+1)*blksz, m) - first_col;
               for(int i = rsa; i < nr; i++) {
                  // First col in contrib block
                  int first_row = std::max(i*blksz, n);
                  // Tile height
                  int blkm = std::min((i+1)*blksz, m) - first_row;
                  contrib_blocks.emplace_back(i-rsa, j-rsa, blkm, blkn, blkm, pool_alloc_);
               }
            }

#if defined(MEMLAYOUT_1D)

            // Here we allocate block-columns and let the diagonal
            // block hold the data.
            
            // printf("[alloc_contrib_blocks]\n");

            for(int j = rsa; j < nr; j++) {
               // First col in contrib block
               int first_col = std::max(j*blksz, n);
               // Tile width
               int blk_n = std::min((j+1)*blksz, m) - first_col;
               // Column height
               int col_m = m - first_col;
               int col_ld = col_m;
               size_t col_dimn = col_ld*blk_n;
               T *col = (col_dimn>0) ? 
                  PATraits::allocate(pool_alloc_, col_dimn) : nullptr;  

               for(int i = j; i < nr; i++) {

                  int row = std::max(i*blksz, n) - first_col; 
                  
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

         int m = get_nrow();
         int n = get_ncol();

         size_t contrib_dimn = m-n; // Dimension of contribution block
         if (contrib_dimn>0) {

            int nr = get_nr();

            int rsa = n / blksz; // index of first block in contribution blocks  
            int ncontrib = nr-rsa;

            contrib_blocks.reserve(ncontrib*ncontrib);

            // Setup data structures for contribution block
            for(int j = rsa; j < nr; j++) {
               int first_col = std::max(j*blksz, n); // First col in contrib block
               int blkn = std::min((j+1)*blksz, m) - first_col; // Tile width
               for(int i = rsa; i < nr; i++) {
                  int first_row = std::max(i*blksz, n); // First col in contrib block
                  int blkm = std::min((i+1)*blksz, m) - first_row; // Tile height
                  contrib_blocks.emplace_back(i-rsa, j-rsa, blkm, blkn, blkm, pool_alloc_);
               }
            }
            
            // Allocate memory using 1D memory layout
            for(int j = rsa; j < nr; j++) {
               // First col in contrib block
               int first_col = std::max(j*blksz, n);
               int blk_n = std::min((j+1)*blksz, m) - first_col; // Tile width
               assert(blk_n > 0);
               int col_m = m-n; // Column height
               int col_ld = col_m;
               size_t col_dimn = col_ld*blk_n;
               T *col = (col_dimn>0) ? 
                  PATraits::allocate(pool_alloc_, col_dimn) : nullptr;

               for(int i = rsa; i < nr; i++) {

                  int row = std::max(i*blksz, n) - n; // First row in block
                  
                  contrib_blocks[(j-rsa)*ncontrib+(i-rsa)].a = &col[row];
                  contrib_blocks[(j-rsa)*ncontrib+(i-rsa)].lda = col_ld; 
               }                  
            }
            
         }         
      }

      /// @brief Cleanup memory associated with contribution blocks
      void free_contrib_blocks() {
         int m = get_nrow();
         int n = get_ncol();           
         int rsa = n / blksz; // index of first block in contribution blocks  
         size_t contrib_dimn = m-n; // Dimension of contribution block
         
         if (contrib_dimn>0 && contrib_blocks.size()>0) {
         
            int nr = get_nr(); // number of block rows in front amtrix
            int rsa = n / blksz; // index of first block in contribution blocks  
            int ncontrib = nr-rsa;
            for(int j = 0; j < ncontrib; j++) {

#if defined(MEMLAYOUT_1D)
               // Free data for the block column that are begin held
               // by the diagonal block. Then nullify data pointer in
               // each row tile.

               int jj = j + rsa; // Column index in the global matrix
               int first_col = std::max(jj*blksz, n);
               // Column height
               int col_m = m - first_col;
               int col_ld = col_m;
               // Column width
               int blk_n = std::min((jj+1)*blksz, m) - first_col; // TODO check & test blk_n 
               size_t col_dimn = col_ld*blk_n;

               // Diagonal block holding the data for the block-column
               T *col = contrib_blocks[j*(ncontrib+1)].a;
               assert(col != nullptr);
               PATraits::deallocate(pool_alloc_, col, col_dimn);

               for(int i = j; i < ncontrib; i++) {

                  int row = i*blksz - first_col; 
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

         int m = get_nrow();
         int n = get_ncol();           
         int rsa = n / blksz; // index of first block in contribution blocks  
         size_t contrib_dimn = m-n; // Dimension of contribution block

         if (contrib_dimn>0 && contrib_blocks.size()>0) {

            int nr = get_nr(); // number of block rows in front amtrix
            int rsa = n / blksz; // index of first block in contribution blocks  
            int ncontrib = nr-rsa;
            int col_m = m-n; // Column height
            int col_ld = col_m;
            for(int j = 0; j < ncontrib; j++) {
               int jj = j + rsa; // Column index in the global matrix
               int first_col = std::max(jj*blksz, n); // Index of first column in block-column
               int blk_n = std::min((jj+1)*blksz, m) - first_col;
               size_t col_dimn = col_ld*blk_n;
               T *col = contrib_blocks[j*ncontrib].a;
               assert(col != nullptr);
               PATraits::deallocate(pool_alloc_, col, col_dimn); // Release memory for this block-column

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

         int m = get_nrow();
         int n = get_ncol();            
         size_t contrib_dimn = m-n; // Dimension of contribution block
         if (contrib_dimn>0 && contrib_blocks.size()>0) {
            int nr = get_nr(); // number of block rows in front amtrix
            int rsa = n / blksz; // index of first block in contribution blocks  
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

      /// @brief Allocate backups to prepare for the LDL^T
      /// facorization using APTP strategy
      void alloc_backup() {
         backup = 
            new spldlt::ldlt_app_internal::CopyBackup<T, PoolAllocator>(
                  get_nrow(), get_ncol(), blksz, pool_alloc_);
      }
      
      void free_backup() {
         if(!backup) return;
         delete backup;
         backup = nullptr;
      }
      
      void alloc_cdata() {
         cdata = 
            new spldlt::ldlt_app_internal::ColumnData<T, IntAlloc> (
                  get_ncol(), blksz, IntAlloc(pool_alloc_));
      }
      
      void free_cdata() {
         if (!cdata) return;
         delete cdata;
         cdata = nullptr;
      }

      /// @brief Allocate blocks (structre only)
      /// Note: factor data must be allocated
      /// Note: cdata must be allocated and intialized
      void alloc_blocks() {

         int const m = get_nrow();
         int const n = get_ncol();
         int const ldl = get_ldl();
         int const mblk = get_nr();
         int const nblk = get_nc();
         int const num_blocks = nblk*mblk;
         blocks.reserve(num_blocks);

         for(int jblk=0; jblk<nblk; jblk++) {
            for(int iblk=0; iblk<mblk; iblk++) {
               // Create and insert block at the end (column-wise storage)
               blocks.emplace_back(iblk, jblk, m, n, *cdata, &lcol[jblk*blksz*ldl+iblk*blksz], ldl, blksz);
               // alternativel store pointer
               // blocks[jblk*mblk + iblk] = new BlockSpec(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
// #if defined(SPLDLT_USE_STARPU)
//                // register handle for block (iblk, jblk)
//                if (iblk >= jblk) blocks[jblk*mblk+iblk].register_handle(); 
// #endif
            }
         }
      }

      void alloc_blocks_unsym() {
         
         typedef typename std::allocator_traits<PoolAllocator>::template rebind_traits<BlockUnsym<T>> BlkAllocTraits;
         typename BlkAllocTraits::allocator_type blkAlloc(pool_alloc_);
         
         int const m = get_nrow(); // Number of fully summed rows and columns
         int const n = get_ncol(); // Number of fully summed rows and columns
         int const mblk = get_nr();
         // FIXME: overestimation but it might be better to use the
         // same array for fully-summed and contrib blocks
         int const num_blocks = mblk*mblk;
         blocks_unsym_ = BlkAllocTraits::allocate(blkAlloc, num_blocks);
         int const ldl = get_ldl();
         int const ldu = get_ldu();

         printf("[alloc_blocks_unsym] blksz = %d\n", blksz);

         for(int jblk=0; jblk<mblk; jblk++) {

            int first_col = jblk*blksz; // Global index of first column in block
            int last_col = std::min((jblk+1)*blksz, m)-1;
            int blk_n = last_col-first_col+1; 
            assert(blk_n > 0);
            for(int iblk=0; iblk<mblk; iblk++) {

               int first_row = iblk*blksz; // First col in current block
               int last_row = std::min((iblk+1)*blksz, m)-1; // First col in current block
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
                  int mb = std::min((iblk+1)*blksz, n) - first_row;
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
         
         typedef typename std::allocator_traits<PoolAllocator>::template rebind_traits<BlockUnsym<T>> BlkAllocTraits;
         typename BlkAllocTraits::allocator_type blkAlloc(pool_alloc_);

         int const mblk = get_nr();
         int const num_blocks = mblk*mblk;
         BlkAllocTraits::deallocate(blkAlloc, blocks_unsym_, num_blocks);
         blocks_unsym_ = nullptr;
         
      }

      /// @brief Allocate backups in all blocks for the unsymmetric
      /// case
      void alloc_backup_blocks_unsym() {

         int const n = get_ncol();
         int const nr = get_nr(); 
         int const nc = get_nc();

         int en = (n-1)/blksz; // Last block-row/column in factors

         for (int j = 0; j < nc; ++j) {
            for (int i =  0; i < nr; ++i) {
               // Loop if we are in the cb
               if ((i > en) && (j > en)) continue;
               BlockUnsym<T>& blk = get_block_unsym(i, j);
               blk.alloc_backup(pool_alloc_);
            }
         }
      }

      void release_backup_blocks_unsym() {

         int const n = get_ncol();
         int const nr = get_nr(); 
         int const nc = get_nc();

         int en = (n-1)/blksz; // Last block-row/column in factors

         for (int j = 0; j < nc; ++j) {
            for (int i =  0; i < nr; ++i) {
               // Loop if we are in the cb
               if ((i > en) && (j > en)) continue;
               BlockUnsym<T>& blk = get_block_unsym(i, j);
               blk.release_backup(pool_alloc_);
            }
         }
         
      }
      
      /// @brief Return the number of rows in the node
      inline int get_nrow() const {
         return symb.nrow + ndelay_in;
      }

      /// @brief Return the number of columns in the node
      inline int get_ncol() const {
         return symb.ncol + ndelay_in;
      }

      /// @brief Return the number of block rows in the node
      inline int get_nr() const {
         return (get_nrow()-1) / blksz + 1;
      }

      /// @brief Return the number of block rows in the node
      inline int get_nc() const {
         return (get_ncol()-1) / blksz + 1;
      }

      /** \brief Return leading dimension of node's lcol member. */
      inline size_t get_ldl() const {
         return spral::ssids::cpu::align_lda<T>(symb.nrow + ndelay_in);
      }

      /** \brief Return leading dimension of node's ucol member. */
      inline size_t get_ldu() const {
         return spral::ssids::cpu::align_lda<T>(symb.ncol + ndelay_in);
      }

      /// @Brief Return block (i,j) in the contribution block
      /// @param i row index of block in the frontal matrix
      /// @param j column index of block in the frontal matrix
      inline Tile<T, PoolAllocator>& get_contrib_block(int i, int j) {

         // No bound checks when accessing the element in the
         // contrib_blocks vector
         assert(symb.nrow > symb.ncol);
         
         int n = get_ncol();
         int sa = n/blksz;
         int nr = get_nr();
         int ncontrib = nr-sa;

         return contrib_blocks[(i-sa)+(j-sa)*ncontrib];
      }

      /// @Brief Return block (i,j) in the factors
      /// 
      /// @param i row index of block in the frontal matrix 
      /// @param j column index of block in the frontal matrix
      // inline Tile<T, PoolAllocator>& get_block(int i, int j) {

      //    int nr = get_nr();
      //    return blocks[i+j*nr];
      // }

      inline BlockUnsym<T>& get_block_unsym(int i, int j) {

         int nr = get_nr();
         
         assert(i < nr);
         assert(j < get_nc());

         return blocks_unsym_[i+nr*j];
         
      }
      
#if defined(SPLDLT_USE_STARPU)
      /// @brief Return StarPU symbolic handle
      starpu_data_handle_t get_hdl() const {
         return symb.hdl;
      }

      /// @brief Return StarPU symbolic handle on contribution blocks 
      starpu_data_handle_t get_contrib_hdl() const {
         return contrib_hdl; 
      }
#endif
      
   public:
      /* Symbolic node associate with this one */
      SymbolicFront& symb;

      /* Fixed data from analyse */
      NumericFront<T, PoolAllocator>* first_child; // Pointer to our first child
      NumericFront<T, PoolAllocator>* next_child; // Pointer to parent's next child

      int blksz; // Tileing size

      /* Data that changes during factorize */
      int ndelay_in; // Number of delays arising from children
      int ndelay_out; // Number of delays arising to push into parent
      int nelim1; // Number of columns succesfully eliminated during first pass
      int nelim; // Number of columns succesfully eliminated
      // Factors
      T *lcol; // Pointer to start of factor data
      T *ucol; // Factor U
      // Permutations
      int *perm; // Pointer to permutation
      int *cperm; // Pointer to permutation (column)
      T *contrib; // Pointer to contribution block
      std::vector<spldlt::Tile<T, PoolAllocator>> contrib_blocks; // Tile structures containing contrib
      spldlt::ldlt_app_internal::CopyBackup<T, PoolAllocator> *backup; // Stores backups of matrix blocks
      // Structures for indef factor
      spldlt::ldlt_app_internal::ColumnData<T, IntAlloc> *cdata;
      std::vector<BlockSpec> blocks; // FIXME use array instead, using vector isn't really necessary here
#if defined(SPLDLT_USE_STARPU)
      starpu_data_handle_t contrib_hdl; // Symbolic handle for contribution blocks
#endif
   private:
      BlockUnsym<T> *blocks_unsym_; // Block array (unsym case) 
      PoolAllocator pool_alloc_; // Our own version of pool allocator for freeing
      // contrib
   };

} /* namespaces spldlt */
