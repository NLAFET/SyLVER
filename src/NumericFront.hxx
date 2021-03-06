/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author    Jonathan Hogg
/// @author    Florent Lopez
#pragma once

// SyLVER
#include "kernels/ldlt_app.hxx"
#include "sylver/kernels/contrib.hxx"
#include "sylver/kernels/ColumnData.hxx"
#include "sylver/kernels/CopyBackup.hxx"
#include "sylver/NumericFrontBase.hxx"
#include "sylver/SymbolicFront.hxx"
#include "sylver/Tile.hxx"

// STD
#include <cassert>

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"

namespace sylver {
namespace spldlt {

template<
   typename T,
   typename FactorAllocator,
   typename PoolAllocator
   >
class NumericFront : public sylver::NumericFrontBase<T, FactorAllocator, PoolAllocator> {
public:
   // Integer allocator
   using IntAlloc = typename std::allocator_traits<PoolAllocator>::template rebind_alloc<int>;
   // Backup strategy
   using Backup = sylver::CopyBackup<T, PoolAllocator>;
private:
   // Block type
   using BlockType = spldlt::ldlt_app_internal::Block<T, INNER_BLOCK_SIZE, IntAlloc>;
   // Pool allocator traits
   using PATraits = std::allocator_traits<PoolAllocator>; 
   // Front type
   using NumericFrontType = NumericFront<T, FactorAllocator, PoolAllocator>;
public:
   /**
    * \brief Constructor
    * \param symb Associated symbolic node.
    * \param pool_alloc Pool Allocator to use for contrib allocation.
    */
   NumericFront(
         sylver::SymbolicFront& symb, // Symbolic front
         FactorAllocator const& factor_alloc,
         PoolAllocator const& pool_alloc,
         int blksz // Block size
         )
      :
      sylver::NumericFrontBase<T, FactorAllocator, PoolAllocator>(symb, factor_alloc, pool_alloc, blksz),
      contrib(nullptr), backup(nullptr), cdata(nullptr), lcol(nullptr),
      first_child(nullptr), next_child(nullptr), perm(nullptr), cperm(nullptr)
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
   
   // Initialize front: allocate memory for storing factors
   void allocate(void** child_contrib) {

      this->ndelay_out(0);
      this->ndelay_in(0);

      // Count incoming delays and determine size of node
      for(auto* child=this->first_child; child!=NULL; child=child->next_child) {
         // Make sure we're not in a subtree
         if (child->symb().exec_loc == -1) {
            this->ndelay_in_add(child->ndelay_out());
         } 
         else {
            int cn, ldcontrib, ndelay, lddelay;
            T const *cval, *delay_val;
            int const *crlist, *delay_perm;
            // spral_ssids_contrib_get_data(
            contrib_get_data(
                  child_contrib[child->symb().contrib_idx], &cn, &cval,
                  &ldcontrib, &crlist, &ndelay, &delay_perm, &delay_val,
                  &lddelay);
            this->ndelay_in_add(ndelay);
         }
      }

      // Get space for node now we know it size using factor
      // allocator
      // NB L is  nrow x ncol and D is 2 x ncol
      size_t ldl = spral::ssids::cpu::align_lda<T>(this->nrow());
      size_t len = (ldl+2) * this->ncol(); // indef (includes D)

      using FAValueTypeTraits = typename std::allocator_traits<FactorAllocator>::template rebind_traits<T>;
      using FAValueType = typename FAValueTypeTraits::allocator_type;
      FAValueType factor_alloc_value_type(this->factor_alloc_); 

      // Allocate factor memory
      this->lcol = FAValueTypeTraits::allocate(factor_alloc_value_type, len);
      // Make sure memory was allocated
      assert(this->lcol != nullptr);

      
#if defined(SPLDLT_USE_GPU)
      // Pin memory on host for faster and asynchronous CPU-GPU memory
      // transfer
#if defined(SPLDLT_USE_STARPU)
      int ret = starpu_memory_pin(this->lcol, len*sizeof(T));
      // STARPU_CHECK_RETURN_VALUE(ret, "starpu_memory_pin");
#endif
#endif

      // Get space for contribution block + (explicitly do not zero it!)
      this->alloc_contrib_blocks();

      using FAIntTraits = typename std::allocator_traits<FactorAllocator>::template rebind_traits<int>;
      typename FAIntTraits::allocator_type factor_alloc_int(this->factor_alloc_);

      // Alloc + set perm for expected eliminations at this node
      // (delays are set when they are imported from children)
      this->perm = FAIntTraits::allocate(factor_alloc_int, this->ncol()); // ncol fully summed variables
      // Set permuatation array
      assert(this->symb().rlist);
      for(int i=0; i< this->symb().ncol; i++) {
         this->perm[i] = this->symb().rlist[i];
      }
      
      // TODO: Backup is needed only when pivot_method is set to APTP

      // Allocate backups
      this->alloc_backup();      
      // Allocate cdata
      this->alloc_cdata();
      // Allocate frontal matrix blocks
      this->alloc_blocks();  
   }

   // TODO: Move to new type NumericFrontPosdef
   void allocate_posdef() {

      std::string const context = "NumericFront::allocate_posdef";

      this->ndelay_out(0);
      this->ndelay_in(0);

      using FAValueTypeTraits = typename std::allocator_traits<FactorAllocator>::template rebind_traits<T>;
      using FAValueType = typename FAValueTypeTraits::allocator_type;
      FAValueType factor_alloc_value_type(this->factor_alloc_); 

      /* Get space for node now we know it size using Fortran allocator + zero it*/
      // NB L is  nrow x ncol and D is 2 x ncol (but no D if posdef)
      size_t ldl = spral::ssids::cpu::align_lda<T>(this->nrow());
      size_t len = ldl * this->ncol();  // posdef
      this->lcol = FAValueTypeTraits::allocate(factor_alloc_value_type, len);

#if defined(SPLDLT_USE_GPU)
#if defined(SPLDLT_USE_STARPU)
      int ret = starpu_memory_pin(this->lcol, len*sizeof(T));
      // STARPU_CHECK_RETURN_VALUE(ret, "starpu_memory_pin");
#endif
#endif
      int err;
      // Get space for contribution block + (explicitly do not zero it!)
      this->alloc_contrib_blocks();
      // Allocate cdata (required for allocating blocks)
      // FIXME not needed for posdef case
      this->alloc_cdata();
      // Allocate frontal matrix blocks
      err = this->alloc_blocks(); // FIXME specialize for posdef case
      sylver::sylver_check_error(err, context, "Failed to allocate blocks");
   }

#if defined(SPLDLT_USE_STARPU)   
   // TODO: test routine
   void register_node() {

      using ValueType = T;
         
      auto& sfront = this->symb();
      int blksz = this->blksz();
      int m = this->nrow();
      int n = this->ncol();
      ValueType *a = this->lcol;
      int lda = spral::ssids::cpu::align_lda<ValueType>(m);
      int nr = this->nr(); // number of block rows
      int nc = this->nc(); // number of block columns
      auto& cdata = *this->cdata;

      // Block diagonal matrix 
      ValueType *d = &a[n*lda];

      // FIXME: DO we still need these handles?
      sfront.handles.resize(nr*nc); // allocate handles
      
      for(int j = 0; j < nc; ++j) {

         int blkn = std::min(blksz, n - j*blksz);

         // Register cdata for APP factorization.
         // FIXME: Only if pivot_method is APP
         cdata[j].register_handle(); // Symbolic handle on column j
         cdata[j].register_d_hdl(d, 2*std::min((j+1)*blksz, n)); // Handle on diagonal D 
         // cdata[j].register_d_hdl(d, 2*n); // Handle on diagonal D 

         for(int i = j; i < nr; ++i) {
            int blkm = std::min(blksz, m - i*blksz);

            // TODO remove sfront.handles registration for indef case
            starpu_matrix_data_register(
                  &(sfront.handles[i + j*nr]), // StarPU handle ptr 
                  STARPU_MAIN_RAM, // memory 
                  reinterpret_cast<uintptr_t>(&a[(j*blksz)*lda+(i*blksz)]),
                  lda, blkm, blkn,
                  sizeof(ValueType));

            // Register StarPU handle for block (i,j)
            this->blocks[j*nr+i].register_handle(); 
         }
      }

      // Register blocks in contribution block
      this->register_contrib_blocks();

   }

   // TODO: move to a new structure NumericFrontPosdef 
   // TODO: est routine
   void register_node_posdef() {

      using ValueType = T;

      auto& sfront = this->symb();
      int blksz = this->blksz();

      int const m = this->nrow();
      int const n = this->ncol();
      ValueType *a = this->lcol;
      int const lda = this->ldl();
      int const nr = this->nr(); // number of block rows
      int const nc = this->nc(); // number of block columns
      // sfront.handles.reserve(nr*nc);
      sfront.handles.resize(nr*nc); // Allocate handles

      for(int j = 0; j < nc; ++j) {
         int blkn = std::min(blksz, n - j*blksz);

         for(int i = j; i < nr; ++i) {
            int blkm = std::min(blksz, m - i*blksz);

            // TODO: remove the following register and test
            starpu_matrix_data_register(
                  &(sfront.handles[i + j*nr]), // StarPU handle ptr 
                  STARPU_MAIN_RAM, // memory 
                  reinterpret_cast<uintptr_t>(&a[(j*blksz)*lda+(i*blksz)]),
                  lda, blkm, blkn,
                  sizeof(ValueType));

            // Register StarPU handle for block (i,j)
            this->blocks[j*nr+i].register_handle(); 

         }
      }

      // Register blocks in contribution block
      this->register_contrib_blocks();

   }
#endif

   /// @brief Activate front: allocate meemory associated with factors and
   /// contrib block. Register data handles in StarPU.
   void activate(void** child_contrib) {
      this->allocate(child_contrib);
#if defined(SPLDLT_USE_STARPU)   
      this->register_node();
#endif      
   }

   /// @brief Activate front posdef: allocate meemory associated with
   /// factors and contrib block. Register data handles in StarPU.
   void activate_posdef() {
      this->allocate_posdef();
#if defined(SPLDLT_USE_STARPU)   
      this->register_node_posdef();
#endif      
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
               // STARPU_CHECK_RETURN_VALUE(ret, "starpu_memory_pin");
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
            // STARPU_CHECK_RETURN_VALUE(ret, "starpu_memory_unpin");
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
         new Backup(
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
         new sylver::ColumnData<T, IntAlloc> (
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
            // blocks[jblk*mblk + iblk] = new BlockType(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
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
   inline BlockType& get_block(int i, int j) {

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
   inline BlockType& operator()(int i, int j) {
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
   NumericFrontType* first_child; // Pointer to our first child
   NumericFrontType* next_child; // Pointer to parent's next child

   /* Data that changes during factorize */
   // Factors
   T *lcol; // Pointer to start of factor data
   // Permutations
   int *perm; // Pointer to permutation
   int *cperm; // Pointer to permutation (column)
   T *contrib; // Pointer to contribution block
   Backup *backup; // Stores backups of matrix blocks
   // Structures for indef factor
   sylver::ColumnData<T, IntAlloc> *cdata;
   std::vector<BlockType> blocks;
};

}} // End of namespaces sylver::spldlt
