#pragma once

#include "kernels/ldlt_app.hxx"
#include "kernels/factor_unsym.hxx"

// STD
#include <cassert>

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"

namespace spldlt {

   // Block structure for unsymmetric matrices
   //
   // TODO Create generic block type to cover sym posdef, indef and
   // unsym matrices
   template<typename T>
   class BlockUnsym {
   public:
      BlockUnsym()
         : m(0), n(0), a(nullptr), lda(0), b(nullptr), ldb(0), mb_(0), nb_(0),
           lrperm_(nullptr), cpy_(nullptr), ldcpy_(0)
      {}

      BlockUnsym(int i, int j, int m, int n, T* a, int lda)
         : i(i), j(j), m(m), n(n), a(a), lda(lda), mb_(0), nb_(0),
           lrperm_(nullptr), cpy_(nullptr), ldcpy_(0)
      {}

      BlockUnsym(int i, int j, int m, int n, T* a, int lda, int mb, int nb,
                 T* b, int ldb)
         : i(i), j(j), m(m), n(n), a(a), lda(lda), mb_(mb), nb_(nb), b(b),
           ldb(ldb), lrperm_(nullptr), cpy_(nullptr), ldcpy_(0)
      {
         assert(mb_ <= m);
         assert(nb_ < n);
      }

      ~BlockUnsym() {
         release_lrperm();
      }
      
      void alloc_lrperm() {
         // TODO use PoolAlloc
         if (!lrperm_ && (m > 0))
            lrperm_ = new int[m];
      }

      void alloc_init_lrperm() {
         alloc_lrperm();
         for (int i=0; i<m; ++i) lrperm_[i] = i;
      }
      
      void release_lrperm() {
         if(lrperm_)
            delete[] lrperm_;
      }

      /// @brief Get number of fully-summed rows and columns in a
      /// block lying on the digonal
      inline int get_nfs() {
         if (i != j) return -1;
         return get_na();
      }
      
      inline int *get_lrperm() { return lrperm_; }
      inline int get_mb() { return mb_;}
      inline int get_nb() { return nb_;}

      /// @brief Get number of rows in a
      inline int get_ma() {
         return m;
      }

      /// @brief Get number of columns in a
      inline int get_na() {
         return n-get_nb();
      }

      /// @ Allocate copy of block
      template <typename Allocator>
      void alloc_backup(Allocator& alloc) {
         if (!cpy_) {
            ldcpy_ = spral::ssids::cpu::align_lda<T>(m);
            cpy_ = alloc.allocate(ldcpy_*n);
         }
      }

      /// @brief Release copy of block
      template <typename Allocator>
      void release_backup(Allocator& alloc) {
         if (!cpy_) return;
         alloc.deallocate(cpy_, ldcpy_*n);
         cpy_ = nullptr;
      }
      
      /// @brief Create a backup of this block
      void backup() {

         // Copy part from b
         int na = get_na();         
         for (int j = 0; j < na; ++j) {
            for (int i = 0; i < m; ++i) {
               cpy_[j*ldcpy_+i] = a[j*lda+i];
            }
         }

         // Copy part from b
         if (n > na) {
            for (int j = 0; j < nb_; ++j) {
               for (int i = 0; i < mb_; ++i) {
                  cpy_[j*ldcpy_+i] = b[j*lda+i];
               }
            }
         }
      }

      /// @brief Return the block row index
      inline int get_row() { return i; }
      /// @brief Return the block column index
      inline int get_col() { return j; }

      ////////////////////////////////////////
      // Factorization operations

      /// @brief Factorize block and update both row rperm and column
      /// permutation cperm
      int factor(int *rperm, int *cperm) {
         return factor_lu_pp(rperm);
      }

      /// @brief Factorize block using LU with partial pivoting and
      /// update global permutation perm
      ///
      /// @param perm Global permutation matrix
      int factor_lu_pp(int *perm) {

         int nelim = 0;
         // dblk.alloc_lrperm();
         alloc_init_lrperm();
         // Number of fully-summed rows/columns in dblk
         int nfs = get_nfs();
      
         // Note: lrperm is 0-indexed in factor_block_lu_pp 
         factor_block_lu_pp(
               m, nfs, lrperm_, a, lda, b, ldb);
         // TODO block might be arbitrarily close to singularity so
         // nelim might be lower that nsf
         nelim = nfs;

         // printf("nfs = %d\n", nfs);
         // printf("lrperm\n");          
         // for (int i=0; i < nfs; ++i) printf(" %d ", lrperm[i]);
         // printf("\n");            

         // Update perm using local permutation lrperm
         int *temp = new int[nfs];
         for (int i = 0; i < nfs; ++i)
            temp[i] = perm[lrperm_[i]];
         for (int i = 0; i < nfs; ++i)
            perm[i] = temp[i]; 
         delete[] temp;

         return nelim;
      }
      
      /// @brief Apply U factor to this block and check for failed
      /// entries i.e. entries such that |l_pq| > u^{-1}. 
      ///
      /// @param u threshold parameter.
      /// @returns Number of columns successfully eliminated columns
      /// with respect to threshold u.
      template <typename IntAlloc>
      int applyU_app(
            BlockUnsym<T> const& dblk, T u, 
            spldlt::ldlt_app_internal::ColumnData<T, IntAlloc>& cdata) {
         
         assert(get_row() != dblk.get_row());
         
         if (get_row() < dblk.get_row()) {
            // Super-diagonal block
            applyU_block(
                  dblk.m-cdata[i].nelim , cdata[j].nelim, 
                  dblk.a, dblk.lda, &a[cdata[i].nelim], lda);
            return spldlt::ldlt_app_internal::check_threshold<OP_N>(
                  cdata[i].nelim, m, 0, cdata[j].nelim, u, a, lda);
         }
         else {
            // Sub-diagonal block
            applyU_block(
                  m, cdata[j].nelim, dblk.a, dblk.lda, a, lda);            
            return spldlt::ldlt_app_internal::check_threshold<OP_N>(
                  0, m, 0, cdata[j].nelim, u, a, lda);
         }
      }

      /// @brief Apply L factor to this block
      /// @Note This routine assume that permutation has already been
      /// applied
      template <typename IntAlloc>
      void applyL(BlockUnsym<T> const& dblk) {

         int na = get_na();
         
         applyL_block(
               m, na, dblk.a, dblk.lda, a, lda);
         
         if (nb_ > 0) {
            
            assert(b != nullptr);            
            // Apply L in parts stored un ucol
            applyL_block(
                  mb_, nb_, dblk.a, dblk.lda, b, ldb);
         }         
      }

      /// @brief Apply L factor to this block when using APTP
      /// strategy.
      template <typename IntAlloc>
      void applyL_app(
            BlockUnsym<T> const& dblk,
            spldlt::ldlt_app_internal::ColumnData<T, IntAlloc>& cdata) {

         assert(get_col() != dblk.get_col());

         if (get_col() < dblk.get_col()) {
            // Left-digonal block
            applyL_block(
                  cdata[i].nelim, n-cdata[j].nelim, dblk.a, dblk.lda,
                  &a[lda*(n-cdata[j].nelim)], lda);
         }
         else {
            
            // Right-digonal block
            int na = get_na();
            applyL_block(
               cdata[i].nelim, na, dblk.a, dblk.lda, a, lda);

            if (nb_ > 0) {
            
               assert(b != nullptr);            
               assert(cdata[i].nelim <= mb_);
               
               // Apply L in parts stored un ucol
               applyL_block(
                     cdata[i].nelim, nb_, dblk.a, dblk.lda, b, ldb);
            }
         }
      }
      
      ////////////////////////////////////////
      
      
      int i; // block row index
      int j; // block column index
      int m; // Number of rows in block
      int n; // Number of columns in block
      int lda; // leading dimension of underlying storage
      T* a; // pointer to underlying matrix storage
      // Note that the block might be slip in two different memory
      // areas (e.g. one for factor L and another one for U). In this
      // case we use b to point on the reminder:
      int ldb; // leading dimension of underlying storage
      T* b; // pointer to underlying matrix storage  
   private:
      int mb_;
      int nb_;
      int *lrperm_; // Local row permutation
      T *cpy_; // Copy of block 
      int ldcpy_;
      
      // case 1) Block is in a single memory location (lcol or ucol)
      // e.g.
     
      // +-------+
      // |       |
      // |   a   | 
      // |       |
      // +-------+

      // case 2) Block split between 2 memory locations(lcol and ucol)
      // e.g.
      
      // +---+---+
      // |   |   |
      // | a | b | 
      // |   |   |
      // +---+---+

      // or

      // +---+---+  ---   ---
      // |   | b |   | mb  |
      // | a +---+  ---    | m
      // |   |             | 
      // +---+            ---
      //
      // |---|---|
      //   na  nb
      // |-------|
      //     n
   };
   
} // namespaces spldlt
