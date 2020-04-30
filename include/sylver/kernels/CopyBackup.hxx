/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Jonathan Hogg
/// @author Florent Lopez
#pragma once

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"

// SyLVER
#include "kernels/common.hxx"

namespace sylver {

/** \brief Stores backups of matrix blocks using a complete copy of matrix.
 *  \details Note that whilst a complete copy of matrix is allocated, copies
 *           of blocks are still stored individually to facilitate cache
 *           locality.
 *  \tparam T underlying data type, e.g. double
 *  \tparam Allocator allocator to use when allocating memory
 */
template <typename T, typename Allocator=std::allocator<T>>
class CopyBackup {
   // \{
   // typedef typename std::allocator_traits<Allocator>::template rebind_traits<bool> BATraits;
   using BATraits = typename std::allocator_traits<Allocator>::template rebind_traits<bool>;
   // \}
public:
   // \{
   CopyBackup(CopyBackup const&) =delete;
   CopyBackup& operator=(CopyBackup const&) =delete;
   // \}
   /** \brief constructor
    *  \param m number of rows in matrix
    *  \param n number of blocks in matrix
    *  \param block_size dimension of a block in rows or columns
    *  \param alloc allocator instance to use when allocating memory
    */
   CopyBackup(int m, int n, int block_size, Allocator const& alloc=Allocator())
      : alloc_(alloc), m_(m), n_(n), mblk_(calc_nblk(m,block_size)),
        block_size_(block_size), ldcopy_(spral::ssids::cpu::align_lda<T>(m_)),
        acopy_(alloc_.allocate(n_*ldcopy_))
   {
      typename BATraits::allocator_type boolAlloc(alloc_);
   }
   ~CopyBackup() {
      release_all_memory();
   }

   /** \brief release all associated memory; no further operations permitted.
    *  \details Storing a complete copy of the matrix is memory intensive, this
    *           routine is provided to free that storage whilst the instance is
    *           still in scope, for cases where it cannot otherwise easily be
    *           reclaimed as soon as required.
    */
   void release_all_memory() {
      if(acopy_) {
         alloc_.deallocate(acopy_, n_*ldcopy_);
         acopy_ = nullptr;
      }
   }

   /** \brief Release memory associated with backup of given block.
    *  \details Provided for compatability with PoolBackup, this
    *           routine is a no-op for CopyBackup.
    *  \param iblk row index of block.
    *  \param jblk column index of block.
    */
   void release(int iblk, int jblk) { /* no-op */ }

   /** \brief Create a restore point for the given block.
    *  \param iblk row index of block.
    *  \param jblk column index of block.
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void create_restore_point(int iblk, int jblk, T const* aval, int lda) {
      T* lwork = get_lwork(iblk, jblk);
      for(int j=0; j<get_ncol(jblk); j++)
         for(int i=0; i<get_nrow(iblk); i++)
            lwork[j*ldcopy_+i] = aval[j*lda+i];
   }

   /** \brief Apply row permutation to block and create a restore point.
    *  \details The row permutation is applied before taking the copy. This
    *           routine is provided as the row permutation requires taking a
    *           copy anyway, so they can be profitably combined.
    *  \param iblk row index of block
    *  \param jblk column index of block
    *  \param nperm number of rows to permute (allows for rectangular blocks)
    *  \param perm the permutation to apply
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void create_restore_point_with_row_perm(int iblk, int jblk, int nperm,
                                           int const* perm, T* aval, int lda) {
      T* lwork = get_lwork(iblk, jblk);
      for(int j=0; j<get_ncol(jblk); j++) {
         for(int i=0; i<nperm; i++) {
            int r = perm[i];
            lwork[j*ldcopy_+i] = aval[j*lda+r];
         }
         for(int i=nperm; i<get_nrow(iblk); i++) {
            lwork[j*ldcopy_+i] = aval[j*lda+i];
         }
      }
      for(int j=0; j<get_ncol(jblk); j++)
         for(int i=0; i<nperm; i++)
            aval[j*lda+i] = lwork[j*ldcopy_+i];
   }

   /** \brief Apply column permutation to block and create a restore point.
    *  \details The column permutation is applied before taking the copy. This
    *           routine is provided as the permutation requires taking a
    *           copy anyway, so they can be profitably combined.
    *  \param iblk row index of block
    *  \param jblk column index of block
    *  \param perm the permutation to apply
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void create_restore_point_with_col_perm(int iblk, int jblk, const int *perm, T* aval, int lda) {
      T* lwork = get_lwork(iblk, jblk);
      for(int j=0; j<get_ncol(jblk); j++) {
         int c = perm[j];
         for(int i=0; i<get_nrow(iblk); i++)
            lwork[j*ldcopy_+i] = aval[c*lda+i];
      }
      for(int j=0; j<get_ncol(jblk); j++)
         for(int i=0; i<get_nrow(iblk); i++)
            aval[j*lda+i] = lwork[j*ldcopy_+i];
   }

   /** \brief Restore submatrix (rfrom:, cfrom:) of block from backup.
    *  \param iblk row of block
    *  \param jblk column of block
    *  \param rfrom row from which to start restoration
    *  \param cfrom column from which to start restoration
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void restore_part(int iblk, int jblk, int rfrom, int cfrom, T* aval, int lda) {
      T* lwork = get_lwork(iblk, jblk);
      for(int j=cfrom; j<get_ncol(jblk); j++)
         for(int i=rfrom; i<get_nrow(iblk); i++)
            aval[j*lda+i] = lwork[j*ldcopy_+i];
   }

   /** \brief Restore submatrix (from:, from:) from a symmetric permutation of
    *         backup.
    *  \details The backup will have been stored pritor to a symmetric
    *           permutation associated with the factorization of a diagonal
    *           block. This routine restores any failed columns, taking into
    *           account the supplied permutation.
    *  \param iblk row of block
    *  \param jblk column of block
    *  \param from row and column from which to start restoration
    *  \param perm permutation to apply
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void restore_part_with_sym_perm(int iblk, int jblk, int from, const int *perm, T* aval, int lda) {
      T* lwork = get_lwork(iblk, jblk);
      for(int j=from; j<get_ncol(jblk); j++) {
         int c = perm[j];
         for(int i=from; i<get_ncol(jblk); i++) {
            int r = perm[i];
            aval[j*lda+i] = (r>c) ? lwork[c*ldcopy_+r]
               : lwork[r*ldcopy_+c];
         }
         for(int i=get_ncol(jblk); i<get_nrow(iblk); i++)
            aval[j*lda+i] = lwork[c*ldcopy_+i];
      }
   }

private:
   /** \brief returns pointer to internal backup of given block */
   inline T* get_lwork(int iblk, int jblk) {
      return &acopy_[jblk*block_size_*ldcopy_+iblk*block_size_];
   }
   /** \brief return number of columns in given block column */
   inline int get_ncol(int blk) const {
      return calc_blkn(blk, n_, block_size_);
   }
   /** \brief return number of rows in given block row */
   inline int get_nrow(int blk) const {
      return calc_blkn(blk, m_, block_size_);
   }

   Allocator alloc_; ///< internal copy of allocator needed for destructor
   int const m_; ///< number of rows in matrix
   int const n_; ///< number of columns in matrix
   int const mblk_; ///< number of block rows in matrix
   int const block_size_; ///< block size
   size_t const ldcopy_; ///< leading dimension of acopy_
   T* acopy_; ///< internal storage for copy of matrix
};

   
} // End of namespace sylver
