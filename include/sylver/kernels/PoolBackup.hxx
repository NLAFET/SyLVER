/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Jonathan Hogg
/// @author Florent Lopez
#pragma once

#include "ssids/cpu/BlockPool.hxx"

namespace sylver {

// FIXME: there are race conditions possible if SSIDS is compiled
// without OMP because of BlockPool
   
/** \brief Stores backups of matrix blocks using a pool of memory.
 *  \details The pool is not necessarily as large as the full matrix, so in
 *  some cases allocation will have to wait until a block is released by
 *  another task. In this case, OpenMP taskyield is used.
 *  \tparam T underlying data type, e.g. double
 *  \tparam Allocator allocator to use when allocating memory
 */
template <typename T, typename Allocator=std::allocator<T*>>
class PoolBackup {
   //! \{
   typedef typename std::allocator_traits<Allocator>::template rebind_alloc<T*> TptrAlloc;
   //! \}
public:
   /** \brief Constructor
    *  \param m number of rows in matrix
    *  \param n number of blocks in matrix
    *  \param block_size dimension of a block in rows or columns
    *  \param alloc allocator instance to use when allocating memory
    */
   // FIXME: reduce pool size
   PoolBackup(int m, int n, int block_size, Allocator const& alloc=Allocator())
      : m_(m), n_(n), block_size_(block_size), mblk_(calc_nblk(m,block_size)),
        pool_(calc_nblk(n,block_size)*((calc_nblk(n,block_size)+1)/2+mblk_), block_size, alloc),
        ptr_(mblk_*calc_nblk(n,block_size), alloc)
   {}

   /** \brief Release memory associated with backup of given block.
    *  \param iblk row index of block.
    *  \param jblk column index of block.
    */
   void release(int iblk, int jblk) {
      pool_.release(ptr_[jblk*mblk_+iblk]);
      ptr_[jblk*mblk_+iblk] = nullptr;
   }

   /** \brief Create a restore point for the given block.
    *  \param iblk row index of block.
    *  \param jblk column index of block.
    *  \param aval pointer to block to be stored.
    *  \param lda leading dimension of aval.
    */
   void create_restore_point(int iblk, int jblk, T const* aval, int lda) {
      T*& lwork = ptr_[jblk*mblk_+iblk];
      lwork = pool_.get_wait();
      for(int j=0; j<get_ncol(jblk); j++)
         for(int i=0; i<get_nrow(iblk); i++)
            lwork[j*block_size_+i] = aval[j*lda+i];
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
      T*& lwork = ptr_[jblk*mblk_+iblk];
      lwork = pool_.get_wait();
      for(int j=0; j<get_ncol(jblk); j++) {
         for(int i=0; i<nperm; i++) {
            int r = perm[i];
            lwork[j*block_size_+i] = aval[j*lda+r];
         }
         for(int i=nperm; i<get_nrow(iblk); i++) {
            lwork[j*block_size_+i] = aval[j*lda+i];
         }
      }
      for(int j=0; j<get_ncol(jblk); j++)
         for(int i=0; i<nperm; i++)
            aval[j*lda+i] = lwork[j*block_size_+i];
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
   void create_restore_point_with_col_perm(int iblk, int jblk,
                                           int const* perm, T* aval, int lda) {
      T*& lwork = ptr_[jblk*mblk_+iblk];
      lwork = pool_.get_wait();
      for(int j=0; j<get_ncol(jblk); j++) {
         int c = perm[j];
         for(int i=0; i<get_nrow(iblk); i++)
            lwork[j*block_size_+i] = aval[c*lda+i];
      }
      for(int j=0; j<get_ncol(jblk); j++)
         for(int i=0; i<get_nrow(iblk); i++)
            aval[j*lda+i] = lwork[j*block_size_+i];
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
      T*& lwork = ptr_[jblk*mblk_+iblk];
      for(int j=cfrom; j<get_ncol(jblk); j++)
         for(int i=rfrom; i<get_nrow(iblk); i++)
            aval[j*lda+i] = lwork[j*block_size_+i];
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
   void restore_part_with_sym_perm(int iblk, int jblk, int from,
                                   int const* perm, T* aval, int lda) {
      T*& lwork = ptr_[jblk*mblk_+iblk];
      for(int j=from; j<get_ncol(jblk); j++) {
         int c = perm[j];
         for(int i=from; i<get_ncol(jblk); i++) {
            int r = perm[i];
            aval[j*lda+i] = (r>c) ? lwork[c*block_size_+r]
               : lwork[r*block_size_+c];
         }
         for(int i=get_ncol(jblk); i<get_nrow(iblk); i++)
            aval[j*lda+i] = lwork[c*block_size_+i];
      }
   }

private:
   /** \brief return number of columns in given block column */
   inline int get_ncol(int blk) {
      return calc_blkn(blk, n_, block_size_);
   }
   /** \brief return number of rows in given block row */
   inline int get_nrow(int blk) {
      return calc_blkn(blk, m_, block_size_);
   }

   int const m_; ///< number of rows in main matrix
   int const n_; ///< number of columns in main matrix
   int const block_size_; ///< block size of main matrix
   int const mblk_; ///< number of block rows in main matrix
   spral::ssids::cpu::BlockPool<T, Allocator> pool_; ///< pool of blocks
   std::vector<T*, TptrAlloc> ptr_; ///< map from pointer matrix entry to block
};

   
} // End of namespace sylver
