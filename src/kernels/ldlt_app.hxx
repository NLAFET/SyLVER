/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Jonathan Hogg
/// @author Florent Lopez
#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <ostream>
#include <sstream>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

#include "compat.hxx"
#include "ssids/profile.hxx"
#include "ssids/cpu/BlockPool.hxx"
// #include "ssids/cpu/BuddyAllocator.hxx"
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/Workspace.hxx"
#include "ssids/cpu/kernels/block_ldlt.hxx"
#include "ssids/cpu/kernels/calc_ld.hxx"
#include "ssids/cpu/kernels/ldlt_tpp.hxx"
#include "ssids/cpu/kernels/common.hxx"
#include "ssids/cpu/kernels/wrappers.hxx"

#if defined(SPLDLT_USE_STARPU)
#include <starpu.h>
#include <mutex>
// #include <atomic>
#endif

#include "BuddyAllocator.hxx"

// namespace spral { namespace ssids { namespace cpu {
namespace spldlt {

   const int INNER_BLOCK_SIZE = 32;
   
   template<typename T, int iblksz, typename Backup, bool debug,
            typename Allocator>
   class FactorSymIndef;

using namespace spral::ssids::cpu;

template<typename T, typename Allocator>
int ldlt_app_factor(int m, int n, int *perm, T *a, int lda, T *d, T beta, T* upd, int ldupd, struct cpu_factor_options const& options, std::vector<spral::ssids::cpu::Workspace>& work, Allocator const& alloc);

template <typename T>
void ldlt_app_solve_fwd(int m, int n, T const* l, int ldl, int nrhs, T* x, int ldx);

template <typename T>
void ldlt_app_solve_diag(int n, T const* d, int nrhs, T* x, int ldx);

template <typename T>
void ldlt_app_solve_bwd(int m, int n, T const* l, int ldl, int nrhs, T* x, int ldx);

namespace ldlt_app_internal {

   // static const int INNER_BLOCK_SIZE = 32;

/** \return number of blocks for given n */
inline int calc_nblk(int n, int block_size) {
   return (n-1) / block_size + 1;
}

/** \return block size of block blk if maximum in dimension is n */
inline int calc_blkn(int blk, int n, int block_size) {
   return std::min(block_size, n-blk*block_size);
}

/** \brief Data about block column of factorization; handles operations
 *         concerning number of eliminated variables and stores D.
 *  \tparam T underlying data type, e.g. double
 */
template<typename T>
class Column {
public:
   bool first_elim; ///< True if first column with eliminations
   int nelim; ///< Number of eliminated entries in this column
   T *d; ///< Pointer to local d

   // \{
   Column(Column const&) =delete; // must be unique
   Column& operator=(Column const&) =delete; // must be unique
   Column() =default;
   // \}

   /** \brief Initialize number of passed columns ready for reduction
    *  \param passed number of variables passing a posteori pivot test in block
    */
   void init_passed(int passed) {
      npass_ = passed;
   }
   /** \brief Update number of passed columns.
    *  \details Aquires a lock before doing a minimum reduction across blocks
    *  \param passed number of variables passing a posteori pivot test in block
    */
   void update_passed(int passed) {
      
#if defined(SPLDLT_USE_STARPU)
      std::lock_guard<std::mutex> lock(mtx);
#else
      spral::omp::AcquiredLock scopeLock(lock_);
#endif
      npass_ = std::min(npass_, passed);
   }
   /** \brief Test if column has failed (in unpivoted case), recording number of
    *         blocks in column that have passed. To be called once per block
    *         in the column.
    *  \details Whilst this check could easily be done without calling this
    *           routine, the atomic recording of number that have passed would
    *           not be done, and this is essential for calculating number of
    *           sucessful columns in the case of a global cancellation.
    *  \param passed number of pivots that succeeded for a block
    *  \returns true if passed < nelim */
   bool test_fail(int passed) {
      bool fail = (passed < nelim);
      if(!fail) {
         // Record number of blocks in column passing this test
         
         // TODO find an alternative to the omp atomic update when omp
         // is not used.
         #pragma omp atomic update
         ++npass_;
      }
      return fail;
   }

   /** \brief Adjust nelim after all blocks of row/column have completed to
    *         avoid split 2x2 pivots. Also updates next_elim.
    *  \details If a split 2x2 pivot is detected, the number of eliminated
    *           variables is reduced by one. This routine also sets first_elim
    *           to true if this is the first column to successfully eliminated
    *           a variable, and sets nelim for this column.
    *  \param next_elim global number of eliminated pivots to be updated based
    *         on number eliminated in this column. */
   void adjust(int& next_elim) {
      // Test if last passed column was first part of a 2x2: if so,
      // decrement npass
      if(npass_>0) {
         T d11 = d[2*(npass_-1)+0];
         T d21 = d[2*(npass_-1)+1];
         if(std::isfinite(d11) && // not second half of 2x2
               d21 != 0.0)        // not a 1x1 or zero pivot
            npass_--;              // so must be first half 2x2
      }
      // Update elimination progress
      first_elim = (next_elim==0 && npass_>0);
      next_elim += npass_;
      nelim = npass_;
   }

   /** \brief Move entries of permutation for eliminated entries backwards to
    *         close up space from failed columns, whilst extracting failed
    *         entries.
    *  \details n entries of perm are moved to elim_perm (that may overlap
    *           with perm). Uneliminated variables are placed into failed_perm.
    *  \param n number of entries in block to be moved to elim_perm or failed.
    *  \param perm[n] source pointer
    *  \param elim_perm destination pointer for eliminated columns
    *         from perm, first nelim entries are filled on output.
    *  \param failed_perm destination pointer for failed columns from
    *         perm first (n-nelim) entries are filled on output.
    *  \internal Note that there is no need to consider a similar operation for
    *            d[] as it is only used for eliminated variables.
    */
   void move_back(int n, int const* perm, int* elim_perm, int* failed_perm) {
      if(perm != elim_perm) { // Don't move if memory is identical
         for(int i=0; i<nelim; ++i)
            *(elim_perm++) = perm[i];
      }
      // Copy failed perm
      for(int i=nelim; i<n; ++i)
         *(failed_perm++) = perm[i];
   }

   /** \brief return number of passed columns */
   int get_npass() const { return npass_; }

#if defined(SPLDLT_USE_STARPU)
   /* return StarPU handle on column */
   starpu_data_handle_t get_hdl() const { return hdl_; }
   /* register handle on column*/
   void register_handle() {
      starpu_void_data_register(&hdl_);
   }

   // @brief Unregister data handle on column in StarPU
   template<bool async=true>
   void unregister_handle() {
      if (async) starpu_data_unregister_submit(hdl_);
      else       starpu_data_unregister(hdl_);
   }

   /// @brief Return StarPU handle on diagonal D
   starpu_data_handle_t get_d_hdl() const { return d_hdl_; }

   void register_d_hdl(const T *d, int dimn) {
      starpu_vector_data_register(
            &d_hdl_, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(d), dimn,
            sizeof(T));
   }

   template<bool async=true>
   void unregister_d_hdl() {
      if (async) starpu_data_unregister_submit(d_hdl_);
      else       starpu_data_unregister(d_hdl_);
   }

#endif

private:
#if defined(SPLDLT_USE_STARPU)
   starpu_data_handle_t hdl_;

   // spral::omp::Lock lock_; ///< lock for altering npass
   int npass_=0; ///< reduction variable for nelim

   std::mutex mtx;
   // std::atomic<int> npass_=0; ///< reduction variable for nelim

   // StarPU handle for diagonal D
   starpu_data_handle_t d_hdl_;

#else
   spral::omp::Lock lock_; ///< lock for altering npass
   int npass_=0; ///< reduction variable for nelim
#endif
};

/** \brief Stores data about block columns
 *  \details A wrapper around a vector of Column, also handles local permutation
 *           vector and calculation of nelim in unpivoted factorization
 *  \tparam T underlying datatype e.g. double
 *  \tparam IntAlloc Allocator specialising in int used for internal memory
 *          allocation.
 * */
template<typename T, typename IntAlloc>
class ColumnData {
   // \{
   typedef typename std::allocator_traits<IntAlloc>::template rebind_traits<Column<T>> ColAllocTraits;
   typedef typename std::allocator_traits<IntAlloc> IntAllocTraits;
   // \}
public:
   // \{
   ColumnData(ColumnData const&) =delete; //not copyable
   ColumnData& operator=(ColumnData const&) =delete; //not copyable
   // \}
   /** \brief Constructor
    *  \param n number of columns
    *  \param block_size block size
    *  \param alloc allocator instance to use for allocation
    */
   ColumnData(int n, int block_size, IntAlloc const& alloc)
   : n_(n), block_size_(block_size), alloc_(alloc)
   {
      int nblk = calc_nblk(n_, block_size_);
      typename ColAllocTraits::allocator_type colAlloc(alloc_);
      cdata_ = ColAllocTraits::allocate(colAlloc, nblk);
      for(int i=0; i<nblk; ++i)
         ColAllocTraits::construct(colAlloc, &cdata_[i]);
      lperm_ = IntAllocTraits::allocate(alloc_, nblk*block_size_);
   }
   ~ColumnData() {
      int nblk = calc_nblk(n_, block_size_);
      IntAllocTraits::deallocate(alloc_, lperm_, nblk*block_size_);
      typename ColAllocTraits::allocator_type colAlloc(alloc_);
      ColAllocTraits::deallocate(colAlloc, cdata_, nblk);
   }

   /** \brief Returns Column instance for given column
    *  \param idx block column
    */
   Column<T>& operator[](int idx) { return cdata_[idx]; }

   /** \brief Return local permutation pointer for given column
    *  \param blk block column
    *  \return pointer to local permutation
    */
   int* get_lperm(int blk) { return &lperm_[blk*block_size_]; }

   /** \brief Calculate number of eliminated columns in unpivoted case
    *  \param m number of rows in matrix
    *  \return number of sucesfully eliminated columns
    */
   int calc_nelim(int m) const {
      int mblk = calc_nblk(m, block_size_);
      int nblk = calc_nblk(n_, block_size_);
      int nelim = 0;
      for(int j=0; j<nblk; ++j) {
         if(cdata_[j].get_npass() == mblk-j) {
            nelim += cdata_[j].nelim;
         } else {
            break; // After first failure, no later pivots are valid
         }
      }
      return nelim;
   };

// #if defined(SPLDLT_USE_STARPU)
//    starpu_data_handle_t get_d_hdl() const { return d_hdl_; }

//    void register_d_hdl(const T *d) {
//       starpu_vector_data_register(
//             &d_hdl_, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(d), 2*n_,
//             sizeof(T));
//    }

//    void unregister_d_hdl() {
//       starpu_data_unregister_submit(d_hdl_);
//    }
// #endif

private:
   int const n_; ///< number of columns in matrix
   int const block_size_; ///< block size for matrix
   IntAlloc alloc_; ///< internal copy of allocator to be used in destructor
   Column<T> *cdata_; ///< underlying array of columns
   int* lperm_; ///< underlying local permutation
#if defined(SPLDLT_USE_STARPU)
   starpu_data_handle_t d_hdl_;
#endif
};


/** Returns true if ptr is suitably aligned for AVX, false if not */
// bool is_aligned(void* ptr) {
//    const int align = 32;
//    return (reinterpret_cast<uintptr_t>(ptr) % align == 0);
// }

/** Move up eliminated entries to fill any gaps left by failed pivots
 *  within diagonal block.
 *  Note that out and aval may overlap. */
template<typename T, typename Column>
void move_up_diag(Column const& idata, Column const& jdata, T* out, T const* aval, int lda) {
   if(out == aval) return; // don't bother moving if memory is the same
   for(int j=0; j<jdata.nelim; ++j)
   for(int i=0; i<idata.nelim; ++i)
      out[j*lda+i] = aval[j*lda+i];
}

/** Move up eliminated entries to fill any gaps left by failed pivots
 *  within rectangular block of matrix.
 *  Note that out and aval may overlap. */
template<typename T, typename Column>
void move_up_rect(int m, int rfrom, Column const& jdata, T* out, T const* aval, int lda) {
   if(out == aval) return; // don't bother moving if memory is the same
   for(int j=0; j<jdata.nelim; ++j)
   for(int i=rfrom; i<m; ++i)
      out[j*lda+i] = aval[j*lda+i];
}

/** Copies failed rows and columns^T to specified locations */
template<typename T, typename Column>
void copy_failed_diag(int m, int n, Column const& idata, Column const& jdata, T* rout, T* cout, T* dout, int ldout, T const* aval, int lda) {
   /* copy rows */
   for(int j=0; j<jdata.nelim; ++j)
   for(int i=idata.nelim, iout=0; i<m; ++i, ++iout)
      rout[j*ldout+iout] = aval[j*lda+i];
   /* copy cols in transpose (not for diagonal block) */
   if(&idata != &jdata) {
      for(int j=jdata.nelim, jout=0; j<n; ++j, ++jout)
      for(int i=0; i<idata.nelim; ++i)
         cout[i*ldout+jout] = aval[j*lda+i];
   }
   /* copy intersection of failed rows and cols */
   for(int j=jdata.nelim, jout=0; j<n; j++, ++jout)
   for(int i=idata.nelim, iout=0; i<m; ++i, ++iout)
      dout[jout*ldout+iout] = aval[j*lda+i];
}

/** Copies failed columns to specified location */
template<typename T, typename Column>
void copy_failed_rect(int m, int n, int rfrom, Column const& jdata, T* cout, int ldout, T const* aval, int lda) {
   for(int j=jdata.nelim, jout=0; j<n; ++j, ++jout)
      for(int i=rfrom; i<m; ++i)
         cout[jout*ldout+i] = aval[j*lda+i];
}

/** Check if a block satisifies pivot threshold (colwise version) */
template <enum operation op, typename T>
int check_threshold(int rfrom, int rto, int cfrom, int cto, T u, T* aval, int lda) {
   // Perform threshold test for each uneliminated row/column
   int least_fail = (op==OP_N) ? cto : rto;
   for(int j=cfrom; j<cto; j++)
   for(int i=rfrom; i<rto; i++)
      if(fabs(aval[j*lda+i]) > 1.0/u) {
         // printf("[check_threshold] failed entry = %f\n", fabs(aval[j*lda+i]));
         if(op==OP_N) {
            // must be least failed col
            return j;
         } else {
            // may be an earlier failed row
            least_fail = std::min(least_fail, i);
            break;
         }
      }
   // If we get this far, everything is good
   return least_fail;
}

/** Performs solve with diagonal block \f$L_{21} = A_{21} L_{11}^{-T} D_1^{-1}\f$. Designed for below diagonal. */
/* NB: d stores (inverted) pivots as follows:
 * 2x2 ( a b ) stored as d = [ a b Inf c ]
 *     ( b c )
 * 1x1  ( a )  stored as d = [ a 0.0 ]
 * 1x1  ( 0 ) stored as d = [ 0.0 0.0 ]
 */
template <enum operation op, typename T>
void apply_pivot(int m, int n, int from, const T *diag, const T *d, const T small, T* aval, int lda) {
   if(op==OP_N && from > m) return; // no-op
   if(op==OP_T && from > n) return; // no-op

   if(op==OP_N) {
      // Perform solve L_11^-T
      host_trsm<T>(SIDE_RIGHT, FILL_MODE_LWR, OP_T, DIAG_UNIT,
            m, n, 1.0, diag, lda, aval, lda);
      // Perform solve L_21 D^-1
      for(int i=0; i<n; ) {
         if(i+1==n || std::isfinite(d[2*i+2])) {
            // 1x1 pivot
            T d11 = d[2*i];
            if(d11 == 0.0) {
               // Handle zero pivots carefully
               for(int j=0; j<m; j++) {
                  T v = aval[i*lda+j];
                  aval[i*lda+j] = 
                     (fabs(v)<small) ? 0.0
                                     : std::numeric_limits<T>::infinity()*v;
                  // NB: *v above handles NaNs correctly
               }
            } else {
               // Non-zero pivot, apply in normal fashion
               for(int j=0; j<m; j++)
                  aval[i*lda+j] *= d11;
            }
            i++;
         } else {
            // 2x2 pivot
            T d11 = d[2*i];
            T d21 = d[2*i+1];
            T d22 = d[2*i+3];
            for(int j=0; j<m; j++) {
               T a1 = aval[i*lda+j];
               T a2 = aval[(i+1)*lda+j];
               aval[i*lda+j]     = d11*a1 + d21*a2;
               aval[(i+1)*lda+j] = d21*a1 + d22*a2;
            }
            i += 2;
         }
      }
   } else { /* op==OP_T */
      // Perform solve L_11^-1
      host_trsm<T>(SIDE_LEFT, FILL_MODE_LWR, OP_N, DIAG_UNIT,
            m, n-from, 1.0, diag, lda, &aval[from*lda], lda);
      // Perform solve D^-T L_21^T
      for(int i=0; i<m; ) {
         if(i+1==m || std::isfinite(d[2*i+2])) {
            // 1x1 pivot
            T d11 = d[2*i];
            if(d11 == 0.0) {
               // Handle zero pivots carefully
               for(int j=from; j<n; j++) {
                  T v = aval[j*lda+i];
                  aval[j*lda+i] = 
                     (fabs(v)<small) ? 0.0 // *v handles NaNs
                                     : std::numeric_limits<T>::infinity()*v;
                  // NB: *v above handles NaNs correctly
               }
            } else {
               // Non-zero pivot, apply in normal fashion
               for(int j=from; j<n; j++) {
                  aval[j*lda+i] *= d11;
               }
            }
            i++;
         } else {
            // 2x2 pivot
            T d11 = d[2*i];
            T d21 = d[2*i+1];
            T d22 = d[2*i+3];
            for(int j=from; j<n; j++) {
               T a1 = aval[j*lda+i];
               T a2 = aval[j*lda+(i+1)];
               aval[j*lda+i]     = d11*a1 + d21*a2;
               aval[j*lda+(i+1)] = d21*a1 + d22*a2;
            }
            i += 2;
         }
      }
   }
}

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
   typedef typename std::allocator_traits<Allocator>::template rebind_traits<bool> BATraits;
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
     block_size_(block_size), ldcopy_(align_lda<T>(m_)),
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
   BlockPool<T, Allocator> pool_; ///< pool of blocks
   std::vector<T*, TptrAlloc> ptr_; ///< map from pointer matrix entry to block
};

// template<typename T,
//          int BLOCK_SIZE,
//          typename Backup,
//          bool use_tasks, // Use tasks, so we can disable on one or more levels
//          bool debug=false,
//          typename Allocator=std::allocator<T>
//          >
//class LDLT;

/** \brief Functional wrapper around a block of the underlying matrix.
 *  \details Provides a light-weight wrapper around blocks of the matrix
 *           to provide location-aware functionality and thus safety.
 *  \tparam T Underlying datatype, e.g. double.
 *  \tparam INNER_BLOCK_SIZE The inner block size to be used for recursion
 *          decisions in factor().
 *  \tparam IntAlloc an allocator for type int used in specification of
 *          ColumnData type.
 */
template<typename T, int INNER_BLOCK_SIZE, typename IntAlloc>
class Block {
public:
   /** \brief Constuctor.
    *  \param i Block's row index.
    *  \param j Block's column index.
    *  \param m Number of rows in matrix.
    *  \param n Number of columns in matrix.
    *  \param cdata ColumnData for factorization.
    *  \param a Pointer to underlying storage of matrix.
    *  \param lda Leading dimension of a.
    *  \param block_size The block size.
    */
   Block(int i, int j, int m, int n, ColumnData<T,IntAlloc>& cdata, T* a,
         int lda, int block_size)
   : i_(i), j_(j), m_(m), n_(n), lda_(lda), block_size_(block_size),
     cdata_(cdata), aval_(a) // aval_(&a[j*block_size*lda+i*block_size])
   {}

   /** \brief Create backup of this block.
    *  \tparam Backup Underlying backup type.
    *  \param backup Storage containing backup.
    */
   template <typename Backup>
   void backup(Backup& backup) {
      backup.create_restore_point(i_, j_, aval_, lda_);
   }

   /** \brief Apply column permutation to block and create a backup.
    *  \tparam Backup Underlying backup type.
    *  \param backup Storage containing backup.
    */
   template <typename Backup>
   void apply_rperm_and_backup(Backup& backup) {
      backup.create_restore_point_with_row_perm(
            i_, j_, get_ncol(i_), cdata_.get_lperm(i_), aval_, lda_
            );
   }

   /** \brief Apply row permutation to block.
    *  \param work Thread-specific workspace.
    */
   void apply_rperm(spral::ssids::cpu::Workspace& work) {
      int ldl = align_lda<T>(block_size_);
      T* lwork = work.get_ptr<T>(ncol()*ldl);
      int* lperm = cdata_.get_lperm(i_);
      // Copy into lwork with permutation
      for(int j=0; j<ncol(); ++j) {
         for(int i=0; i<get_ncol(i_); ++i) {
            int r = lperm[i];
            lwork[j*ldl+i] = aval_[j*lda_+r];
         }
      }
      // Copy back again
      for(int j=0; j<ncol(); ++j)
      for(int i=0; i<get_ncol(i_); ++i)
         aval_[j*lda_+i] = lwork[j*ldl+i];
   }

   /** \brief Apply inverse of row permutation to block.
    *  \details Intended for recovery from failed Cholesky-like factorization.
    *  \param work Thread-specific workspace.
    */
   void apply_inv_rperm(spral::ssids::cpu::Workspace& work) {
      int ldl = align_lda<T>(block_size_);
      T* lwork = work.get_ptr<T>(ncol()*ldl);
      int* lperm = cdata_.get_lperm(i_);
      // Copy into lwork with permutation
      for(int j=0; j<ncol(); ++j) {
         for(int i=0; i<get_ncol(i_); ++i) {
            int r = lperm[i];
            lwork[j*ldl+r] = aval_[j*lda_+i];
         }
      }
      // Copy back again
      for(int j=0; j<ncol(); ++j)
      for(int i=0; i<get_ncol(i_); ++i)
         aval_[j*lda_+i] = lwork[j*ldl+i];
   }

   /** \brief Apply column permutation to block and create a backup.
    *  \tparam Backup Underlying backup type.
    *  \param backup Storage containing backup.
    */
   template <typename Backup>
   void apply_cperm_and_backup(Backup& backup) {
      backup.create_restore_point_with_col_perm(
            i_, j_, cdata_.get_lperm(j_), aval_, lda_
            );
   }

   /** \brief Apply column permutation to block.
    *  \param work Thread-specific workspace.
    */
   void apply_cperm(Workspace& work) {
      int ldl = align_lda<T>(block_size_);
      T* lwork = work.get_ptr<T>(ncol()*ldl);
      int* lperm = cdata_.get_lperm(j_);
      // Copy into lwork with permutation
      for(int j=0; j<ncol(); ++j) {
         int c = lperm[j];
         for(int i=0; i<nrow(); ++i)
            lwork[j*ldl+i] = aval_[c*lda_+i];
      }
      // Copy back again
      for(int j=0; j<ncol(); ++j)
      for(int i=0; i<nrow(); ++i)
         aval_[j*lda_+i] = lwork[j*ldl+i];
   }

   /** \brief Restore the entire block from backup.
    *  \details Intended for recovery from failed Cholesky-like factorization.
    *  \tparam Backup Underlying backup type.
    *  \param backup Storage containing backup.
    */
   template <typename Backup>
   void full_restore(Backup& backup) {
      backup.restore_part(i_, j_, 0, 0, aval_, lda_);
   }

   /** \brief Restore any failed columns from backup.
    *  \details Storage associated with backup is released by this routine
    *           once we are done with it. This routine should only be called
    *           for blocks in the eliminated row/column.
    *  \tparam Backup Underlying backup type.
    *  \param backup Storage containing backup.
    *  \param elim_col The block column we've just finished eliminating and
    *         wish to perform restores associated with.
    */
   template <typename Backup>
   void restore_if_required(Backup& backup, int elim_col) {
      if(i_ == elim_col && j_ == elim_col) { // In eliminated diagonal block
         if(cdata_[i_].nelim < ncol()) { // If there are failed pivots
            backup.restore_part_with_sym_perm(
                  i_, j_, cdata_[i_].nelim, cdata_.get_lperm(i_), aval_, lda_
                  );
         }
         // Release resources regardless, no longer required
         backup.release(i_, j_);
      }
      else if(i_ == elim_col) { // In eliminated row
         if(cdata_[i_].nelim < nrow()) // If there are failed pivots
            backup.restore_part(
                  i_, j_, cdata_[i_].nelim, cdata_[j_].nelim, aval_, lda_
                  );
         // Release resources regardless, no longer required
         backup.release(i_, j_);
      }
      else if(j_ == elim_col) { // In eliminated col
         if(cdata_[j_].nelim < ncol()) { // If there are failed pivots
            int rfrom = (i_ <= elim_col) ? cdata_[i_].nelim : 0;
            backup.restore_part(i_, j_, rfrom, cdata_[j_].nelim, aval_, lda_);
         }
         // Release resources regardless, no longer required
         backup.release(i_, j_);
      }
   }

   /** \brief Factorize diagonal block.
    *  \details Performs the in-place factorization
    *           \f[ A_{ii} = P L_{ii} D_i L_{ii}^T P^T. \f]
    *           The mechanism to do so varies:
    *           - If block_size != BLOCK_SIZE then recurse with a call to
    *             LDLT::factor() using BLOCK_SIZE as the new block size.
    *           - Otherwise, if the block is a full block of size BLOCK_SIZE,
    *             call block_ldlt().
    *           - Otherwise, if the block is not full, call ldlt_tpp_factor().
    *           Note that two permutations are maintained, the user permutation
    *           perm, and the local permutation lperm obtained from
    *           ColumnData::get_lperm() that represents P above.
    *  \tparam Allocator allocator type to be used on recursion to
    *          LDLT::factor().
    *  \param next_elim Next variable to be eliminated, used to determine
    *         location in d to be used.
    *  \param perm User permutation: entries are permuted in same way as
    *         matrix columns.
    *  \param d pointer to global array for D.
    *  \param options user-supplied options
    *  \param work vector of thread-specific workspaces
    *  \param alloc allocator instance to be used on recursion to
    *         LDLT::factor().
    */
   template <typename Allocator>
   int factor(int next_elim, int* perm, T* d,
              struct cpu_factor_options &options,
              spral::ssids::cpu::Workspace& work, /* std::vector<spral::ssids::cpu::Workspace>& work, */Allocator const& alloc) {
      if(i_ != j_)
         throw std::runtime_error("factor called on non-diagonal block!");
      int* lperm = cdata_.get_lperm(i_);
      for(int i=0; i<ncol(); i++)
         lperm[i] = i;
      cdata_[i_].d = &d[2*next_elim];
      int id = 0;
// #if defined(SPLDLT_USE_STARPU)
//       id = std::max(starpu_worker_get_id(), 0);
//       // printf("Block::factor, worker id: %d\n", id);
// #else
//       id = omp_get_thread_num();
// #endif
      if(block_size_ != INNER_BLOCK_SIZE) {

         // Recurse
         CopyBackup<T, Allocator> inner_backup(
               nrow(), ncol(), INNER_BLOCK_SIZE, alloc
               );
         bool const use_tasks = false; // Don't run in parallel at lower level
         bool const debug = false; // Don't print debug info for inner call

         // workspace because we call factor without using tasks.

         // FIXME: note that we are recurssing on the factor routine
         // from the LDLT class (taken from SSIDS) because we run it in
         // sequential. It would certainly be simpler to recurse on
         // the ldlt_app routine from FactorSymIndef class.
         
         // cdata_[i_].nelim =
         //    LDLT<T, INNER_BLOCK_SIZE, CopyBackup<T,Allocator>,
         //         use_tasks, debug, Allocator>
         //        ::factor(
         //              nrow(), ncol(), lperm, aval_, lda_,
         //              cdata_[i_].d, inner_backup, options, options.pivot_method,
         //              INNER_BLOCK_SIZE, 0, nullptr, 0, work, alloc
         //              );

         // printf("[Block::factor] nrow = %d, ncol = %d\n", nrow(), ncol());

         cdata_[i_].nelim =
            FactorSymIndef
            <T, INNER_BLOCK_SIZE, CopyBackup<T,Allocator>, debug, Allocator>
            ::ldlt_app_notask(
                  nrow(), ncol(), lperm, aval_, lda_,
                  cdata_[i_].d, inner_backup, options, /*options.pivot_method,*/
                  INNER_BLOCK_SIZE, 0, nullptr, 0, work, alloc
                  );
         
         // printf("cdata_[i_].nelim: %d\n", cdata_[i_].nelim);

         if(cdata_[i_].nelim < 0) return cdata_[i_].nelim;
         int* temp = work.get_ptr<int>(ncol());
         int* blkperm = &perm[i_*block_size_];
         for(int i=0; i<ncol(); ++i)
            temp[i] = blkperm[lperm[i]];
         for(int i=0; i<ncol(); ++i)
            blkperm[i] = temp[i];

      } else { /* block_size == INNER_BLOCK_SIZE */
         // Call another routine for small block factorization
         if(ncol() < INNER_BLOCK_SIZE || !is_aligned(aval_)) {
            // printf("[factor] ncol = %d, is_aligned = %d\n", ncol(), is_aligned(aval_));
            // printf("[Block::factor] nrow = %d, ncol = %d\n", nrow(), ncol());
         // if(ncol() < INNER_BLOCK_SIZE || !(reinterpret_cast<uintptr_t>(aval_) % 32 == 0)) {
            T* ld = work.get_ptr<T>(2*INNER_BLOCK_SIZE);
            cdata_[i_].nelim = ldlt_tpp_factor(
                  nrow(), ncol(), lperm, aval_, lda_,
                  cdata_[i_].d, ld, INNER_BLOCK_SIZE, options.action,
                  options.u, options.small
                  );
            if(cdata_[i_].nelim < 0) return cdata_[i_].nelim;
            int* temp = work.get_ptr<int>(ncol());
            int* blkperm = &perm[i_*INNER_BLOCK_SIZE];
            for(int i=0; i<ncol(); ++i)
               temp[i] = blkperm[lperm[i]];
            for(int i=0; i<ncol(); ++i)
               blkperm[i] = temp[i];
         } else {
            int* blkperm = &perm[i_*INNER_BLOCK_SIZE];
            T* ld = work.get_ptr<T>(
                  INNER_BLOCK_SIZE*INNER_BLOCK_SIZE
                  );
            block_ldlt<T, INNER_BLOCK_SIZE>(
                  0, blkperm, aval_, lda_, cdata_[i_].d, ld, options.action,
                  options.u, options.small, lperm
                  );
            cdata_[i_].nelim = INNER_BLOCK_SIZE;
         }
      }
      return cdata_[i_].nelim;
   }

   /** \brief Apply pivots to this block and return number of pivots passing
    *         a posteori pivot test.
    *  \details If this block is below dblk, perform the operation
    *           \f[ L_{ij} = A_{ij} (D_j L_{jj})^{-T} \f]
    *           otherwise, if this block is to left of dblk, perform the
    *           operation
    *           \f[ L_{ij} = (D_i L_{ii})^{-1} A_{ij} \f]
    *           but only to uneliminated columns.
    *           After operation has completed, check a posteori pivoting
    *           condition \f$ l_{ij} < u^{-1} \f$ and return first column
    *           (block below dblk) or row (block left of dblk) in which
    *           it fails, or the total number of rows/columns otherwise.
    *  \param dblk The diagonal block to apply.
    *  \param u The pivot threshold for threshold test.
    *  \param small The drop tolerance for zero testing.
    *  \returns Number of successful pivots in this block.
    */
   int apply_pivot_app(Block const& dblk, T u, T small) {
      if(i_ == j_)
         throw std::runtime_error("apply_pivot called on diagonal block!");
      if(i_ == dblk.i_) { // Apply within row (ApplyT)
         apply_pivot<OP_T>(
               cdata_[i_].nelim, ncol(), cdata_[j_].nelim, dblk.aval_,
               cdata_[i_].d, small, aval_, lda_
               );
         return check_threshold<OP_T>(
               0, cdata_[i_].nelim, cdata_[j_].nelim, ncol(), u, aval_, lda_
               );
      } else if(j_ == dblk.j_) { // Apply within column (ApplyN)
         apply_pivot<OP_N>(
               nrow(), cdata_[j_].nelim, 0, dblk.aval_,
               cdata_[j_].d, small, aval_, lda_
               );
         return check_threshold<OP_N>(
               0, nrow(), 0, cdata_[j_].nelim, u, aval_, lda_
               );
      } else {
         throw std::runtime_error("apply_pivot called on block outside eliminated column");
      }
   }

   /** \brief Perform update of this block.
    *  \details Perform an update using the outer product of the supplied
    *           blocks:
    *           \f[ A_{ij} = A_{ij} - L_{ik} D_k L_{jk}^T \f]
    *           If this block is in the last "real" block column, optionally
    *           apply the same update to the supplied part of the contribution
    *           block that maps on to the "missing" part of this block.
    *  \param isrc The Block L_{ik}.
    *  \param jsrc The Block L_{jk}.
    *  \param work Thread-specific workspace.
    *  \param beta Global coefficient of original \f$ U_{ij} \f$ value.
    *         See form_contrib() for details.
    *  \param upd Optional pointer to \f$ U_{ij} \f$ values to be updated.
    *         If this is null, no such update is performed.
    *  \param ldupd Leading dimension of upd.
    */
   void update(Block const& isrc, Block const& jsrc, Workspace& work,
         double beta=1.0, T* upd=nullptr, int ldupd=0) {

      if(isrc.i_ == i_ && isrc.j_ == jsrc.j_) {
         
         // Update to right of elim column (UpdateN)
         int elim_col = isrc.j_;
         if(cdata_[elim_col].nelim == 0) return; // nothing to do
         int rfrom = (i_ <= elim_col) ? cdata_[i_].nelim : 0;
         int cfrom = (j_ <= elim_col) ? cdata_[j_].nelim : 0;
         int ldld = align_lda<T>(block_size_);
         T* ld = work.get_ptr<T>(block_size_*ldld);
         // NB: we use ld[rfrom] below so alignment matches that of aval[rfrom]
         calcLD<spral::ssids::cpu::OP_N>(
               nrow()-rfrom, cdata_[elim_col].nelim, &isrc.aval_[rfrom],
               lda_, cdata_[elim_col].d, &ld[rfrom], ldld
               );
         host_gemm(
               OP_N, OP_T, nrow()-rfrom, ncol()-cfrom, cdata_[elim_col].nelim,
               -1.0, &ld[rfrom], ldld, &jsrc.aval_[cfrom], lda_,
               1.0, &aval_[cfrom*lda_+rfrom], lda_
               );
         if(upd && j_==calc_nblk(n_,block_size_)-1) {
            // Handle fractional part of upd that "belongs" to this block
            int u_ncol = std::min(block_size_-ncol(), m_-n_); // ncol for upd
            beta = (cdata_[elim_col].first_elim) ? beta : 1.0; // user beta only on first update
            if(i_ == j_) {
               // diagonal block
               host_gemm(
                     OP_N, OP_T, u_ncol, u_ncol, cdata_[elim_col].nelim,
                     -1.0, &ld[ncol()], ldld,
                     &jsrc.aval_[ncol()], lda_,
                     beta, upd, ldupd
                     );
            } else {
               // off-diagonal block
               T* upd_ij =
                  &upd[(i_-calc_nblk(n_,block_size_))*block_size_+u_ncol];
               host_gemm(
                     OP_N, OP_T, nrow(), u_ncol, cdata_[elim_col].nelim,
                     -1.0, &ld[rfrom], ldld, &jsrc.aval_[ncol()], lda_,
                     beta, upd_ij, ldupd
                     );
            }
         }
      } 
      
      else {
         // Update to left of elim column (UpdateT)
         int elim_col = jsrc.i_;
         if(cdata_[elim_col].nelim == 0) return; // nothing to do
         int rfrom = (i_ <= elim_col) ? cdata_[i_].nelim : 0;
         int cfrom = (j_ <= elim_col) ? cdata_[j_].nelim : 0;
         int ldld = align_lda<T>(block_size_);
         T* ld = work.get_ptr<T>(block_size_*ldld);
         // NB: we use ld[rfrom] below so alignment matches that of aval[rfrom]
         if(isrc.j_==elim_col) {
            calcLD<spral::ssids::cpu::OP_N>(
                  nrow()-rfrom, cdata_[elim_col].nelim,
                  &isrc.aval_[rfrom], lda_,
                  cdata_[elim_col].d, &ld[rfrom], ldld
                  );
         } else {
            calcLD<spral::ssids::cpu::OP_T>(
                  nrow()-rfrom, cdata_[elim_col].nelim, &
                  isrc.aval_[rfrom*lda_], lda_,
                  cdata_[elim_col].d, &ld[rfrom], ldld
                  );
         }
         host_gemm(
               OP_N, OP_N, nrow()-rfrom, ncol()-cfrom, cdata_[elim_col].nelim,
               -1.0, &ld[rfrom], ldld, &jsrc.aval_[cfrom*lda_], lda_,
               1.0, &aval_[cfrom*lda_+rfrom], lda_
               );
      }
   }

   /** \brief Update this block as part of contribution block.
    *  \details Treat this block's coordinates as beloning to the trailing
    *           matrix (contribution block/generated elment) and perform an
    *           update using the outer product of the supplied blocks.
    *           \f[ U_{ij} = U_{ij} - L_{ik} D_k L_{jk}^T \f]
    *           If this is the first update to \f$ U_{ij} \f$, the existing
    *           values are multipled by a user-supplied coefficient
    *           \f$ \beta \f$.
    *  \param isrc the Block L_{ik}.
    *  \param jsrc the Block L_{jk}.
    *  \param work this thread's workspace.
    *  \param beta Global coefficient of original \f$ U_{ij} \f$ value.
    *  \param upd_ij pointer to \f$ U_{ij} \f$ values to be updated.
    *  \param ldupd leading dimension of upd_ij.
    */
   void form_contrib(Block const& isrc, Block const& jsrc, Workspace& work, double beta, T* upd_ij, int ldupd) {
      int elim_col = isrc.j_;
      int ldld = align_lda<T>(block_size_);
      T* ld = work.get_ptr<T>(block_size_*ldld);
      calcLD<spral::ssids::cpu::OP_N>(
            nrow(), cdata_[elim_col].nelim, isrc.aval_, lda_,
            cdata_[elim_col].d, ld, ldld
            );
      // User-supplied beta only on first update; otherwise 1.0
      T rbeta = (cdata_[elim_col].first_elim) ? beta : 1.0;
      int blkn = get_nrow(j_); // nrow not ncol as we're on contrib
      host_gemm(
            OP_N, OP_T, nrow(), blkn, cdata_[elim_col].nelim,
            -1.0, ld, ldld, jsrc.aval_, lda_,
            rbeta, upd_ij, ldupd
            );
   }

   /** \brief Returns true if block contains NaNs or Infs (debug only).
    *  \param elim_col if supplied, the block column currently being considered
    *         for elimination. Entries in that block row/column marked as
    *         failed are ignored.
    */
   bool isnan(int elim_col=-1) const {
      int m = (i_==elim_col) ? cdata_[i_].get_npass() : nrow();
      int n = (j_==elim_col) ? cdata_[j_].get_npass() : ncol();
      for(int j=0; j<n; ++j)
      for(int i=((i_==j_)?j:0); i<m; ++i) {
         if(std::isnan(aval_[j*lda_+i])) {
            printf("%d, %d is nan\n", i, j);
            return true;
         }
         if(!std::isfinite(aval_[j*lda_+i])) {
            printf("%d, %d is inf\n", i, j);
            return true;
         }
      }
      return false;
   }

   /** \brief Prints block (debug only) */
   void print() const {
      printf("Block %d, %d (%d x %d):\n", i_, j_, nrow(), ncol());
      for(int i=0; i<nrow(); ++i) {
         printf("%d:", i);
         for(int j=0; j<ncol(); ++j)
            printf(" %e", aval_[j*lda_+i]);
         printf("\n");
      }
   }

   /** \brief return number of rows in this block */
   int nrow() const { return get_nrow(i_); }
   /** \brief return number of columns in this block */
   int ncol() const { return get_ncol(j_); }
   /* return block's row */
   int get_row() const { return i_; }
   /* return block's col */
   int get_col() const { return j_; }
   /* return number of rows in the matrix */
   int get_m() const { return m_; }
   /* return number of columns in the matrix*/
   int get_n() const { return n_; }
   /* return the block size */
   int get_blksz() const { return block_size_; }
   /* return a pointer on the data */
   // T* const get_a() const { return aval_; }
   //T* get_a() const { return aval_; }
   const T* get_a() const { return aval_; }
   /// @brief return leading dimension of block
   int get_lda() const {return lda_; }

#if defined(SPLDLT_USE_STARPU)
   /* return StarPU on block */
   starpu_data_handle_t get_hdl() const { return hdl_; }

   /// @brief Register data handle in StarPU
   void register_handle() {
      
      // printf("[Block::register_handle]\n");
      
      starpu_matrix_data_register (
            &hdl_, 0, (uintptr_t) aval_, lda_, nrow(), ncol(),
            sizeof(T));
   }

   /// @brief Unregister data handle in StarPU
   template<bool async=true>
   void unregister_handle() {
      if (async) starpu_data_unregister_submit(hdl_);
      else       starpu_data_unregister(hdl_);
   }
#endif

private:
   /** \brief return number of columns in given block column */
   inline int get_ncol(int blk) const {
      return calc_blkn(blk, n_, block_size_);
   }
   /** \brief return number of rows in given block row */
   inline int get_nrow(int blk) const {
      return calc_blkn(blk, m_, block_size_);
   }

   bool is_aligned(void* ptr) {
#if defined(__AVX512F__)
      const int align = 64;
#elif defined(__AVX__)
      const int align = 32;
#else
      const int align = 16;
#endif
      // printf("[is_aligned] align = %d\n", align);
      return (reinterpret_cast<uintptr_t>(ptr) % align == 0);

      // const int align = 32;
      // return (reinterpret_cast<uintptr_t>(ptr) % align == 0);
   }

   int const i_; ///< block's row
   int const j_; ///< block's column
   int const m_; ///< number of rows in matrix
   int const n_; ///< number of columns in matrix
   int const lda_; ///< leading dimension of underlying storage
   int const block_size_; ///< block size
   ColumnData<T,IntAlloc>& cdata_; ///< global column data array
   T* aval_; ///< pointer to underlying matrix storage
#if defined(SPLDLT_USE_STARPU)
   starpu_data_handle_t hdl_;
#endif
};
 
/** \brief Grouping of assorted functions for LDL^T factorization that share
 *         template paramters.
 *  \tparam T underlying datatype, e.g. double
 *  \tparam BLOCK_SIZE inner block size for factorization, must be a multiple
 *          of vector length.
 *  \tparam Backup class to be used for handling block backups,
 *          e.g. PoolBackup or CopyBackup.
 *  \tparam use_tasks enable use of OpenMP tasks if true (used to serialise
 *          internal call for small block sizes).
 *  \tparam debug enable debug output.
 *  \tparam Allocator allocator to use for internal memory allocations
 */
// template<typename T,
//          int BLOCK_SIZE,
//          typename Backup,
//          bool use_tasks,
//          bool debug,
//          typename Allocator
//          >
// class LDLT {
//    /// \{
//    typedef typename std::allocator_traits<Allocator>::template rebind_alloc<int> IntAlloc;
//    typedef typename std::allocator_traits<Allocator>::template rebind_alloc<T> TAlloc;
//    /// \}
// private:
//    static
//    void print_mat(int m, int n, const int *perm, std::vector<bool> const& eliminated, const T *a, int lda) {
//       for(int row=0; row<m; row++) {
//          if(row < n)
//             printf("%d%s:", perm[row], eliminated[row]?"X":" ");
//          else
//             printf("%d%s:", row, "U");
//          for(int col=0; col<std::min(n,row+1); col++)
//             printf(" %10.4f", a[col*lda+row]);
//          printf("\n");
//       }
//    }

//    /** \brief return number of columns in given block column */
//    static
//    inline int get_ncol(int blk, int n, int block_size) {
//       return calc_blkn(blk, n, block_size);
//    }
//    /** \brief return number of rows in given block row */
//    static
//    inline int get_nrow(int blk, int m, int block_size) {
//       return calc_blkn(blk, m, block_size);
//    }

// public:

// };

} /* namespace spldlt::ldlt_app_internal */

} /* namespaces spldlt */
