/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Jonathan Hogg
/// @author Florent Lopez
#pragma once

// STD
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

#if defined(SPLDLT_USE_STARPU)
#include <starpu.h>
#endif

namespace sylver {

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
   
}
