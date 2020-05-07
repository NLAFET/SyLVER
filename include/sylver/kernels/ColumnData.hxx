/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Jonathan Hogg
/// @author Florent Lopez
#pragma once

#include "sylver/kernels/common.hxx"

namespace sylver {

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
   sylver::Column<T>& operator[](int idx) { return cdata_[idx]; }

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

private:
   int const n_; ///< number of columns in matrix
   int const block_size_; ///< block size for matrix
   IntAlloc alloc_; ///< internal copy of allocator to be used in destructor
   sylver::Column<T> *cdata_; ///< underlying array of columns
   int* lperm_; ///< underlying local permutation
#if defined(SPLDLT_USE_STARPU)
   starpu_data_handle_t d_hdl_;
#endif
};

}
