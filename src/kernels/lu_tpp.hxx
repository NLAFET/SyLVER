#pragma once

namespace spldlt {

   /// @brief Simple LU with threshold partial pivoting.
   /// @param m number of rows
   /// @param k number of rows/columns to be eliminated
   /// @param u threshold parameter
   template <typename T>
   void lu_tpp_factor(int m, int k, int* perm, T *a, int lda, double u) {
      
      assert(m >= nelim);
      
      int nelim = 0; // Number of eliminated variables
      
      // TODO: manage zero pivots
      

   }

} // end of namespace spldlt
