#pragma once

#include <cassert>

namespace spldlt {

   /// @param m number of rows
   /// @param n number of columns
   template <typename T>
   void lu_pp_factor(int m, int nelim, int* perm, T *a, int lda) {

      // Only consider case where there are more rows than columns
      assert(m >= n);

      
   }
