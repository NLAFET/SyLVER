#pragma once

#include <algorithm>
#include <cstdio>

namespace spldlt {

   /// @brief spldlt::operation enumerates operations that can be applied
   /// to a matrix * argument of a BLAS call.
   enum operation {
      /// No operation (i.e. non-transpose). Equivalent to BLAS op='N'.
      OP_N,
      /// Transposed. Equivalent to BLAS op='T'.
      OP_T
   };

   /// @brief spldlt::diagonal enumerates nature of matrix diagonal.
   enum diagonal {
      /// All diagonal elements are assumed to be identically 1.0
      DIAG_UNIT,
      /// Diagonal elements are specified in matrix data
      DIAG_NON_UNIT
   };
   
   /// @brief spldlt::fillmode enumerates which part of the matrix is
   /// specified.
   enum fillmode {
      /// The lower triangular part of the matrix is specified
      FILL_MODE_LWR,
      /// The upper triangular part of the matrix is specified
      FILL_MODE_UPR
   };

   // @brief bub::side enumerates whether the primary operand is
   //  applied on the left or right of a secondary operand
   enum side {
      /// Primary operand applied on left of secondary
      SIDE_LEFT,
      /// Primary operand applied on right of secondary
      SIDE_RIGHT
   };

   template<typename T>
   void print_mat(int m, int n, const T *a, int lda) {
      for(int row=0; row<m; row++) {
         for(int col=0; col<std::min(n,row+1); col++)
            printf(" %10.4f", a[col*lda+row]);
         printf("\n");
      }
   }

   
}
