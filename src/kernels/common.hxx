#pragma once

namespace spldlt {

   /// \brief bub::fillmode enumerates which part of the matrix is specified.
   enum fillmode {
      /// The lower triangular part of the matrix is specified
      FILL_MODE_LWR,
      /// The upper triangular part of the matrix is specified
      FILL_MODE_UPR
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
