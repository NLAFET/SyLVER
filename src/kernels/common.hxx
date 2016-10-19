#pragma once

namespace spldlt {

   template<typename T>
   void print_mat(int m, int n, const T *a, int lda) {
      for(int row=0; row<m; row++) {
         for(int col=0; col<std::min(n,row+1); col++)
            printf(" %10.4f", a[col*lda+row]);
         printf("\n");
      }
   }
}
