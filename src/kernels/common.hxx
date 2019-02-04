/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "sylver_ciface.hxx"
// STD
#include <algorithm>
#include <cstdio>
#include <cassert>
#include <iostream>

namespace sylver {

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

   /// @brief find index of max abs value in a column starting at a
   template <typename T>
   int find_col_abs_max(int from, int to, T const* a) {
      
      assert(from>=0);
      assert(to>=from);

      T maxval = fabs(a[from]); 
      int maxidx = from;
      
      // printf("[find_col_abs_max] from = %d, init maxval = %e\n", from, maxval);

      for (int idx=from; idx <= to; ++idx) {
         // printf("[find_col_abs_max] a = %.2f\n", a[idx]);
         if (fabs(a[idx]) > maxval) {
            // printf("[find_col_abs_max] fabs(a[idx]) = %e, maxval = %e\n", fabs(a[idx]), maxval);
            maxval = fabs(a[idx]);
            maxidx = idx;
         }
      }

      return maxidx;
   }

   /// @brief find index of max abs value in a column starting at a
   template <typename T>
   void find_col_abs_max(int from, int to, T const* a, int& maxidx, T& maxval) {
      
      assert(from>=0);
      assert(to>=from);

      maxidx = from;
      maxval = fabs(a[from]); 
      
      // printf("[find_col_abs_max] from = %d, init maxval = %e\n", from, maxval);

      for (int idx=from; idx <= to; ++idx) {
         if (fabs(a[idx]) > maxval) {
            // printf("[find_col_abs_max] fabs(a[idx]) = %e, maxval = %e\n", fabs(a[idx]), maxval);
            maxval = fabs(a[idx]);
            maxidx = idx;
         }
      }
   }

   /// @brief Copy block a into out
   /// @param m Height of block to be copied
   /// @param n Width of block to be copied
   template <typename T>
   void copy_2d(int m, int n, T const* a, int lda, T *out, int ldout) {
      for (int j=0; j < n; ++j) {
         for (int i=0; i < m; ++i) {
            out[j*ldout+i] = a[j*lda+i];
         }
      }
   }

   ////////////////////////////////////////////////////////////

   // @brief Check SyLVER error and exit if error is detected
   inline void sylver_check_error(
         int err, std::string fname, 
         std::string const& msg = std::string()) {
      if (err != sylver::Flag::SUCCESS) {
         std::cout << "[" << fname << "][SyLVER error] "
                   << msg
                   << " (" << err << ")" << std::endl;
         std::exit(1);
      }
   }

   
} // End of namespace sylver
