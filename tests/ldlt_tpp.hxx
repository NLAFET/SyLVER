#pragma once

#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include "ssids/cpu/kernels/wrappers.hxx"

using namespace spral::ssids::cpu;

namespace spldlt {

   int run_ldlt_tpp_tests();

   template<typename T>
   void print_d(int n, T *d) {
      bool first = true;
      for(int i=0; i<n; i++) {
         if(d[2*i]==std::numeric_limits<T>::infinity() && d[2*i+1]!=0.0) {
            // Second half of 2x2 pivot: don't print brackets
            std::cout << " ";
         } else if(first) {
            std::cout << "(";
         } else {
            std::cout << ") (";
         }
         if(d[2*i] == std::numeric_limits<T>::infinity() && d[2*i+1]!=0.0) std::cout << std::setw(8) << " ";
         else              std::cout << std::setw(8) << d[2*i];
         first = false;
      }
      std::cout << ")" << std::endl;
      first = true;
      for(int i=0; i<n; i++) {
         if(d[2*i]==std::numeric_limits<T>::infinity() && d[2*i+1]!=0.0) {
            // Second half of 2x2 pivot: don't print brackets
            std::cout << " ";
         } else if(first) {
            std::cout << "(";
         } else {
            std::cout << ") (";
         }
         if(d[2*i+1] == 0.0) std::cout << std::setw(8) << " ";
         else                std::cout << std::setw(8) << d[2*i+1];
         first = false;
      }
      std::cout << ")" << std::endl;
   }

   template <typename T>
   void print_mat (int n, int *perm, T *a, int lda) {
      for(int i=0; i<n; i++) {
         printf("%d:", perm[i]);
         for(int j=0; j<n; j++)
            printf(" %10.3le", a[j*lda+i]);
         printf("\n");
      }
   }

   /// Update A22 -= A_21 D_11 A_21^T
   template <typename T>
   void do_update(int n, int k, T *a22, const T *a21, int lda, const T* d) {
      // Form A_21 D_11
      T *ad21 = new T[n*k];
      for(int j=0; j<k;) {
         if(j+1<k && std::isinf(d[2*j+2])) {
            // 2x2 pivot
            // (Actually stored as D^-1 so need to invert it again)
            T di11 = d[2*j]; T di21 = d[2*j+1]; T di22 = d[2*j+3];
            T det = di11*di22 - di21*di21;
            T d11 = di22 / det; T d21 = -di21 / det; T d22 = di11 / det;
            // And calulate ad21
            for(int i=0; i<n; i++) {
               ad21[j*n+i]     = d11*a21[j*lda+i] + d21*a21[(j+1)*lda+i];
               ad21[(j+1)*n+i] = d21*a21[j*lda+i] + d22*a21[(j+1)*lda+i];
            }
            // Increment j
            j += 2;
         } else {
            // 1x1 pivot
            // (Actually stored as D^-1 so need to invert it again)
            if(d[2*j] == 0.0) {
               // Handle zero pivots with care
               for(int i=0; i<n; i++) {
                  ad21[j*n+i] = 0.0;
               }
            } else {
               // Standard 1x1 pivot
               T d11 = 1/d[2*j];
               // And calulate ad21
               for(int i=0; i<n; i++) {
                  ad21[j*n+i] = d11*a21[j*lda+i];
               }
            }
            // Increment j
            j++;
         }
      }

      /*printf("a21:\n");
        for(int i=0; i<n; i++) {
        for(int j=0; j<k; j++)
        printf(" %le", a21[j*lda+i]);
        printf("\n");
        }
        printf("ad21:\n");
        for(int i=0; i<n; i++) {
        for(int j=0; j<k; j++)
        printf(" %le", ad21[j*n+i]);
        printf("\n");
        }
        printf("a22:\n");
        for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++)
        printf(" %le", a22[j*lda+i]);
        printf("\n");
        }*/

      // Perform actual update
      host_gemm<T>(OP_N, OP_T, n, n, k, -1.0, ad21, n, a21, lda, 1.0, a22, lda);

      // Free memory
      delete[] ad21;
   }

}
