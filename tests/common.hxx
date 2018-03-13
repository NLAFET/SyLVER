#pragma once

// SpLDLT
#include "NumericFront.hxx"

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <iostream>

// SSIDS
#include "ssids/cpu/kernels/wrappers.hxx"
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/kernels/ldlt_app.hxx"

//using namespace spral::ssids::cpu;

namespace spldlt { namespace tests {

    //using namespace spldlt::ldlt_app_internal;

   /// @brief Sovle Ax = b whre A = PLDL^{T}P^{T}
   template<typename T>
   void solve(int m, int n, const int *perm, const T *l, int ldl, const T *d, const T *b, T *x) {
      for(int i=0; i<m; i++) x[i] = b[perm[i]];
      // Fwd slv
      // Lz = P^{T}b
      spral::ssids::cpu::ldlt_app_solve_fwd(m, n, l, ldl, 1, x, m);
      spral::ssids::cpu::ldlt_app_solve_fwd(m-n, m-n, &l[n*(ldl+1)], ldl, 1, &x[n], m);
      // Diag slv
      // Dy = z
      spral::ssids::cpu::ldlt_app_solve_diag(n, d, 1, x, m);
      spral::ssids::cpu::ldlt_app_solve_diag(m-n, &d[2*n], 1, &x[n], m);
      // Bwd slv
      // L^{T}P^{T}x = y
      spral::ssids::cpu::ldlt_app_solve_bwd(m-n, m-n, &l[n*(ldl+1)], ldl, 1, &x[n], m);
      spral::ssids::cpu::ldlt_app_solve_bwd(m, n, l, ldl, 1, x, m);
      // Undo permutation
      T *temp = new T[m];
      for(int i=0; i<m; i++) temp[i] = x[i];
      for(int i=0; i<m; i++) x[perm[i]] = temp[i];
      // Free mem
      delete[] temp;
   }

   // Permutes rows of a as per perm such that row i becomes row (perm[i]-offset)
   template<typename T>
   void permute_rows(int n, int k, const int *reord, int *perm, T *a, int lda) {
      // Copy a and perm without permutation
      T *a2 = new T[n*k];
      for(int j=0; j<k; j++)
         for(int i=0; i<n; i++)
            a2[j*n+i] = a[j*lda+i];
      int *perm2 = new int[n];
      for(int i=0; i<n; i++)
         perm2[i] = perm[i];

      // Copy back with permutation
      for(int j=0; j<k; j++)
         for(int i=0; i<n; i++) {
            int row = reord[i];
            a[j*lda+i] = a2[j*n+row];
         }
      for(int i=0; i<n; i++) {
         int row = reord[i];
         perm[i] = perm2[row];
      }

      // Free memory
      delete[] a2;
      delete[] perm2;
   }

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
   T find_l_abs_max(int n, T *a, int lda) {
      double best = 0.0;
      for(int c=0; c<n; ++c)
         for(int r=c; r<n; ++r)
            best = std::max(best, std::abs(a[c*lda+r]));
      return best;
   }

   // Makes a (symmetric, half storage) matrix singular by making col2 an
   // appropriate multiple of col1
   template <typename T>
   void make_singular(int n, int col1, int col2, T *a, int lda) {
      T *col = new T[n];
   
      T a11 = a[col1*(lda+1)];
      T a21 = (col1 < col2) ? a[col1*lda + col2]
         : a[col2*lda + col1];
      T scal = a21 / a11;

      // Read col1 and double it
      for(int i=0; i<col1; i++)
         col[i] = scal*a[i*lda+col1];
      for(int i=col1; i<n; i++)
         col[i] = scal*a[col1*lda+i];

      // Store col to col2
      for(int i=0; i<col2; i++)
         a[i*lda+col2] = col[i];
      for(int i=col2; i<n; i++)
         a[col2*lda+i] = col[i];

      // Free mem
      delete[] col;
   }

   // Pick n/8 random rows and multiply by 1000. Then do the same for n/8 random entries.
   template <typename T>
   void cause_delays(int n, T *a, int lda) {
      int nsing = n/8;
      if(nsing==0) nsing=1;
      for(int i=0; i<nsing; i++) {
         // Add a row of oversized values
         int idx = n*((float) rand())/RAND_MAX;
         for(int c=0; c<idx; c++)
            a[c*lda+idx] *= 1000;
         for(int r=idx; r<n; r++)
            a[idx*lda+r] *= 1000;
         int row = n*((float) rand())/RAND_MAX;
         int col = n*((float) rand())/RAND_MAX;
         if(row > col) a[col*lda+row] *= 1000;
         else          a[row*lda+col] *= 1000;
      }
   }

   // Pick n/8 random rows and multiply by 1000. Then do the same for n/8 random entries.
   template <typename T, int BLOCK_SIZE>
   void cause_delays(int n, T *a, int lda) {
      // printf("blocksize: %d\n", BLOCK_SIZE);
      int nsing = n/8;
      if(nsing==0) nsing=1;
      for(int i=0; i<nsing; i++) {
         // Add a row of oversized values
         int idx = n*((float) rand())/RAND_MAX;
         if(i==0 && n>BLOCK_SIZE && idx<BLOCK_SIZE) idx += BLOCK_SIZE;
         idx = std::min(idx, n);
         for(int c=0; c<idx; c++)
            a[c*lda+idx] *= 1000;
         for(int r=idx; r<n; r++)
            a[idx*lda+r] *= 1000;
         int row = n*((float) rand())/RAND_MAX;
         int col = n*((float) rand())/RAND_MAX;
         if(row > col) a[col*lda+row] *= 1000;
         else          a[row*lda+col] *= 1000;
      }
   }

   /* Modify test matrix:
      singular: make the matrix singular
      delays: cause delays during factorization
    */
   template <typename T>
   void modify_test_matrix(bool singular, bool delays, int m, int n, T *a, int lda) {
      if(delays)
         cause_delays(m, a, lda);
      if(singular && n!=1) {
         int col1 = n * ((float) rand())/RAND_MAX;
         int col2 = col1;
         while(col1 == col2)
            col2 = n * ((float) rand())/RAND_MAX;
         make_singular(m, col1, col2, a, lda);
      }
   }

   // Makes a specified diagonal block of a matrix singular by making first and
   //  last columns linearlly dependent
   template <typename T, 
             int BLOCK_SIZE>
   void make_dblk_singular(int blk, int nblk, T *a, int lda) {
      int col1 = 0;
      int col2 = BLOCK_SIZE-1;
      T *adiag = &a[blk*BLOCK_SIZE*(lda+1)];
      make_singular(BLOCK_SIZE, col1, col2, adiag, lda);
   }

   /* Modify test matrix:
      singular: make the matrix singular
      delays: cause delays during factorization
    */
   template <typename T,
             int BLOCK_SIZE
             >
   void modify_test_matrix(bool singular, bool delays, bool dblk_singular, int m, int n, T *a, int lda) {
      int mblk = m / BLOCK_SIZE;
      int nblk = n / BLOCK_SIZE;
      if(delays)
         cause_delays<T,BLOCK_SIZE>(m, a, lda);
      if(dblk_singular) {
         int blk = nblk * ((float) rand())/RAND_MAX;
         make_dblk_singular<T, BLOCK_SIZE>(blk, mblk, a, lda);
      }
      if(n>1 && singular) {
         int col1 = n * ((float) rand())/RAND_MAX;
         int col2 = col1;
         while(col1 == col2)
            col2 = n * ((float) rand())/RAND_MAX;
         make_singular<T>(m, col1, col2, a, lda);
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

   /// @brief Copy the array a into the contribution blocks
   template<typename T, typename PoolAllocator,
            bool debug=false>
   void copy_a_to_cb(T* a, int lda, NumericFront<T, PoolAllocator>& node) {

      int m = node.get_nrow();
      int n = node.get_ncol();
      size_t contrib_dimn = m-n; // Dimension of contribution block
      int blksz =  node.blksz;

      if (contrib_dimn <= 0) return;
      if(debug) printf("[copy_front_cb] contrib dimn = %zu\n", contrib_dimn);

      int nr = node.get_nr();
      int rsa = n/blksz;
      int ncontrib = nr-rsa;

      for(int j = rsa; j < nr; j++) {
         // First col in contrib block
         int first_col = std::max(j*blksz, n);
         // Tile width
         // int blkn = std::min((j+1)*blksz, m) - first_col;
         for(int i = j; i < nr; i++) {

            spldlt::Tile<T, PoolAllocator>& cb = node.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib];
            
            // First col in contrib block
            int first_row = std::max(i*blksz, n);
            // Tile height
            // int blkm = std::min((i+1)*blksz, m) - first_row;
             
            // FIXME: use copy routine from BLAS
            for (int c = 0; c < cb.n; ++c) {
               memcpy(
                     &cb.a[c*cb.lda],
                     &a[(first_col+c)*lda + first_row],
                     cb.m*sizeof(T));

            }
         }
      }

   }

   /// @brief Copy the contribution blocks into the array a
   template<typename T, typename PoolAllocator,
            bool debug=false>
   void copy_cb_to_a(NumericFront<T, PoolAllocator>& node, T* a, int lda) {

      int m = node.get_nrow();
      int n = node.get_ncol();
      size_t contrib_dimn = m-n; // Dimension of contribution block
      int blksz =  node.blksz;
      if (contrib_dimn <= 0) return;

      int nr = node.get_nr();
      int rsa = n/blksz;
      int ncontrib = nr-rsa;

      for(int j = rsa; j < nr; j++) {
         // First col in contrib block
         int first_col = std::max(j*blksz, n);
         // Tile width
         // int blkn = std::min((j+1)*blksz, m) - first_col;
         for(int i = j; i < nr; i++) {
            // First col in contrib block
            int first_row = std::max(i*blksz, n);
            // Tile height
            // int blkm = std::min((i+1)*blksz, m) - first_row;
           
            spldlt::Tile<T, PoolAllocator>& cb = node.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib];

            // FIXME: use copy routine from BLAS
            for (int c = 0; c < cb.n; ++c) {
               memcpy(
                     &a[(first_col+c)*lda + first_row],
                     &cb.a[c*cb.lda],
                     cb.m*sizeof(T));

            }
         }
      }

   }

   /// @brief Print node's contribution blocks
   template<typename T, typename PoolAllocator>
   void print_cb(char const* format, NumericFront<T, PoolAllocator>& node) {

      int m = node.get_nrow();
      int n = node.get_ncol();
      size_t contrib_dimn = m-n; // Dimension of contribution block
      int blksz =  node.blksz;
      if (contrib_dimn <= 0) return;

      int nr = node.get_nr();
      int rsa = n/blksz;
      int ncontrib = nr-rsa;
      
      for (int r = n; r < m; ++r) {

         int iblk = (r/blksz)-rsa;
         // First row in contrib block
         int first_row = std::max(iblk*blksz, n);
         
         for (int c = n; c <= r; ++c) {

            int jblk = (c/blksz)-rsa;
            // First col in contrib block
            int first_col = std::max(jblk*blksz, n);

            T *a = node.contrib_blocks[iblk+jblk*ncontrib].a;
            int lda = node.contrib_blocks[iblk+jblk*ncontrib].lda;
            printf(format, a[(c-first_col)*lda+(r-first_row)]);            
         }
         printf("\n");

      }
      
   }
   
   }} // namespace spldlt::tests
