#pragma once

// SyLVER
#include "NumericFront.hxx"
#if defined(SPLDLT_USE_GPU)
#include "kernels/gpu/common.hxx"
#include "kernels/gpu/wrappers.hxx"
#endif

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>

// SSIDS
#include "ssids/cpu/kernels/wrappers.hxx"
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/kernels/ldlt_app.hxx"

#if defined(SPLDLT_USE_GPU)
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

//using namespace spral::ssids::cpu;

namespace spldlt {
   namespace tests {

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
         T best = 0.0;
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
      
      // Update A22 -= A_21 D_11 A_21^T
      template <typename T>
      void do_update(int m, int n, int k, T *a22, const T *a21, int lda, const T* d) {
         // Form A_21 D_11
         T *ad21 = new T[m*k];
         for(int j=0; j<k;) {
            if(j+1<k && std::isinf(d[2*j+2])) {
               // 2x2 pivot
               // (Actually stored as D^-1 so need to invert it again)
               T di11 = d[2*j]; T di21 = d[2*j+1]; T di22 = d[2*j+3];
               T det = di11*di22 - di21*di21;
               T d11 = di22 / det; T d21 = -di21 / det; T d22 = di11 / det;
               // And calulate ad21
               for(int i=0; i<m; i++) {
                  ad21[j*m+i]     = d11*a21[j*lda+i] + d21*a21[(j+1)*lda+i];
                  ad21[(j+1)*m+i] = d21*a21[j*lda+i] + d22*a21[(j+1)*lda+i];
               }
               // Increment j
               j += 2;
            } else {
               // 1x1 pivot
               // (Actually stored as D^-1 so need to invert it again)
               if(d[2*j] == 0.0) {
                  // Handle zero pivots with care
                  for(int i=0; i<m; i++) {
                     ad21[j*m+i] = 0.0;
                  }
               } else {
                  // Standard 1x1 pivot
                  T d11 = 1/d[2*j];
                  // And calulate ad21
                  for(int i=0; i<m; i++) {
                     ad21[j*m+i] = d11*a21[j*lda+i];
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
         host_gemm<T>(OP_N, OP_T, m, n, k, -1.0, ad21, m, a21, lda, 1.0, a22, lda);

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

      /// @brief Add the contribution blocks into the array a
      template<typename T, typename PoolAllocator,
               bool debug=false>
      void add_cb_to_a(NumericFront<T, PoolAllocator>& node, T* a, int lda) {

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
                  for (int r = 0; r < cb.m; ++r) {
                     a[(first_col+c)*lda + (first_row+r)] += cb.a[c*cb.lda+r];
                  }
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

      template<typename T>
      void gen_mat(int m, int n, T* a, int lda) {
         /* Fill matrix with random numbers from Unif [-1.0,1.0] */
         for(int j=0; j<n; ++j)
            for(int i=0; i<m; ++i)
               a[j*lda+i] = 1.0 - (2.0*rand()) / RAND_MAX ;
      }

      // Generates a random general matrix. Off diagonal entries are
      // Unif[-1,1]. Each diagonal entry a_ii = Unif[0.1,1.1] +
      // sum_{i!=j} |a_ij|.
      /// @param m matrix order
      template<typename T>
      void gen_unsym_diagdom(int m, T* a, int lda) {
         // Generate general unsym matrix
         gen_mat(m, m, a, lda);
         // Make it diagonally dominant
         for(int i=0; i<m; ++i) a[i*lda+i] = fabs(a[i*lda+i]) + 0.1;
         for(int j=0; j<m; ++j) {
            for(int i=0; i<m; ++i) {
               if (i != j)
                  a[i*lda+i] += fabs(a[j*lda+i]);
            }
         }
      }

      template<typename T>
      void gen_unsym_diagdomblock(int m, T* a, int lda, int blksz) {

         // Generate general unsym matrix
         gen_mat(m, m, a, lda);

         // Make it diagonally dominant by block
         std::vector<int> perm;
         for (int i=0; i<blksz; ++i) perm.push_back(i);
         for (int j=0; j<m; j+=blksz) {
            random_shuffle(perm.begin(), perm.end());
            // for (int i=0; i<blksz; ++i)
            for (int k=0; k<blksz && j+k<m; ++k) {
               T s = 0.0;
               for (int i=0; i<m; ++i) s += fabs(a[j*lda+i]);
               a[(j+k)*lda+j+perm[k]] = s + 0.1;
            }
         }

      }
      
      // Generate a single rhs corresponding to solution x = 1.0
      /// @param m matrix order
      template<typename T>
      void gen_unsym_rhs(int m,  T* a, int lda, double* rhs) {
         memset(rhs, 0, m*sizeof(T));
         for (int i=0; i<m; i++) {
            for (int j=0; j<m; j++) {
               rhs[i] += a[j*lda+i] * 1.0; 
            }
         }
      }

      // Calculate scaled backward error ||Ax-b|| / ( ||A|| ||x|| + ||b|| ).
      // All norms are infinity norms execpt for matrix A which is one norm.
      template<typename T>
      double unsym_backward_error(
            int m,int n, T const* a, int lda, T const* rhs, int nrhs,
            T const* soln, int ldsoln) {
         
         int const ldrhs = m; // Assume ldrhs = m
         
         /* Allocate memory */
         double *resid = new double[m];
         double *rowsum = new double[n];

         memset(rowsum, 0, n*sizeof(double));
         for(int j=0; j<n; ++j) {
            for(int i=0; i<m; ++i) {
               rowsum[j] += static_cast<double>(a[j*lda+i]);
            }
         }         
         double anorm = 0.0;
         for(int j=0; j<n; ++j)
            anorm = std::max(anorm, rowsum[j]);

         /* Calculate residual vector and anorm */
         double worstbwderr = 0.0;
         for(int r=0; r<nrhs; ++r) {
            // memcpy(resid, &rhs[r*ldrhs], m*sizeof(T));
            for(int i=0; i<m; ++i)
               resid[i] = static_cast<double>(rhs[r*ldsoln+i]);
            
            for(int j=0; j<n; ++j) {
               for(int i=0; i<m; ++i) {
                  resid[i] -= static_cast<double>(a[j*lda+i])*static_cast<double>(soln[r*ldsoln+j]); 
               }
            }

            /* Check scaled backwards error */
            double rhsnorm=0.0, residnorm=0.0, solnnorm=0.0;
            for(int i=0; i<m; ++i) {
               // Calculate max norms
               rhsnorm = std::max(rhsnorm, fabs(static_cast<double>(rhs[r*ldrhs+i])));
               residnorm = std::max(residnorm, fabs(resid[i]));
               if(std::isnan(resid[i])) residnorm = resid[i]; 
            }

            for(int i=0; i<n; ++i) {
               solnnorm = std::max(solnnorm, fabs(static_cast<double>(soln[i+r*ldsoln])));
            }
            
            worstbwderr = std::max(worstbwderr, residnorm/(anorm*solnnorm + rhsnorm));
            if(std::isnan(residnorm)) worstbwderr = residnorm;
         }

         /* Cleanup */
         delete[] resid;
         delete[] rowsum;

         // Return largest error
         return worstbwderr;

      }

      template<typename T>
      void print_mat(char const* format, int n, T const* a, int lda,
                     int *perm=nullptr) {
         for(int i=0; i<n; ++i) {
            printf("%d:", (perm) ? perm[i] : i);
            for(int j=0; j<=i; ++j)
               printf(format, a[j*lda+i]);
            printf("\n");
         }
      }

      template<typename T>
      void print_mat_unsym(char const* format, int n, T const* a, int lda,
                     int *rperm=nullptr) {
         for(int i=0; i<n; ++i) {
            printf("%d:", (rperm) ? rperm[i] : i);
            for(int j=0; j<n; ++j)
               printf(format, a[j*lda+i]);
            printf("\n");
         }
      }

   } // namespace tests
} // namespace spldlt

namespace sylver {
namespace tests {

   // Algorithm to be tested
   enum algo {
      /// SyLVER routine
      SyLVER,
      /// SyLVER routine using half precision (compute type 32F)
      SyLVER_HP,
      /// SyLVER routine using half precision and compute type 16F
      SyLVER_HP_C16,
      /// SyLVER routine using half precision and compute type 16F
      SyLVER_HP_U32,
      /// Use routine from cuSOLVER 
      cuSOLVER,
      /// Use routine from cuSOLVER using half precision
      cuSOLVER_HP,
      /// Use the CUTLASS library
      CUTLASS,
      /// Use the CUTLASS library using half precision
      CUTLASS_WMMA_HP
   };

   // Working precision
   enum prec {
      // Halh
      FP16,
      // Single
      FP32,
      // Single
      FP64
   };

   // Generates a random dense positive definte matrix. Entries are
   // Unif[-1,1]. Only lower triangle is used, rest is filled with
   // NaNs.
   template<typename T>
   void gen_sym_indef(int n, T* a, int lda) {
      /* Fill matrix with random numbers from Unif [-1.0,1.0] */
      for(int j=0; j<n; ++j)
         for(int i=j; i<n; ++i)
            a[j*lda+i] = 1.0 - (2.0*rand()) / RAND_MAX ;
      // Fill upper triangle with NaN
      // for(int j=0; j<n; ++j)
      //    for(int i=0; i<j; ++i)
      //       a[j*lda+i] = std::numeric_limits<T>::signaling_NaN();
      // Fill upper triangle with zeros
      // for(int j=0; j<n; ++j)
      //    for(int i=0; i<j; ++i)
      //       a[j*lda+i] = 0.0;
      // Symmetrize
      for(int j=0; j<n; ++j)
         for(int i=0; i<j; ++i)
            a[j*lda+i] = a[i*lda+j];

   }

   // Generates a random dense positive definte matrix. Off
   // diagonal entries are Unif[-1,1]. Each diagonal entry a_ii =
   // Unif[0.1,1.1] + sum_{i!=j} |a_ij|. Only lower triangle is
   // used, rest is filled with NaNs.
   template<typename T>
   void gen_posdef(int n, T* a, int lda) {
      /* Get general sym indef matrix */
      gen_sym_indef(n, a, lda);
      /* Make diagonally dominant */
      for(int i=0; i<n; ++i) a[i*lda+i] = fabs(a[i*lda+i]) + 0.1;
      for(int j=0; j<n; ++j)
         for(int i=j; i<n; ++i) {
            a[j*lda+j] += fabs(a[j*lda+i]);
            a[i*lda+i] += fabs(a[j*lda+i]);
         }
   }

   /// @brief Generates a random, dense, positive-definte matrix with
   /// a condition specific condition number and eignen value
   /// distribution.
   template<typename T>
   void gen_posdef_cond(int n, T* a, int lda, T cond, T gamma) {
      // /* Get general sym indef matrix */
      // gen_sym_indef(n, a, lda);
      // /* Make diagonally dominant */
      // for(int i=0; i<n; ++i) a[i*lda+i] = fabs(a[i*lda+i]) + 0.1;
      // for(int j=0; j<n; ++j)
      //    for(int i=j; i<n; ++i) {
      //       a[j*lda+j] += fabs(a[j*lda+i]);
      //       a[i*lda+i] += fabs(a[j*lda+i]);
      //    }

      // Error handling
      std::string context = "gen_posdef_cond";
      std::cout << context << ", cond = " << cond << ", gamma = " << gamma << std::endl;
      // Timers
      std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
      
      std::default_random_engine generator;
      // std::uniform_int_distribution<long long int> distribution(1.0, std::pow(10.0,cond));
      std::uniform_real_distribution<double> distribution(std::pow(10.0, -cond), 1.0);
 
      // Generate diagonal matrix with eigen values 
      // T d = 1.0;
      // for(int i=0; i<n; ++i) a[i*lda+i] = (T)1.0;
      // for(int i=0; i<n; ++i) a[i*lda+i] = distribution(generator);
      for(int i=0; i<n; ++i) a[i*lda+i] = std::pow(10.0, (T)-cond*std::pow( ((T)i)/((T)n-1), gamma ) );
      // for(int i=0; i<n; ++i) a[i*lda+i] = (T)1.0 - ((T)i/(T)(n - 1))*(1.0 - std::pow(10.0,-cond));
      for(int j=0; j<n; ++j) {
         for(int i=0; i<n; ++i) {
            if (i != j)
               a[j*lda+i] = 0.0;
         }
      }
      // a[0] = 1e-3;
      // a[0] = (T)1.0/std::pow(10.0,cond);
      // a[(n-1)*lda+(n-1)] = (T)1.0/std::pow(10.0,cond);
      // std::cout << "ann = " << a[(n-1)*lda+(n-1)] << std::endl;
      
      T *lambda = new T[lda*n];
      // Fill up lambda with random values
      ::spldlt::tests::gen_mat(n, n, lambda, lda);

#if defined(SPLDLT_USE_GPU)
      cudaError_t cuerr; // CUDA error
      cublasStatus_t custat; // cuBLAS status 
      T *d_lambda = nullptr;
      cuerr = cudaMalloc((void**)&d_lambda, n*lda*sizeof(T));      
      sylver::gpu::cuda_check_error(cuerr, context);
      // Send lambda to the GPU
      custat = cublasSetMatrix(n, n, sizeof(T), lambda, lda, d_lambda, lda);
      sylver::gpu::cublas_check_error(custat, context);
      T *d_a = nullptr;
      cuerr = cudaMalloc((void**)&d_a, n*lda*sizeof(T));      
      sylver::gpu::cuda_check_error(cuerr, context);
      // Send A to the GPU
      custat = cublasSetMatrix(n, n, sizeof(T), a, lda, d_a, lda);
      sylver::gpu::cublas_check_error(custat, context);

      cusolverStatus_t cusolstat;
      cusolverDnHandle_t cusolhandle;
      cusolstat = cusolverDnCreate(&cusolhandle);
      int lwork; // Workspace dimensions
      sylver::gpu::dev_geqrf_buffersize(cusolhandle, n, n, d_lambda, lda, &lwork);
      std::cout << "[" << context << "]" <<" dev_geqrf lwork = " << lwork << std::endl;
      // Workspace on the device
      T *d_work;
      cuerr = cudaMalloc((void**)&d_work, lwork*sizeof(T));      
      sylver::gpu::cuda_check_error(cuerr, context);
      T *d_tau;
      cuerr = cudaMalloc((void**)&d_tau, n*sizeof(T));      
      sylver::gpu::cuda_check_error(cuerr, context);
      // Allocate info paramater on device
      int *dinfo;
      cuerr = cudaMalloc((void**)&dinfo, sizeof(int));      
      sylver::gpu::cuda_check_error(cuerr, context);

      start = std::chrono::high_resolution_clock::now();
      sylver::gpu::dev_geqrf(
            cusolhandle, n, n, d_lambda, lda,
            d_tau, d_work, lwork,
            dinfo);

      // A = Q * D
      sylver::gpu::dev_ormqr(
            cusolhandle,
            CUBLAS_SIDE_LEFT, CUBLAS_OP_N,
            n, n, n,
            d_lambda, lda, d_tau,
            d_a, lda,
            d_work, lwork,
            dinfo);

      // A = A * Q^T
      sylver::gpu::dev_ormqr(
            cusolhandle,
            CUBLAS_SIDE_RIGHT, CUBLAS_OP_T,
            n, n, n,
            d_lambda, lda, d_tau,
            d_a, lda,
            d_work, lwork,
            dinfo);
      end = std::chrono::high_resolution_clock::now();

      // Retrieve A on the host
      custat = cublasGetMatrix(n, n, sizeof(T), d_a, lda, a, lda);
      sylver::gpu::cublas_check_error(custat, context);

      // Cleanup
      cusolstat = cusolverDnDestroy(cusolhandle);

      cuerr = cudaFree(dinfo);      
      cuerr = cudaFree(d_tau);      
      cuerr = cudaFree(d_work);      
      cuerr = cudaFree(d_a);      
      cuerr = cudaFree(d_lambda);      

#else
      
      T *tau = new T[n];
      T worksz;
      sylver::host_geqrf(n, n,
                         lambda, lda,
                         tau,
                         &worksz, -1);
      // std::cout << "geqrf worksz = " << worksz << std::endl;

      int lwork = (int)worksz;
      T *work = new T[lwork];

      sylver::host_geqrf(n, n,
                         lambda, lda,
                         tau,
                         work, lwork);

      // delete [] work;
      
      // sylver::host_ormqr(
      //       sylver::SIDE_LEFT, sylver::OP_N,
      //       n, n, n,
      //       lambda, lda, tau,
      //       a, lda,
      //       &worksz, -1);

      // std::cout << "ormqr worksz = " << worksz << std::endl;

      // lwork = (int)worksz;
      // work = new T[lwork];
      
      // // A = Q * D
      // sylver::host_ormqr(
      //       sylver::SIDE_LEFT, sylver::OP_N,
      //       n, n, n,
      //       lambda, lda, tau,
      //       a, lda,
      //       work, lwork);

      // // A = A * Q^T
      // sylver::host_ormqr(
      //       sylver::SIDE_RIGHT, sylver::OP_T,
      //       n, n, n,
      //       lambda, lda, tau,
      //       a, lda,
      //       work, lwork);

      delete[] work;
      delete[] tau;
#endif
      
      delete[] lambda;

      // Calculate walltime
      long ttotal =  
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (end-start).count();

      std::cout << "[" << context << "]" <<" matrix generation time (s) = " << ttotal*1e-9 << std::endl;

   }

   // Generate one or more right-hand sides corresponding to soln x =
   // 1.0.
   template<typename T>
   void gen_rhs(int n, T* a, int lda, T* rhs) {
      memset(rhs, 0, n*sizeof(T));
      for(int j=0; j<n; ++j) {
         rhs[j] += a[j*lda+j] * 1.0;
         for(int i=j+1; i<n; ++i) {
            rhs[j] += a[j*lda+i] * 1.0;
            rhs[i] += a[j*lda+i] * 1.0;
         }
      }
   }

   // Calculates forward error ||soln-x||_inf assuming x=1.0
   template<typename T>
   T forward_error(int n, int nrhs, T const* soln, int ldx) {
      /* Check scaled backwards error */
      T fwderr=0.0;
      for(int r=0; r<nrhs; ++r)
         for(int i=0; i<n; ++i) {
            T diff = std::fabs(soln[r*ldx+i] - 1.0);
            fwderr = std::max(fwderr, diff);
         }
      return fwderr;
   }

   // Calculate scaled backward error ||Ax-b|| / ( ||A|| ||x|| + ||b||
   // ). All norms are infinity norms.
   template<typename T>
   double backward_error(int n, T const* a, int lda, T const* rhs, int nrhs, T const* soln, int ldsoln) {
      /* Allocate memory */
      double *resid = new double[n];
      double *rowsum = new double[n];

      /* Calculate residual vector and anorm*/
      double worstbwderr = 0.0;
      for(int r=0; r<nrhs; ++r) {
         // memcpy(resid, rhs, n*sizeof(T));
         for (int j=0; j<n; ++j)
            resid[j] = (double)rhs[r*ldsoln+j];
         memset(rowsum, 0, n*sizeof(double));
         for(int j=0; j<n; ++j) {
            resid[j] -= (double)a[j*lda+j] * soln[r*ldsoln+j];
            rowsum[j] += fabs((double)a[j*lda+j]);
            for(int i=j+1; i<n; ++i) {
               resid[j] -= (double)a[j*lda+i] * soln[r*ldsoln+i];
               resid[i] -= (double)a[j*lda+i] * soln[r*ldsoln+j];
               rowsum[j] += fabs((double)a[j*lda+i]);
               rowsum[i] += fabs((double)a[j*lda+i]);
            }
         }
         double anorm = 0.0;
         for(int i=0; i<n; ++i)
            anorm = std::max(anorm, rowsum[i]);

         /* Check scaled backwards error */
         double rhsnorm=0.0, residnorm=0.0, solnnorm=0.0;
         for(int i=0; i<n; ++i) {
            rhsnorm = std::max(rhsnorm, fabs((double)rhs[i]));
            residnorm = std::max(residnorm, fabs(resid[i]));
            if(std::isnan(resid[i])) residnorm = resid[i]; 
            solnnorm = std::max(solnnorm, fabs((double)soln[r*ldsoln+i]));
         }

         //printf("%e / %e %e %e\n", residnorm, anorm, solnnorm, rhsnorm);
         worstbwderr = std::max(worstbwderr, residnorm/(anorm*solnnorm + rhsnorm));
         if(std::isnan(residnorm)) worstbwderr = residnorm;
      }

      /* Cleanup */
      delete[] resid;
      delete[] rowsum;

      /* Return result */
      //printf("worstbwderr = %e\n", worstbwderr);
      return worstbwderr;
   }

} // End of namespace tests
} // End of namespace sylver
