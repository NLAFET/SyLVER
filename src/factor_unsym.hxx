#pragma once

// SyVLER
#include "NumericFront.hxx"
#include "tasks/tasks_unsym.hxx"

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/Workspace.hxx"

namespace spldlt {

   template <typename T, typename PoolAlloc>
   void factor_front_unsym_app(
         struct cpu_factor_options& options,
         NumericFront<T, PoolAlloc> &node,
         std::vector<spral::ssids::cpu::Workspace>& workspaces) {

      // typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> IntAlloc;
      typedef typename NumericFront<T, PoolAlloc>::IntAlloc IntAlloc;

      int m = node.get_nrow(); // Frontal matrix order
      int n = node.get_ncol(); // Number of fully-summed rows/columns 
      int nr = node.get_nr(); // number of block rows
      int nc = node.get_nc(); // number of fully-summed block columns
      size_t contrib_dimn = m-n;
      int blksz = node.blksz;
      ColumnData<T, IntAlloc>& cdata = *node.cdata; // Column data

      T u = options.u; // Threshold parameter
      node.nelim = 0; //
      
      for (int k = 0; k < nc; ++k) {
         
         BlockUnsym<T>& dblk = node.get_block_unsym(k, k);
         int *rperm = &node.perm [k*blksz];
         int *cperm = &node.cperm[k*blksz];
         factor_block_unsym_app_task(dblk, rperm, cperm, cdata);

         // Compute L factor
         for (int i = 0; i < k; ++i) {
            // Super-diagonal block
            BlockUnsym<T>& lblk = node.get_block_unsym(i, k);
            appyU_block_app_task(dblk, u, lblk, cdata);
         }
         for (int i = k+1; i < nr; ++i) {
            // Sub-diagonal block
            BlockUnsym<T>& lblk = node.get_block_unsym(i, k);
            appyU_block_app_task(dblk, u, lblk, cdata);
         }

         adjust_unsym_app_task(k, cdata, node.nelim); // Update nelim in node
         
         // Restore failed entries
         for (int i = 0; i < nr; ++i) {
            BlockUnsym<T>& blk = node.get_block_unsym(i, k);
            restore_block_unsym_app_task(k, blk, cdata);
         }

         // Compute U factor
         for (int j = 0; j < k; ++j) {
            // Left-diagonal block
            BlockUnsym<T>& ublk = node.get_block_unsym(k, j);
            appyL_block_app_task(dblk, ublk, cdata, workspaces);
         }
         for (int j = k+1; j < nr; ++j) {
            // Right-diagonal block
            BlockUnsym<T>& ublk = node.get_block_unsym(k, j);
            appyL_block_app_task(dblk, ublk, cdata, workspaces);
         }
         
         // Update previously failed entries
         for (int j = 0; j < k; ++j) {
            BlockUnsym<T>& ublk = node.get_block_unsym(k, j);
            for (int i = 0; i < k; ++i) {
               BlockUnsym<T>& lblk = node.get_block_unsym(i, k);
               BlockUnsym<T>& blk = node.get_block_unsym(i, j);
               update_block_unsym_app_task(lblk, ublk, blk, cdata);
            }
         }
         
         // Update uneliminated entries in L
         for (int j = 0; j < k; ++j) {
            BlockUnsym<T>& ublk = node.get_block_unsym(k, j);
            for (int i = k+1; i < nr; ++i) {
               BlockUnsym<T>& lblk = node.get_block_unsym(i, k);
               BlockUnsym<T>& blk = node.get_block_unsym(i, j);
               update_block_unsym_app_task(lblk, ublk, blk, cdata);
            }
         }

         // Update uneliminated entries in U
         for (int j = k+1; j < nr; ++j) {
            BlockUnsym<T>& ublk = node.get_block_unsym(k, j);
            for (int i = 0; i < k; ++i) {
               BlockUnsym<T>& lblk = node.get_block_unsym(i, k);
               BlockUnsym<T>& blk = node.get_block_unsym(i, j);
               update_block_unsym_app_task(lblk, ublk, blk, cdata);
            }       
         }
      
         // Udpdate trailing submatrix
         int en = (n-1)/blksz; // Last block-row/column in factors
         for (int j = k+1; j < nr; ++j) {
            
            BlockUnsym<T>& ublk = node.get_block_unsym(k, j);

            for (int i =  k+1; i < nr; ++i) {
               // Loop if we are in the cb
               if ((i > en) && (j > en)) continue;

               BlockUnsym<T>& lblk = node.get_block_unsym(i, k);
               BlockUnsym<T>& blk = node.get_block_unsym(i, j);
               update_block_unsym_app_task(lblk, ublk, blk, cdata);
            }
         }

         // Update contribution blocks
         if (contrib_dimn>0) {
            // Update contribution block
            int rsa = n/blksz; // Last block-row/column in factors
         
            for (int j = rsa; j < nr; ++j) {

               BlockUnsym<T>& ublk = node.get_block_unsym(k, j);

               for (int i = rsa; i < nr; ++i) {

                  BlockUnsym<T>& lblk = node.get_block_unsym(i, k);
                  Tile<T, PoolAlloc>& cblk = node.get_contrib_block(i, j);
                  
                  update_cb_block_unsym_app_task(lblk, ublk, cblk, cdata);               
               }
            }
         }

      }

      printf("[factor_front_unsym_app] first pass front.nelim = %d\n", node.nelim);
      // Permute failed entries at the back of the marix
      // permute_failed_unsym(node);
      
   }

   template <typename T>
   void copy_failed_diag_unsym(
         int m, int n, int rfrom, int cfrom, 
         T const* a, int lda, T *out, int ldout) {

      if(out == a) return; // don't bother moving if memory is the same

      for (int j = cfrom, jout = 0; j < n; ++j, ++jout) {
         for (int i = rfrom, iout = 0; i < n; ++i, ++iout) {
            out[jout*ldout+iout] = a[j*lda+i];
         }
      } 

   }

   template <typename T>
   void move_up_diag_unsym(
         int m, int n, T const* a, int lda, T *out, int ldout) {      
      
      if(out == a) return; // don't bother moving if memory is the same
   
      for (int j = 0; j < n; ++j) {
         for (int i = 0; i < m; ++i) {
            out[j*ldout+i] = a[j*lda+i];
         }
      }
      
   }

   template <typename T, typename PoolAlloc>
   void permute_failed_unsym(
         NumericFront<T, PoolAlloc> &node) {
      
      // Total number of eliminated rows/columns whithin node 
      int block_size = node.blksz;
      int num_elim = node.nelim;
      int n = node.get_ncol(); // Number of fully-summed rows/columns
      // Number of block rows/columns in the fully-summed
      int nblk = node.get_nc();

      int nfail = n-num_elim; // Number of failed columns

      // Factor entries
      T *lcol = node.lcol;
      int ldl = node.get_ldl();

      // Column data
      typedef typename NumericFront<T, PoolAlloc>::IntAlloc IntAlloc;
      ColumnData<T, IntAlloc>& cdata = *node.cdata; // Column data

      // Permutation
      int *rperm = node.perm;
      int *cperm = node.cperm;
      
      // std::vector<int, IntAlloc> failed_perm(n-num_elim, alloc);
      std::vector<int> failed_perm (nfail);
      std::vector<int> failed_cperm(nfail);
      
      // Permute fail entries to the back of the matrix
      for(int jblk=0, insert=0, fail_insert=0; jblk<nblk; jblk++) {
         
         // int blk_n = get_ncol(jblk, n, blksz)
         int blk_n = std::min(block_size, n-(jblk*block_size)); // Number of fully-summed within block

         // Move back failed rows 
         cdata[jblk].move_back(
               blk_n, &rperm[jblk*block_size],
               &rperm[insert], &failed_perm[fail_insert]
               );

         // Move back failed columns 
         cdata[jblk].move_back(
               blk_n, &cperm[jblk*block_size],
               &cperm[insert], &failed_cperm[fail_insert]
               );

         insert += cdata[jblk].nelim;
         fail_insert += blk_n - cdata[jblk].nelim;
      }

      for(int i=0; i < nfail; ++i) {
         rperm[num_elim+i] = failed_perm [i];
         cperm[num_elim+i] = failed_cperm[i];
      }

      std::vector<T> failed_diag(nfail*nfail);
      
      // Extract failed entries

      // Diagonal part
      for (int jblk=0, jfail=0, jinsert=0; jblk<nblk; ++jblk) {

         int blk_n = std::min(block_size, n-(jblk*block_size)); // Number of fully-summed within block

         for (int iblk=0, ifail=0, iinsert=0; iblk<nblk; ++iblk) {

            int blk_m = std::min(block_size, n-(iblk*block_size)); // Number of fully-summed within block
            
            copy_failed_diag_unsym(
                  blk_m, blk_n, cdata[iblk].nelim, cdata[jblk].nelim, 
                  lcol, ldl, &failed_diag[jfail*nfail+ifail], nfail);
            
            iinsert += cdata[iblk].nelim;
            ifail += blk_m - cdata[iblk].nelim;
         }
         
         jinsert += cdata[jblk].nelim;
         jfail += blk_n - cdata[jblk].nelim;
      }

      // Rectangular part
      // ...

      // Move up eliminated entries

      // Diagonal part
      for (int jblk=0, jinsert=0; jblk<nblk; ++jblk) {

         int blk_n = std::min(block_size, n-(jblk*block_size)); // Number of fully-summed within block

         for (int iblk=0, iinsert=0; iblk<nblk; ++iblk) {

            int blk_m = std::min(block_size, n-(iblk*block_size)); // Number of fully-summed within block
            
            move_up_diag_unsym(
                  cdata[iblk].nelim, cdata[jblk].nelim, 
                  &lcol[jblk*block_size*ldl+iblk*block_size], ldl, 
                  &lcol[jinsert*ldl+iinsert], ldl);
            
            iinsert += cdata[iblk].nelim;
         }
         jinsert += cdata[jblk].nelim;
      }

      // Copy failed entries back to factor entries
      
      // Diagonal part
      // move_up_diag_unsym(
      //       nfail, nfail, 
      //       &failed_diag[0], nfail,
      //       &lcol[num_elim*ldl+num_elim], ldl);

      // for (int j = 0; j < nfail; ++j) {
      //    for (int i = 0; i < nfail; ++i) {
      //       move_up_diag_unsym(
      //             cdata[iblk].nelim, cdata[jblk].nelim, 
      //             failed_diag, nfail,
      //             &lcol[num_elim*ldl+num_elim], ldl);
      //    }
      // }

   }
   
   /// @brief Task-based front factorization routine using Restricted
   /// Pivoting (RP)
   /// @Note No delays, potentially unstable
   template <typename T, typename PoolAlloc>
   void factor_front_unsym_rp(
         NumericFront<T, PoolAlloc> &node,
         std::vector<spral::ssids::cpu::Workspace>& workspaces) {

      // Extract front info
      int m = node.get_nrow(); // Frontal matrix order
      int n = node.get_ncol(); // Number of fully-summed rows/columns 
      int nr = node.get_nr(); // number of block rows
      int nc = node.get_nc(); // number of block columns
      size_t contrib_dimn = m-n;
      int blksz = node.blksz;
      
      printf("[factor_front_unsym_rp] m = %d\n", m);
      printf("[factor_front_unsym_rp] n = %d\n", n);
      
      for(int k = 0; k < nc; ++k) {
         BlockUnsym<T>& dblk = node.get_block_unsym(k, k);
         int *perm = &node.perm[k*blksz];
         
         factor_block_lu_pp_task(dblk, perm);

         // Apply permutation
         for (int j = 0; j < k; ++j) {
            BlockUnsym<T>& rblk = node.get_block_unsym(k, j);
            // Apply row permutation on left-diagonal blocks 
            apply_rperm_block_task(dblk, rblk, workspaces);
         }

         // Apply permutation and compute U factors
         for (int j = k+1; j < nc; ++j) {
            BlockUnsym<T>& ublk = node.get_block_unsym(k, j);

            applyL_block_task(dblk, ublk, workspaces);
         }

         // Compute L factors
         for (int i =  k+1; i < nr; ++i) {
            BlockUnsym<T>& lblk = node.get_block_unsym(i, k);
            
            applyU_block_task(dblk, lblk);            
         }

         int en = (n-1)/blksz; // Last block-row/column in factors
            
         // Udpdate trailing submatrix
         for (int j = k+1; j < nr; ++j) {

            BlockUnsym<T>& ublk = node.get_block_unsym(k, j);

            for (int i =  k+1; i < nr; ++i) {
               // Loop if we are in the cb
               if ((i > en) && (j > en)) continue;

               BlockUnsym<T>& lblk = node.get_block_unsym(i, k);
               BlockUnsym<T>& blk = node.get_block_unsym(i, j);
               
               update_block_lu_task(lblk, ublk, blk);
            }
         }
         
         if (contrib_dimn>0) {
            // Update contribution block
            int rsa = n/blksz; // Last block-row/column in factors
         
            for (int j = rsa; j < nr; ++j) {

               BlockUnsym<T>& ublk = node.get_block_unsym(k, j);

               for (int i = rsa; i < nr; ++i) {

                  BlockUnsym<T>& lblk = node.get_block_unsym(i, k);
                  Tile<T, PoolAlloc>& cblk = node.get_contrib_block(i, j);
                  
                  update_cb_block_lu_task(lblk, ublk, cblk);
               
               }
            }
         }
      }

      // Note: we do not check for stability       
      node.nelim = n; // We eliminated all fully-summed rows/columns
      node.ndelay_out = 0; // No delays

   }

}
