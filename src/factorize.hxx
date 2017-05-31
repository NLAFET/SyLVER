#pragma once

#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/kernels/wrappers.hxx"

#include "Workspace.hxx"
#include "SymbolicSNode.hxx"
#include "kernels/factor.hxx"
#include "SymbolicTree.hxx"

#if defined(SPLDLT_USE_STARPU)
#include <starpu.h>
#include "StarPU/kernels.hxx"
using namespace spldlt::starpu;
#endif

// using namespace spldlt;

namespace spldlt {

   /* Activate node: allocate data structures */
   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void activate_node(
         SymbolicSNode &snode,
         NumericNode<T, PoolAlloc> &node,
         int blksz,
         FactorAlloc& factor_alloc,
         PoolAlloc& pool_alloc) {

      alloc_node(snode, node, factor_alloc, pool_alloc);            

#if defined(SPLDLT_USE_STARPU)
      starpu_void_data_register(&(snode.hdl));

      /* Register blocks in StarPU */
      // printf("[NumericTree] regiter node: %d\n", ni);
      register_node(snode, node, blksz);

      // activate_node(snode, nodes_[ni], nb);
#endif
      
   }

   /* Initialize node */ 
   template <typename T, typename PoolAlloc>
   void init_node_task(
         SymbolicSNode &snode,
         NumericNode<T, PoolAlloc> &node,
         T *aval, int prio) {

#if defined(SPLDLT_USE_STARPU)
      
      // init_node(snode, node, aval);
      insert_init_node(
            &snode, &node,
            snode.hdl,
            aval, prio);
#else

      init_node(snode, node, aval);

#endif

   }

   /* Factorize block on daig task */

   // TODO create and use BLK structure
   template <typename T, typename PoolAlloc>
   void factorize_diag_block_task(
         SymbolicSNode const& snode,
         NumericNode<T, PoolAlloc> &node,
         int kk, // block  column (and row) index
         int blksz, int prio) {

      int m = snode.nrow;
      int n = snode.ncol;

      int nr = (m-1) / blksz + 1; // number of block rows
      int nc = (n-1) / blksz + 1; // number of block columns
      
      // printf("[factorize_diag_block_task] nr: %d, nc: %d, kk: %d\n", 
      //        nr, nc, kk);

      // printf("[factorize_diag_block_task] \n");

#if defined(SPLDLT_USE_STARPU)
      
      
      starpu_data_handle_t node_hdl = NULL;
      if (kk == 0) node_hdl = snode.hdl; 
      // printf("[factorize_diag_block_task] handles size: %zu\n", snode.handles.size());
      insert_factorize_block(snode.handles[kk*nr + kk], node_hdl, prio);

#else

      int blkm = std::min(blksz, m - kk*blksz); 
      int blkn = std::min(blksz, n - kk*blksz);
      int lda = align_lda<T>(m);
      T *a = node.lcol;
      T *contrib = node.contrib;
      int ldcontrib = m-n;

      factorize_diag_block(blkm, blkn, 
                           &a[kk*blksz*(lda+1)], lda,
                           contrib, ldcontrib,
                           kk==0);

#endif
   }

   /* Solve block (on subdaig) task */

   template <typename T>
   void solve_block_task(
         SymbolicSNode const& snode,
         int kk, /* block column index of subdiag block */
         int ii, /* block row index of subdiag block*/
         T *a, int lda, 
         T *a_ik, int lda_ik,         
         int blksz, int prio) {
      
      int m = snode.nrow;
      int n = snode.ncol;

      int nr = (m-1) / blksz + 1; // number of block rows
      int nc = (n-1) / blksz + 1; // number of block columns

#if defined(SPLDLT_USE_STARPU)
      
      insert_solve_block(
            snode.handles[kk*nr + kk], // diag block handle 
            snode.handles[kk*nr + ii], // subdiag block handle
            prio);
      
#else

      int blkn = std::min(blksz, n - kk*blksz);
      int blkm = std::min(blksz, m - ii*blksz);
      
      solve_block(blkm, blkn, a, lda, a_ik, lda_ik);

#endif   
   }

   /* Update block on subdaig task */
   
   template <typename T>
   void update_block_task(
         SymbolicSNode const& snode,
         int kk, /* block column index of A_ik and A_jk blocks */
         int ii, /* block row index of A_ik and A_ij blocks  */
         int jj, /* block row index of A_jk and block column index of
                    A_ij blocks */
         T *a_ij, int lda_ij,
         T *a_ik, int lda_ik,
         T *a_jk, int lda_jk,
         int blksz, int prio) {

      int m = snode.nrow;
      int n = snode.ncol;

#if defined(SPLDLT_USE_STARPU)

      int nr = (m-1) / blksz + 1; // number of block rows
      int nc = (n-1) / blksz + 1; // number of block columns

      insert_update_block(
            snode.handles[jj*nr + ii], // A_ij block handle 
            snode.handles[kk*nr + ii], // A_ik block handle
            snode.handles[kk*nr + jj],  // A_jk block handle
            prio);
#else

      int blkm = std::min(blksz, m - ii*blksz);
      int blkn = std::min(blksz, n - jj*blksz);
      int blkk = std::min(blksz, n - kk*blksz);

      update_block(blkm, blkn, a_ij, lda_ij,
                   blkk,
                   a_ik, lda_ik,
                   a_jk, lda_jk);

#endif
   }

   template <typename T>
   void update_diag_block_task(
         SymbolicSNode const& snode,
         int kk, /* block column index of A_ik and A_jk blocks */
         int ii, /* block row index of A_ik and A_ij blocks  */
         int jj, /* block row index of A_jk and block column index of
                    A_ij blocks */
         T *a_ij, int lda_ij,
         T *a_ik, int lda_ik,
         T *a_jk, int lda_jk,
         int blksz, int prio) {

      int m = snode.nrow;
      int n = snode.ncol;

#if defined(SPLDLT_USE_STARPU)

      int nr = (m-1) / blksz + 1; // number of block rows
      int nc = (n-1) / blksz + 1; // number of block columns

      insert_update_diag_block(
            snode.handles[jj*nr + ii], // A_ij block handle 
            snode.handles[kk*nr + ii], // A_ik block handle
            snode.handles[kk*nr + jj],  // A_jk block handle
            prio);

#else

      int blkm = std::min(blksz, m - ii*blksz);
      int blkn = std::min(blksz, n - jj*blksz);
      int blkk = std::min(blksz, n - kk*blksz);

      update_diag_block(blkm, blkn, a_ij, lda_ij,
                        blkk,
                        a_ik, lda_ik,
                        a_jk, lda_jk);

#endif
   };

   /* Factorize node in a MF context
    */
   template <typename T, typename PoolAlloc>
   void factorize_node_posdef_mf(
         SymbolicSNode const& snode,
         NumericNode<T, PoolAlloc> &node,
         struct cpu_factor_options const& options
         ) {

      /* Extract useful information about node */
      int m = snode.nrow;
      int n = snode.ncol;
      int lda = align_lda<T>(m);
      T *lcol = node.lcol;
      T *contrib = node.contrib;
      int ldcontrib = m-n;

      int blksz = options.cpu_block_size;
      int nr = (m-1) / blksz + 1; // number of block rows
      int nc = (n-1) / blksz + 1; // number of block columns
   
      // printf("[factorize_node_posdef_mf] contrib: %p\n", contrib);

      int FACTOR_PRIO = 3;

      for(int j = 0; j < nc; ++j) {

         int blkn = std::min(blksz, n - j*blksz);
         
         /* Diagonal Block Factorization Task */

         int blkm = std::min(blksz, m - j*blksz);
         
         // factorize_diag_block(
         //       blkm, blkn, 
         //       &lcol[j*blksz*(lda+1)], lda,
         //       contrib, ldcontrib,
         //       j==0);

         factorize_diag_block_task(
               snode, node, j, // block  column (and row) index
               blksz, FACTOR_PRIO);

#if defined(SPLDLT_USE_STARPU)
            starpu_task_wait_for_all();
#endif

         /* Column Solve Tasks */
         for(int i = j+1; i < nr; ++i) {

            int blkm = std::min(blksz, m - i*blksz);

            // printf("[factorize_node_posdef_mf] contrib start: %d\n", (i*blksz)-n);

            solve_block(blkm, blkn, 
                        &lcol[j*blksz*(lda+1)], lda, 
                        &lcol[(j*blksz*lda) + (i*blksz)], lda,
                        (contrib) ? &contrib[(i*blksz)-n] : nullptr, ldcontrib,
                        j==0, 
                        blksz);
         }

         /* Schur Update Tasks: mostly internal */
         for(int k = j+1; k < nc; ++k) {

            int blkk = std::min(blksz, n - k*blksz);

            for(int i = k;  i < nr; ++i) {
               
               int blkm = std::min(blksz, m - i*blksz);
               
               // int cbm = (i*blksz < n) ? std::min((i+1)*blksz,m)-n : blkm;
               int cbm = (i*blksz < n) ? blkm+(i*blksz)-n : blkm;
               int cbn = std::min(blksz, m-k*blksz)-blkk;
               T *upd = nullptr;
                  
               if (contrib)
                  upd = (i*blksz < n) ? contrib : &contrib[(i*blksz)-n];

               // int cbm = std::min(blksz, m-std::max(n,i*blksz));
               // int cbn = std::min(blksz, m-k*blksz)-blkk;
               
               // printf("[factorize_node_posdef_mf] m: %d, k: %d\n", m, k);
               // printf("[factorize_node_posdef_mf] cbm: %d, cbn: %d\n", cbm, cbn);
               
               update_block(blkm, blkk, &lcol[ (k*blksz*lda) + (i*blksz)], lda,
                            blkn,
                            &lcol[(j*blksz*lda) + (i*blksz)], lda, 
                            &lcol[(j*blksz*lda) + (k*blksz)], lda,
                            upd, ldcontrib,
                            cbm, cbn,
                            j==0,
                            blksz);

            }
         }

         /* Contrib Schur complement update: external */
         if (contrib) {
            // printf("[factorize_node_posdef_mf] node: %d\n", snode.idx);
            // printf("[factorize_node_posdef_mf] nc: %d, nr: %d\n", nc, nr);
            for (int k = nc; k < nr; ++k) {
               
               int blkk = std::min(blksz, m - k*blksz);

               for (int i = k;  i < nr; ++i) {
               
                  int blkm = std::min(blksz, m - i*blksz);

                  // printf("[factorize_node_posdef_mf] row: %d, col: %d\n", i, k);
                  // printf("[factorize_node_posdef_mf] blkm: %d, blkk: %d\n", i, k);
                  
                  // printf("[factorize_node_posdef_mf] row: %d, col: %d\n", (i*blksz)-n, (k*blksz-n));

                  update_block(
                        blkm, blkk,
                        &contrib[((k*blksz-n)*ldcontrib) + (i*blksz)-n], ldcontrib,
                        blkn,
                        &lcol[(j*blksz*lda) + (i*blksz)], lda, 
                        &lcol[(j*blksz*lda) + (k*blksz)], lda,
                        j==0);
                  
                  // update_block(blkm, blkk, &contrib[ (k*blksz*lda) + (i*blksz)], lda,
                  //              blkn,
                  //              &lcol[(j*blksz*lda) + (i*blksz)], lda,
                  //              &lcol[(j*blksz*lda) + (k*blksz)], lda,
                  //              contrib, ldcontrib,
                  //              cbm, cbn,
                  //              j==0,
                  //              blksz);

               }
            }
         }
      }
   }

   // TODO: error managment
   template <typename T, typename PoolAlloc>
   void factorize_node_posdef(
         SymbolicSNode const& snode,
         NumericNode<T, PoolAlloc> &node,
         struct cpu_factor_options const& options
         ) {

      /* Extract useful information about node */
      int m = snode.nrow;
      int n = snode.ncol;
      int lda = align_lda<T>(m);
      T *a = node.lcol;
      // T *contrib = nodes_[ni].contrib;

      int blksz = options.cpu_block_size;
      int nr = (m-1) / blksz + 1; // number of block rows
      int nc = (n-1) / blksz + 1; // number of block columns

      /* Task priorities:
         init: 4
         facto: 3
         solve: 2
         udpate: 1
         udpate_between: 0
      */
   
      for(int j = 0; j < nc; ++j) {
         
         /* Diagonal Block Factorization Task */
         int blkm = std::min(blksz, m - j*blksz);
         int blkn = std::min(blksz, n - j*blksz);
         
         factorize_diag_block_task(snode, node, j, blksz, 3);
         
         // factorize_diag_block(blkm, blkn, &a[j*blksz*(lda+1)], lda);


         /* Column Solve Tasks */
         for(int i = j+1; i < nr; ++i) {

            int blkm = std::min(blksz, m - j*blksz);
            
            solve_block_task(
                  snode,
                  j, i,
                  &a[j*blksz*(lda+1)], lda,
                  &a[(j*blksz*lda) + (i*blksz)], lda,
                  blksz, 2);

            // solve_block(
            //       blkm, blkn, 
            //       &a[j*blksz*(lda+1)], lda, 
            //       &a[(j*blksz*lda) + (i*blksz)], lda);
         }

         /* Schur Update Tasks: mostly internal */
         for(int k = j+1; k < nc; ++k) {

            update_diag_block_task(
                  snode,
                  j, /* block column index of A_ik and A_jk blocks */
                  k, /* block row index of A_ik and A_ij blocks  */
                  k, /* block row index of A_jk and block column index of
                        A_ij blocks */
                  &a[ (k*blksz*lda) + (k*blksz)], lda,
                  &a[(j*blksz*lda) + (k*blksz)], lda,
                  &a[(j*blksz*lda) + (k*blksz)], lda,
                  blksz, 1);

            // int blkm = std::min(blksz, m - k*blksz);
            // int blkn = std::min(blksz, n - k*blksz);
            // int blkk = std::min(blksz, n - j*blksz);

            // update_diag_block(blkm, blkn, 
            //                   &a[(k*blksz*lda) + (k*blksz)], lda,
            //                   blkk,
            //                   &a[(j*blksz*lda) + (k*blksz)], lda,
            //                   &a[(j*blksz*lda) + (k*blksz)], lda);


            for(int i = k+1;  i < nr; ++i) {
               
               update_block_task(
                     snode,
                     j, /* block column index of A_ik and A_jk blocks */
                     i, /* block row index of A_ik and A_ij blocks  */
                     k, /* block row index of A_jk and block column index of
                           A_ij blocks */
                     &a[ (k*blksz*lda) + (i*blksz)], lda,
                     &a[(j*blksz*lda) + (i*blksz)], lda,
                     &a[(j*blksz*lda) + (k*blksz)], lda,
                     blksz, 1);
            }
         }         
      }
   }

   // Serial version

   // TODO: error managment
   template <typename T, typename PoolAlloc>
   void factorize_node_posdef_notask(
         SymbolicSNode const& snode,
         NumericNode<T, PoolAlloc> &node,
         struct cpu_factor_options const& options
         ) {

      /* Extract useful information about node */
      int m = snode.nrow;
      int n = snode.ncol;
      int lda = align_lda<T>(m);
      T *lcol = node.lcol;
      // T *contrib = nodes_[ni].contrib;

      int blksz = options.cpu_block_size;
      int nr = (m-1) / blksz + 1; // number of block rows
      int nc = (n-1) / blksz + 1; // number of block columns
   
      // int flag;
      // cholesky_factor(
      //       m, n, lcol, ldl, 0.0, NULL, 0, options.cpu_block_size, &flag
      //       );
      // int *info = new int(-1);

      for(int j = 0; j < nc; ++j) {

         int blkn = std::min(blksz, n - j*blksz);
         
         /* Diagonal Block Factorization Task */

         int blkm = std::min(blksz, m - j*blksz);
         // int flag = lapack_potrf(FILL_MODE_LWR, blkn, &a[j*(lda+1)], lda);
         // if(blkm>blkn) {
         //    // Diagonal block factored OK, handle some rectangular part of block
         //    host_trsm(SIDE_RIGHT, FILL_MODE_LWR, OP_T, DIAG_NON_UNIT,
         //              blkm-blkn, blkn, 1.0, &a[j*(lda+1)], lda,
         //              &a[j*(lda+1)+blkn], lda);

         //    // if(upd) {
         //    //    double rbeta = (j==0) ? beta : 1.0;
         //    //    host_syrk(FILL_MODE_LWR, OP_N, blkm-blkn, blkn, -1.0,
         //    //              &a[j*(lda+1)+blkn], lda, rbeta, upd, ldupd);
         //    // }
         // }
         
         factorize_diag_block(blkm, blkn, &lcol[j*blksz*(lda+1)], lda);
         
         /* Column Solve Tasks */
         for(int i = j+1; i < nr; ++i) {
            
            int blkm = std::min(blksz, m - i*blksz);
            
            solve_block(blkm, blkn, 
                        &lcol[j*blksz*(lda+1)], lda, 
                        &lcol[(j*blksz*lda) + (i*blksz)], lda);
            
            // host_trsm(SIDE_RIGHT, FILL_MODE_LWR, OP_T, DIAG_NON_UNIT,
            //           blkm, blkn, 1.0, &a[j*(lda+1)], lda, &a[j*lda+i], lda);

            // if(blkn<blksz && upd) {
            //    double rbeta = (j==0) ? beta : 1.0;
            //    host_gemm(OP_N, OP_T, blkm, blksz-blkn, blkn, -1.0,
            //              &a[j*lda+i], lda, &a[j*(lda+1)+blkn], lda,
            //              rbeta, &upd[i-n], ldupd);
            // }
         }

         /* Schur Update Tasks: mostly internal */
         for(int k = j+1; k < nc; ++k) {

            int blkk = std::min(blksz, n - k*blksz);

            for(int i = k;  i < nr; ++i) {
               
               int blkm = std::min(blksz, m - i*blksz);

               update_block(blkm, blkk, &lcol[ (k*blksz*lda) + (i*blksz)], lda,
                            blkn,
                            &lcol[(j*blksz*lda) + (i*blksz)], lda, 
                            &lcol[(j*blksz*lda) + (k*blksz)], lda);

               // host_gemm(OP_N, OP_T, blkm, blkk, blkn, -1.0, &a[j*lda+i], lda,
               //           &a[j*lda+k], lda, 1.0, &a[k*lda+i], lda);

               // if(blkk < blksz && upd) {
               //    double rbeta = (j==0) ? beta : 1.0;
               //    int upd_width = (m<k+blksz) ? blkm - blkk
               //       : blksz - blkk;
               //    if(i-n < 0) {
               //       // Special case for first block of contrib
               //       host_gemm(OP_N, OP_T, blkm+i-n, upd_width, blkn, -1.0,
               //                 &a[j*lda+n], lda, &a[j*lda+k+blkk], lda, rbeta,
               //                 upd, ldupd);
               //    } else {
               //       host_gemm(OP_N, OP_T, blkm, upd_width, blkn, -1.0,
               //                 &a[j*lda+i], lda, &a[j*lda+k+blkk], lda, rbeta,
               //                 &upd[i-n], ldupd);
               //    }
               // }
            }
         }
         
      }
   }

   /* Update between task */

   template <typename T, typename PoolAlloc>
   void update_between_block_task(
         SymbolicSNode &snode, // symbolic source node
         NumericNode<T, PoolAlloc> &node, // numeric source node
         int cptr, int cptr2, // pointer to the first and last row of
                              // A_jk block
         int rptr, int rptr2, // pointer to the first and last row of
                              // A_ik block
         int kk, // column index of A_jk and A_ik blocks
         SymbolicSNode &asnode, // symbolic destination node
         NumericNode<T, PoolAlloc> &anode, // numeric destination node
         int ii, /* block row index of A_ij block in ancestor node  */
         int jj, /* block column index of A_ij block in ancestor node */
         int blksz, // blocking size
         Workspace &work,
         Workspace &rowmap, Workspace& colmap,
         int prio) {

      /* Extract useful information about node */
      int m = snode.nrow;
      int n = snode.ncol;

      int nr = (m-1) / blksz + 1; // number of block rows
      int nc = (n-1) / blksz + 1; // number of block columns
      // printf("kk: %d, ii: %d, jj: %d\n", kk, ii, jj);
#if defined(SPLDLT_USE_STARPU)

      /* Extract useful information about anode */
      int a_m = asnode.nrow;
      int a_n = asnode.ncol;
      int a_nr = (a_m-1) / blksz +1; // number of block rows
      int a_nc = (a_n-1) / blksz +1; // number of block columns
      
      // A_ik
      // First block in A_ik
      int blk_sa = rptr  / blksz;
      int blk_en = rptr2 / blksz;
      int nblk_ik = blk_en - blk_sa +1;
      // printf("rptr: %d, rptr2: %d, blk_sa: %d, blk_en: %d, nblk_ik: %d\n", rptr, rptr2, blk_sa, blk_en, nblk_ik);
      starpu_data_handle_t *bc_ik_hdls = new starpu_data_handle_t[nblk_ik];
      for (int i = 0; i < nblk_ik; ++i) {
         // printf("[update_between_block_task] a_ik handle %d ptr: %p\n", i, snode.handles[kk*nr + blk_sa + i]);
         bc_ik_hdls[i] = snode.handles[kk*nr + blk_sa + i];
      }

      // A_jk
      // First block in A_jk
      blk_sa = cptr  / blksz;
      blk_en = cptr2 / blksz;
      int nblk_jk = blk_en - blk_sa +1;
      // printf("cptr: %d, cptr2: %d, blk_sa: %d, blk_en: %d, nblk_jk: %d\n", cptr, cptr2, blk_sa, blk_en, nblk_jk);      
      starpu_data_handle_t *bc_jk_hdls = new starpu_data_handle_t[nblk_jk];
      for (int i = 0; i < nblk_jk; ++i) {
         // printf("[update_between_block_task] a_jk handle %d ptr: %p\n", i, snode.handles[kk*nr + blk_sa + i]);
         bc_jk_hdls[i] = snode.handles[kk*nr + blk_sa + i];
      }
      // printf("[update_between_block_task] a_nr: %d, ii: %d, jj: %d\n", a_nr, ii, jj);
      // printf("[update_between_block_task] a_ij handle ptr: %p\n", asnode.handles[jj*a_nr + ii]);
      insert_update_between(
            &snode, &node,
            bc_ik_hdls, nblk_ik,
            bc_jk_hdls, nblk_jk,
            kk,
            &asnode,
            asnode.hdl,
            asnode.handles[jj*a_nr + ii],
            ii, jj,
            work.hdl, rowmap.hdl, colmap.hdl,
            blksz,
            cptr, cptr2,
            rptr, rptr2,
            prio);

      // int blkn = std::min(blksz, n - kk*blksz);
      // T *a_lcol = anode.lcol;
      // int a_ldl = align_lda<T>(asnode.nrow);

      // int mc = cptr2-cptr+1; // number of rows in Ajk
      // int *clst = colmap.get_ptr<int>(mc);
      // int mr = rptr2-rptr+1; // number of rows in Aik
      // int *rlst = rowmap.get_ptr<int>(mr); // Get ptr on rowmap array
      // T *buffer = work.get_ptr<T>(mr*mc);

      // update_between_block(blkn, 
      //                      kk, ii, jj, 
      //                      blksz, 
      //                      cptr, cptr2, rptr, rptr2,
      //                      snode, node,
      //                      asnode, &a_lcol[(jj*blksz*a_ldl)+(ii*blksz)], a_ldl,
      //                      buffer, rlst, clst);
      
      delete[] bc_ik_hdls;
      delete[] bc_jk_hdls;

#else

      int blkn = std::min(blksz, n - kk*blksz);
      T *a_lcol = anode.lcol;
      int a_ldl = align_lda<T>(asnode.nrow);

      int mc = cptr2-cptr+1; // number of rows in Ajk
      int *clst = colmap.get_ptr<int>(mc);
      int mr = rptr2-rptr+1; // number of rows in Aik
      int *rlst = rowmap.get_ptr<int>(mr); // Get ptr on rowmap array
      T *buffer = work.get_ptr<T>(mr*mc);
      // T *buffer = work.get_ptr<T>(blksz*blksz);
      // printf("[update_between_block_task] kk: %d, ii: %d, jj: %d\n", kk, ii, jj);
      // printf("[update_between_block_task] cptr: %d, cptr2: %d, rptr: %d, rptr2: %d\n", cptr, cptr2, rptr, rptr2);

      update_between_block(blkn,
                           kk, ii, jj,
                           blksz, 
                           cptr, cptr2, rptr, rptr2,
                           snode, node,
                           asnode, &a_lcol[(jj*blksz*a_ldl)+(ii*blksz)], a_ldl,
                           buffer, rlst, clst);

#endif

   }

   /* Apply factorization to ancestor node */
   
   template <typename T, typename PoolAlloc>
   void apply_node(
         SymbolicSNode &snode, // symbolic node to be applied 
         NumericNode<T, PoolAlloc> &node, // numeric node to be applied
         int nnodes,
         SymbolicTree &stree, // symbolic tree
         std::vector<NumericNode<T,PoolAlloc>> &nodes, // list of nodes in the tree
         int blksz, // blocking size
         // // struct cpu_factor_options const& options
         Workspace &work,
         Workspace &rowmap, Workspace& colmap
         ) {

      // return;
      // printf("node idx: %d\n", snode.idx);
      // return;

      /* Extract useful information about node */
      int m = snode.nrow; // number of row in node
      int n = snode.ncol; // number of column in node
      int ldl = align_lda<T>(m); // leading dimensions
      T *lcol = node.lcol; // numerical values
      int nc = (n-1)/blksz +1; // number of block column

      Workspace map(stree.n); // maxfront size
      int *amap = map.get_ptr<int>(stree.n); // row to block row mapping array 
      bool map_done = false;
      
      int cptr = n; // point to first row below diag in node
      int cptr2 = 0; 
      
      int parent = snode.parent;

      int prio = 0;

      while (parent < nnodes) {
         // printf("[apply_node] parent: %d\n", parent);
         // NumericNode<T,PoolAllocator> &anode = nodes_[parent];

         SymbolicSNode &asnode = stree[parent]; // parent symbolic node 
         int sa = asnode.sa;
         int en = asnode.en;


         T *a_lcol = nodes[asnode.idx].lcol;
         int a_ldl = align_lda<T>(asnode.nrow);

         map_done = false;
               
         while (cptr < m) {
            if (snode.rlist[cptr] >= sa) break;
            cptr++;
         }
         if (cptr >= m) break;

         while(cptr < m) {
            if (snode.rlist[cptr] > en) break;
                  
            // determine last column index of current block column 
            int cb = (snode.rlist[cptr] - sa) / blksz;
            int jlast = std::min(sa + (cb+1)*blksz-1, en);

            // printf("[NumericTree] cb: %d\n", cb);

            // find cptr2
            cptr2 = cptr;
            while (cptr2 < m) {
               if (snode.rlist[cptr2] > jlast) break;
               cptr2++;
            }
            cptr2--;

            // printf("cptr: %d, rlist[cptr]: %d, sa: %d, cptr2: %d, rlist[cptr2]: %d, en: %d\n", 
            //        cptr, snode.rlist[cptr], sa, cptr2, snode.rlist[cptr2], en);
                  
            if (!map_done) {
               // int a_blksz = blksz;
               int r = 0; // row index
               int rr = 0; // block row index
               for (int row = 0; row < asnode.nrow; ++row) {
                  rr = row / blksz;
                  r = asnode.rlist[row];
                  amap[r] = rr;
               }
               map_done = true;
            }

            int mc = cptr2-cptr+1; // number of rows in Ajk
            int *clst = colmap.get_ptr<int>(mc);

            int rptr = cptr;
            int ii = amap[snode.rlist[cptr]]; // first block row in anode

            int rptr2 = 0;
            for (rptr2 = cptr; rptr2 < m; rptr2++) {
               int k = amap[snode.rlist[rptr2]];
                     
               if (k != ii) {
                  
                  int mr = rptr2-rptr; // number of rows in Aik
                  int *rlst = rowmap.get_ptr<int>(mr); // Get ptr on rowmap array
                  T *buffer = work.get_ptr<T>(mr*mc);
                        
                  for (int kk = 0; kk < nc; ++kk) {

                     int blkn = std::min(blksz, n-(kk*blksz));

                     update_between_block_task(
                           snode, node,
                           cptr, cptr2, rptr, rptr2-1,
                           kk,
                           asnode, nodes[asnode.idx],
                           ii, cb,
                           blksz,
                           work, rowmap, colmap,
                           prio);

                     // update_between_block(blkn,
                     //                      kk, ii, cb,
                     //                      blksz, 
                     //                      cptr, cptr2, rptr, rptr2-1,
                     //                      snode, node,
                     //                      asnode, 
                     //                      &a_lcol[(cb*blksz*a_ldl)+(ii*blksz)], a_ldl,
                     //                      buffer, rlst, clst);

                  }


                  ii = k;
                  rptr = rptr2;
               }
            }

            int mr = rptr2-rptr+1; // number of rows in Aik
            int *rlst = rowmap.get_ptr<int>(mr); // Get ptr on rowmap array
            T *buffer = work.get_ptr<T>(mr*mc);

            for (int kk = 0; kk < nc; ++kk) {

               int blkn = std::min(blksz, n-(kk*blksz));

               update_between_block_task(
                     snode, node,
                     cptr, cptr2, rptr, rptr2-1,
                     kk, 
                     asnode, nodes[asnode.idx],
                     ii, cb,
                     blksz,
                     work, rowmap, colmap,
                     prio);                     

               // update_between_block(blkn,
               //                      kk, ii, cb,
               //                      blksz, 
               //                      cptr, cptr2, rptr, rptr2-1,
               //                      snode, node,
               //                      asnode, 
               //                      &a_lcol[(cb*blksz*a_ldl)+(ii*blksz)], a_ldl,
               //                      buffer, rlst, clst);

            }
            
            cptr = cptr2 + 1; // move cptr
         }

         parent = stree[asnode.idx].parent; // move up the tree
      }
   }

   // Serial version

   template <typename T, typename PoolAlloc>
   void apply_node_notask(
         SymbolicSNode const& snode, // symbolic node to be applied 
         NumericNode<T, PoolAlloc> &node, // numeric node to be applied
         int nnodes,
         SymbolicTree const& stree, // symbolic tree
         std::vector<NumericNode<T,PoolAlloc>> &nodes, // list of nodes in the tree
         int nb, // blocking size
         // struct cpu_factor_options const& options
         Workspace &work,
         Workspace &rowmap, Workspace& colmap
         ) {
      
      /* Extract useful information about node */
      int m = snode.nrow; // number of row in node
      int n = snode.ncol; // number of column in node
      int ldl = align_lda<T>(m); // leading dimensions
      T *lcol = node.lcol; // numerical values
      int nc = (n-1)/nb +1; // number of block column

      Workspace map(stree.n); // maxfront size
      int *amap = map.get_ptr<int>(stree.n); // row to block row mapping array 
      bool map_done = false;
      
      int cptr = n; // point to first row below diag in node
      int cptr2 = 0; 
      
      int parent = snode.parent;
      while (parent < nnodes) {
         // NumericNode<T,PoolAllocator> &anode = nodes_[parent];
         SymbolicSNode const& asnode = stree[parent]; // parent symbolic node 
         int sa = asnode.sa;
         int en = asnode.en;


         T *a_lcol = nodes[asnode.idx].lcol;
         int a_ldl = align_lda<T>(asnode.nrow);

         map_done = false;
         // printf("[NumericTree] node: %d, parent: %d, sa: %d, en: %d\n", ni, asnode.idx, sa, en);

         // printf("cptr: %d, rlist[cptr]: %d, cptr2: %d, rlist[cptr2]: %d\n", 
         //        cptr, snode.rlist[cptr], cptr2, snode.rlist[cptr2]);
               
         // for (int i = 0; i < m; i++) 
         //    printf(" %d ", snode.rlist[i]);
         // printf("\n");
               
         while (cptr < m) {
            if (snode.rlist[cptr] >= sa) break;
            cptr++;
         }
         if (cptr >= m) break;

         while(cptr < m) {
            if (snode.rlist[cptr] > en) break;
                  
            // determine last column index of current block column 
            int cb = (snode.rlist[cptr] - sa) / nb;
            int jlast = std::min(sa + (cb+1)*nb, en);

            // printf("[NumericTree] cb: %d\n", cb);

            // find cptr2
            cptr2 = cptr;
            while (cptr2 < m) {
               if (snode.rlist[cptr2] > jlast) break;
               cptr2++;
            }
            cptr2--;

            // printf("cptr: %d, rlist[cptr]: %d, sa: %d, cptr2: %d, rlist[cptr2]: %d, en: %d\n", 
            //        cptr, snode.rlist[cptr], sa, cptr2, snode.rlist[cptr2], en);
                  
            if (!map_done) {
               // int a_nb = nb;
               int r = 0; // row index
               int rr = 0; // block row index
               for (int row = 0; row < asnode.nrow; ++row) {
                  rr = row / nb;
                  r = asnode.rlist[row];
                  amap[r] = rr;
               }
               map_done = true;
            }

            int rptr = cptr;
            int ii = amap[snode.rlist[cptr]]; // first block row in anode

            int i = 0;
            for (i = cptr; i < m; i++) {
               int k = amap[snode.rlist[i]];
                     
               if (k != ii) {
                        
                  for (int kk = 0; kk < nc; ++kk) {
                     
                     int blkn = std::min(nb, n-(kk*nb));
                     
                     update_between_block(blkn, kk*nb, cptr, cptr2, rptr, i-1,
                                          snode, node,
                                          asnode, nodes[asnode.idx],
                                          work, rowmap, colmap);
                  }


                  ii = k;
                  rptr = i;
               }
            }

            for (int kk = 0; kk < nc; ++kk) {
                     
               int blkn = std::min(nb, n-(kk*nb));
                     
               update_between_block(blkn, kk*nb, cptr, cptr2, rptr, i-1, // cptr, m-1,
                                    snode, node,
                                    asnode, nodes[asnode.idx],
                                    work, rowmap, colmap);
            }

            cptr = cptr2 + 1; // move cptr
         }

         parent = stree[asnode.idx].parent; // move up the tree
      }
   }

} /* end of namespace spldlt */
