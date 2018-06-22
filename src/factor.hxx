#pragma once

#include "tasks.hxx"

namespace spldlt {
   
   ////////////////////////////////////////////////////////////
   // factor_front_posdef
   //
   // Factorize a supernode using a multifrontal mode
   template <typename T, typename PoolAlloc>
   void factor_front_posdef(
         NumericFront<T, PoolAlloc> &node,
         struct cpu_factor_options const& options
         ) {

      SymbolicFront& snode = node.symb; 

      /* Extract useful information about node */
      int m = node.get_nrow();
      int n = node.get_ncol();
      int lda = align_lda<T>(m);
      T *lcol = node.lcol;
      T *contrib = node.contrib;
      int ldcontrib = m-n;

      int blksz = options.cpu_block_size;
      int nr = node.get_nr(); // number of block rows
      int nc = node.get_nc(); // number of block columns
   
      // printf("[factor_front_posdef]\n");

      for(int j = 0; j < nc; ++j) {

         int blkn = std::min(blksz, n - j*blksz);
         
         /* Diagonal Block Factorization Task */

         int blkm = std::min(blksz, m - j*blksz);
         
         // factorize_diag_block(
         //       blkm, blkn, 
         //       &lcol[j*blksz*(lda+1)], lda,
         //       contrib, ldcontrib,
         //       j==0);

         factor_diag_block_task(
               node, j, FACTOR_PRIO);

         // #if defined(SPLDLT_USE_STARPU)
         //             starpu_task_wait_for_all();
         // #endif

         /* Column Solve Tasks */
         for(int i = j+1; i < nr; ++i) {

            int blkm = std::min(blksz, m - i*blksz);

            // printf("[factorize_node_posdef_mf] contrib start: %d\n", (i*blksz)-n);
            // TODO fix STF version
            solve_block_task(
                  node, j, i,
                  &lcol[j*blksz*(lda+1)], lda,
                  &lcol[(j*blksz*lda) + (i*blksz)], lda,
                  SOLVE_PRIO);

            // solve_block(blkm, blkn, 
            //             &lcol[j*blksz*(lda+1)], lda, 
            //             &lcol[(j*blksz*lda) + (i*blksz)], lda,
            //             (contrib) ? &contrib[(i*blksz)-n] : nullptr, ldcontrib,
            //             j==0, 
            //             blksz);

            // #if defined(SPLDLT_USE_STARPU)
            //             starpu_task_wait_for_all();
            // #endif

         }

         // #if defined(SPLDLT_USE_STARPU)
         //             starpu_task_wait_for_all();
         // #endif

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
               // TODO fix STF version
               update_block_task(snode, node, j, i, k,
                                 &lcol[ (k*blksz*lda) + (i*blksz)], lda,
                                 &lcol[(j*blksz*lda) + (i*blksz)], lda, 
                                 &lcol[(j*blksz*lda) + (k*blksz)], lda,
                                 blksz, UPDATE_PRIO);

               // #if defined(SPLDLT_USE_STARPU)
               //                starpu_task_wait_for_all();
               // #endif
               
               // update_block(blkm, blkk, &lcol[ (k*blksz*lda) + (i*blksz)], lda,
               //              blkn,
               //              &lcol[(j*blksz*lda) + (i*blksz)], lda, 
               //              &lcol[(j*blksz*lda) + (k*blksz)], lda,
               //              upd, ldcontrib,
               //              cbm, cbn,
               //              j==0,
               //              blksz);

            }
         }

         // #if defined(SPLDLT_USE_STARPU)
         //          starpu_task_wait_for_all();
         // #endif

         /* Contrib Schur complement update: external */
         // if (contrib) {
         if (ldcontrib>0) {
            // printf("[factorize_node_posdef_mf] node: %d\n", snode.idx);
            // printf("[factorize_node_posdef_mf] nc: %d, nr: %d\n", nc, nr);
            for (int k = nc; k < nr; ++k) {
               
               int blkk = std::min(blksz, m - k*blksz);

               for (int i = k;  i < nr; ++i) {
               
                  int blkm = std::min(blksz, m - i*blksz);

                  update_contrib_task(snode, node,
                                      j, i, k, blksz, 
                                      UPDATE_PRIO);

                  // update_block(
                  //       blkm, blkk,
                  //       &contrib[((k*blksz-n)*ldcontrib) + (i*blksz)-n], ldcontrib,
                  //       blkn,
                  //       &lcol[(j*blksz*lda) + (i*blksz)], lda, 
                  //       &lcol[(j*blksz*lda) + (k*blksz)], lda,
                  //       j==0);
                  
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

      // We've eliminated all columns 
      node.nelim = n;

      /* Record information */
      node.ndelay_out = 0; // no delays
   }

}
