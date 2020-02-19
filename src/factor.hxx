/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "sylver_ciface.hxx"
#include "tasks.hxx"

namespace spldlt {
   
   ////////////////////////////////////////////////////////////
   // factor_front_posdef
   //

   /// @brief Perform the Cholesky factorization of a frontal matrix
   template <typename T, typename PoolAlloc>
   void factor_front_posdef(
         NumericFront<T, PoolAlloc> &node,
         sylver::options_t const& options,
         std::vector<sylver::inform_t>& worker_stats
         ) {

      std::string context = "factor_front_posdef";      
      sylver::SymbolicFront& snode = node.symb(); // Symbolic front data 

      // Extract useful information about node
      int const m = node.get_nrow(); // Number of rows in frontal matrix
      int const n = node.get_ncol(); // Number of columns in frontal matrix
      int const lda = node.get_ldl(); // Leading dimensions
      T *lcol = node.lcol; // Pointer to L factor 
      int const ldcontrib = m-n; // Dimension of contrib block
      int const blksz = options.nb; // Block size
      int const nr = node.get_nr(); // Number of block rows
      int const nc = node.get_nc(); // Number of block columns
   
      // std::cout << "[" << context << "]" << std::endl;

      for(int j = 0; j < nc; ++j) {

         // std::cout << "[" << context << "]" << " j = " << j << std::endl;

         int blkn = std::min(blksz, n-j*blksz);
         int blkm = std::min(blksz, m-j*blksz);
         // Factor block         
         factor_block_task(node, j, FACTOR_PRIO, worker_stats);

         // #if defined(SPLDLT_USE_STARPU)
         //             starpu_task_wait_for_all();
         // #endif

         // Column Solve Tasks
         for(int i = j+1; i < nr; ++i) {

            int blkm = std::min(blksz, m - i*blksz);

            solve_block_task(
                  node, j, i,
                  &lcol[j*blksz*(lda+1)], lda,
                  &lcol[(j*blksz*lda) + (i*blksz)], lda,
                  SOLVE_PRIO);

         }

         // #if defined(SPLDLT_USE_STARPU)
         //             starpu_task_wait_for_all();
         // #endif

         // Schur Update Tasks (fully-summed)
         for(int k = j+1; k < nc; ++k) {

            int blkk = std::min(blksz, n - k*blksz);

            for(int i = k;  i < nr; ++i) {
               
               int blkm = std::min(blksz, m - i*blksz);
               
               // int cbm = (i*blksz < n) ? std::min((i+1)*blksz,m)-n : blkm;
               // int cbm = (i*blksz < n) ? blkm+(i*blksz)-n : blkm;
               // int cbn = std::min(blksz, m-k*blksz)-blkk;
               // T *upd = nullptr;
                  
               // if (contrib)
                  // upd = (i*blksz < n) ? contrib : &contrib[(i*blksz)-n];
               
               // TODO fix STF version
               update_block_task(snode, node, j, i, k,
                                 &lcol[ (k*blksz*lda) + (i*blksz)], lda,
                                 &lcol[(j*blksz*lda) + (i*blksz)], lda, 
                                 &lcol[(j*blksz*lda) + (k*blksz)], lda,
                                 blksz, UPDATE_PRIO);

               // #if defined(SPLDLT_USE_STARPU)
               //                starpu_task_wait_for_all();
               // #endif
            }
         }

         // #if defined(SPLDLT_USE_STARPU)
         //          starpu_task_wait_for_all();
         // #endif

         // Contrib Schur complement update (contribution block)
         if (ldcontrib>0) {

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

      // Record information
      node.ndelay_out(0); // no delays
   }

}
