#pragma once

// #include "factorize.hxx"
#include "kernels/ldlt_app.hxx"

#if defined(SPLDLT_USE_STARPU)
#include "StarPU/kernels_indef.hxx"
using namespace spldlt::starpu;
#endif

namespace spldlt {

   template <typename T, int iblksz, typename Backup, typename PoolAlloc>
   void factor_indef_init() {
#if defined(SPLDLT_USE_STARPU)
      codelet_init_indef<T, iblksz, Backup, PoolAlloc>();
#endif
   }

   using namespace spldlt::ldlt_app_internal;

   /* Collection of routines for the factorization of symmetric
      indefinite problems */
   template<typename T,
            int iblksz,
            typename Backup,
            bool debug,
            typename Allocator
            >
   class FactorSymIndef {
      /// \{
      typedef typename std::allocator_traits<Allocator>::template rebind_alloc<int> IntAlloc;
      typedef typename std::allocator_traits<Allocator>::template rebind_alloc<T> TAlloc;
      // Block type
      typedef Block<T, iblksz, IntAlloc> BlockSpec;
      /// \}
   protected:
      /** \brief return number of columns in given block column */
      static
      inline int get_ncol(int blk, int n, int block_size) {
         return calc_blkn(blk, n, block_size);
      }
      /** \brief return number of rows in given block row */
      static
      inline int get_nrow(int blk, int m, int block_size) {
         return calc_blkn(blk, m, block_size);
      }

      /** \brief Print given matrix (for debug usage)
       *  \param m number of rows
       *  \param n number of columns
       *  \param perm[n] permutation of fully summed variables
       *  \param eliminated[n] status of fully summed variables
       *  \param a matrix values
       *  \param lda leading dimension of a
       */
   
      /* Tasks */

      /* Factor task: factorize a diagonal block 
         A_kk = P_k L_kk D_k L_kk^T P_k
       */ 
      static
      void factor_block_app_task(
            BlockSpec& dblk, int next_elim,
            int* perm, T* d,
            ColumnData<T,IntAlloc>& cdata, Backup& backup,
            struct cpu_factor_options& options,
            /*int const block_size,*/ 
            Workspace& work,
            Allocator const& alloc) {

         int blk = dblk.get_row();

#if defined(SPLDLT_USE_STARPU)

         insert_factor_block_app_task (
               dblk.get_hdl(), cdata[blk].get_hdl(),
               dblk.get_m(), dblk.get_n(), 
               blk,
               next_elim, perm, d,
               &cdata, &backup,
               &options, &work, &alloc);
         
#else
         bool abort=false;

         // BlockSpec dblk(blk, blk, m, n, cdata, &a[blk*block_size*lda+blk*block_size], lda, block_size);
         // if(debug) printf("Factor(%d)\n", blk);
         // Store a copy for recovery in case of a failed column
         dblk.backup(backup);
         int thread_num = 0;
         // Perform actual factorization
         int nelim = dblk.template factor<Allocator>(
               next_elim, perm, d, options, work, alloc
               );
         if(nelim<0) 
            abort=true;
         if(debug) printf("Factor(%d) nelim: %d\n", blk, nelim);

         // Init threshold check (non locking => task dependencies)
         cdata[blk].init_passed(nelim);      
#endif
      }
      
      /*  ApplyN task: apply pivot on sub-diagonal block passing the a
          posteriori pivot test. First perform column permutation form
          factorization of the block on the diagonal, then apply the
          following operation:
          
          L_ik = A_ik (L_kk D_k)^-T
          
      */
      static
      void applyN_block_app_task(
            BlockSpec& dblk, BlockSpec& rblk,
            // int blk, int iblk,
            // int const m, int const n, 
            // T* a, int const lda,
            ColumnData<T,IntAlloc>& cdata, Backup& backup,
            struct cpu_factor_options& options/*, int const block_size*/) {

         int blk = dblk.get_col();
         int iblk = rblk.get_row();

#if defined(SPLDLT_USE_STARPU)

         insert_applyN_block_app (
               dblk.get_hdl(), rblk.get_hdl(),
               cdata[blk].get_hdl(),
               dblk.get_m(), dblk.get_n(), 
               blk, iblk,
               &cdata, &backup, &options);

#else
         if(debug) printf("ApplyN(%d,%d)\n", iblk, blk);
         // BlockSpec dblk(blk, blk, m, n, cdata, &a[blk*block_size*lda+blk*block_size], lda, block_size);
         // BlockSpec rblk(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
         // Apply column permutation from factorization of dblk and in
         // the process, store a (permuted) copy for recovery in case of
         // a failed column
         rblk.apply_cperm_and_backup(backup);
         // Perform elimination and determine number of rows in block
         // passing a posteori threshold pivot test
         int blkpass = rblk.apply_pivot_app(dblk, options.u, options.small);
         // Update column's passed pivot count
         cdata[blk].update_passed(blkpass);
#endif
      }

      /*  ApplyN task: apply pivot on left-diagonal block passing the
          a posteriori pivot test. First perform row permutation form
          factorization of the block on the diagonal, then apply the
          following operation:
          
          L_kj = (L_kk D_k)^-1 A_kj
          
      */

      static 
      void applyT_block_app_task(
            BlockSpec& dblk, BlockSpec& cblk,
            // int blk, int jblk,
            // int const m, int const n, 
            // T* a, int const lda,
            ColumnData<T,IntAlloc>& cdata, Backup& backup,
            struct cpu_factor_options& options/*, int const block_size*/) {

         int blk = dblk.get_col();
         int jblk = cblk.get_col();

#if defined(SPLDLT_USE_STARPU)

         insert_applyT_block_app (
               dblk.get_hdl(), cblk.get_hdl(),
               cdata[blk].get_hdl(),
               dblk.get_m(), dblk.get_n(), 
               blk, jblk,
               &cdata, &backup, &options);

#else
         if(debug) printf("ApplyT(%d,%d)\n", blk, jblk);
         // BlockSpec dblk(blk, blk, m, n, cdata, &a[blk*block_size*lda+blk*block_size], lda, block_size);
         // BlockSpec cblk(blk, jblk, m, n, cdata, &a[jblk*block_size*lda+blk*block_size], lda, block_size);
         // Apply row permutation from factorization of dblk and in
         // the process, store a (permuted) copy for recovery in case of
         // a failed column
         cblk.apply_rperm_and_backup(backup);
         // Perform elimination and determine number of rows in block
         // passing a posteori threshold pivot test
         int blkpass = cblk.apply_pivot_app(
               dblk, options.u, options.small
               );
         // Update column's passed pivot count
         cdata[blk].update_passed(blkpass);
#endif
      }

      /* UpdateN task: peroform update of a block on the right of
         eliminated column. First, if we are in the block row or block
         column that we have jsut eliminated, restore the failed rows
         or columns from the backup. Then perform the following
         operation:
         
         A_ij = A_ij - L_ik D_k L_jk^T
         
       */

      static 
      void updateN_block_app_task(
            BlockSpec& isrc, BlockSpec& jsrc, BlockSpec& ublk,
            // int blk, int iblk, int jblk,
            // int const m, int const n, 
            // T* a, int const lda,
            ColumnData<T,IntAlloc>& cdata, Backup& backup, 
            // int const block_size,
            T const beta, T* upd, int const ldupd,
            Workspace& work) {

         int iblk = ublk.get_row();
         int jblk = ublk.get_col();
         int blk = jsrc.get_col();

#if defined(SPLDLT_USE_STARPU)

         insert_updateN_block_app(
               isrc.get_hdl(), jsrc.get_hdl(), ublk.get_hdl(), 
               ublk.get_m(), ublk.get_n(), 
               iblk, jblk, blk,
               &cdata, &backup,
               beta, upd, ldupd,
               &work, ublk.get_blksz());

#else
         if(debug) printf("UpdateN(%d,%d,%d)\n", iblk, jblk, blk);
         // int thread_num = omp_get_thread_num();
         int thread_num = 0;
         // BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
         // BlockSpec isrc(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
         // BlockSpec jsrc(jblk, blk, m, n, cdata, &a[blk*block_size*lda+jblk*block_size], lda, block_size);
         // If we're on the block col we've just eliminated, restore
         // any failed cols and release resources storing backup
         ublk.restore_if_required(backup, blk);
         // Perform actual update
         ublk.update(isrc, jsrc, work,
                     beta, upd, ldupd);
#endif
      }
      
      /* UpdateT task: peroform update of a block on the left of
         eliminated column. First, if we are in the block row or block
         column that we have jsut eliminated, restore the failed rows
         or columns from the backup. Then perform the following
         operation:

         A_ij = A_ij - L_ik D_k L_jk (updateNT)

         or

         A_ij = A_ij - L_ik^T D_k L_jk (updateTT)
       */

      static
      void updateT_block_app_task(            
            BlockSpec& isrc, BlockSpec& jsrc, BlockSpec& ublk,            
            // int blk, int iblk, int jblk,
            // int const m, int const n, 
            // T* a, int const lda,
            ColumnData<T,IntAlloc>& cdata, Backup& backup, 
            // int const block_size,
            Workspace& work) {

         int iblk = ublk.get_row();
         int jblk = ublk.get_col();
         int blk = jsrc.get_row();

         int isrc_row = isrc.get_row();
         int isrc_col = isrc.get_col();

#if defined(SPLDLT_USE_STARPU)

         insert_updateT_block_app(
               isrc.get_hdl(), jsrc.get_hdl(), ublk.get_hdl(), 
               ublk.get_m(), ublk.get_n(),
               isrc_row, isrc_col,
               iblk, jblk, blk,
               &cdata, &backup,
               &work, ublk.get_blksz());

#else
         if(debug) printf("UpdateT(%d,%d,%d)\n", iblk, jblk, blk);
         // int thread_num = omp_get_thread_num();
         int thread_num = 0;
         // BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
         // int isrc_row = (blk<=iblk) ? iblk : blk;
         // int isrc_col = (blk<=iblk) ? blk : iblk;
         // BlockSpec isrc(isrc_row, isrc_col, m, n, cdata, &a[isrc_col*block_size*lda+isrc_row*block_size], lda,
         //                block_size);
         // BlockSpec jsrc(blk, jblk, m, n, cdata, &a[jblk*block_size*lda+blk*block_size], lda, block_size);

         // If we're on the block row we've just eliminated, restore
         // any failed rows and release resources storing backup
         ublk.restore_if_required(backup, blk);
         // Perform actual update
         ublk.update(isrc, jsrc, work);
#endif
      }
      
      /* adjust task
       */
      static 
      void adjust_task(
            BlockSpec& dblk,
            int& next_elim,
            ColumnData<T,IntAlloc>& cdata) {
         
         int blk = dblk.get_col();

#if defined(SPLDLT_USE_STARPU)
         insert_adjust(
               cdata[blk].get_hdl(), blk,
               &next_elim, &cdata);
#else
         // Adjust column once all applys have finished and we know final
         // number of passed columns.
         if(debug) printf("Adjust(%d)\n", blk);
         cdata[blk].adjust(next_elim);
#endif

      }

      /* update_contrib task: Udpate a block as part of the
         contribution block
       */
      static
      void udpate_contrib_task(
            BlockSpec& isrc, BlockSpec& jsrc, BlockSpec& ublk,
            T const beta, T* upd_ij, int const ldupd,
            Workspace work
            ) {

         int iblk =  ublk.get_row();
         int jblk =  ublk.get_col();
         int blk = jsrc.get_row();

         if(debug) printf("FormContrib(%d,%d,%d)\n", iblk, jblk, blk);

         ublk.form_contrib(
               isrc, jsrc, work, beta, upd_ij, ldupd
               );

      }

      /*
        Factorization routine for indefinite matrices implementing an
        APTP strategy. Failed entires are left in place.
      */
      static
      int factorize_indef_app (
            int const m, int const n, int* perm, T* a,
            int const lda, T* d, ColumnData<T,IntAlloc>& cdata, Backup& backup,
            struct cpu_factor_options& options, int const block_size,
            T const beta, T* upd, int const ldupd, std::vector<spral::ssids::cpu::Workspace>& work,
            Allocator const& alloc, int const from_blk=0) {
         typedef Block<T, iblksz, IntAlloc> BlockSpec;
         // typedef ColumnData<T,IntAlloc> ColumnDataSpec;

         int const nblk = calc_nblk(n, block_size);
         int const mblk = calc_nblk(m, block_size);

         /* Setup */
         int next_elim = from_blk*block_size;      
      
         int num_blocks = nblk*mblk;
         std::vector<BlockSpec> blocks;
         blocks.reserve(num_blocks);

         for(int jblk=0; jblk<nblk; jblk++) {
#if defined(SPLDLT_USE_STARPU)
            // register symbolic handle for each block column
            cdata[jblk].register_handle();
#endif            
            for(int iblk=0; iblk<mblk; iblk++) {
               // Create and insert block at the end (column-wise storage)
               blocks.emplace_back(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
               // alternativel store pointer
               // blocks[jblk*mblk + iblk] = new BlockSpec(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
#if defined(SPLDLT_USE_STARPU)
               // register handle for block (iblk, jblk)
               blocks[jblk*mblk+iblk].register_handle(); 
#endif
            }
         }

         int thread_num = 0;
         
         /* Inner loop - iterate over block columns */
         // try {
         for(int blk=from_blk; blk<nblk; blk++) {
            /*if(debug) {
              printf("Bcol %d:\n", blk);
              print_mat(mblk, nblk, m, n, blkdata, cdata, lda);
              }*/

            // Factor diagonal: depend on perm[blk*block_size] as we init npass
            // {  
            // BlockSpec dblk(blk, blk, m, n, cdata, a, lda, block_size);
            
            // Factorize block on diagonal

            // BlockSpec dblk(blk, blk, m, n, cdata, &a[blk*block_size*lda+blk*block_size], lda, block_size);
           
            factor_block_app_task(
                  blocks[blk*(mblk+1)] /*dblk*/, next_elim,
                  perm, d,
                  cdata, backup,
                  options/*, block_size*/, work[thread_num], alloc);

#if defined(SPLDLT_USE_STARPU)
            starpu_task_wait_for_all();
#endif

            // DEBUG
            // {
            //    if(debug) printf("Factor(%d)\n", blk);
            //    BlockSpec dblk(blk, blk, m, n, cdata, &a[blk*block_size*lda+blk*block_size], lda, block_size);
            //    // Store a copy for recovery in case of a failed column
            //    dblk.backup(backup);
            //    // Perform actual factorization
            //    int nelim = dblk.template factor<Allocator>(
            //          next_elim, perm, d, options, work[0], /*work,*/ alloc
            //          );
            //    if(nelim<0) return nelim;
            //    // Init threshold check (non locking => task dependencies)
            //    cdata[blk].init_passed(nelim);
            // }
            // END DEBUG

            // }
            
            // Loop over off-diagonal blocks applying pivot
            for(int jblk=0; jblk<blk; jblk++) {

               // BlockSpec cblk(blk, jblk, m, n, cdata, &a[jblk*block_size*lda+blk*block_size], lda, block_size);               
               // Apply factorization on uneliminated entries of
               // left-diagonal block
               applyT_block_app_task(
                     /*dblk*/ blocks[blk*(mblk+1)], /*cblk*/blocks[jblk*mblk+blk],
                     cdata, backup,
                     options);

#if defined(SPLDLT_USE_STARPU)
               starpu_task_wait_for_all();
#endif

               // DEBUG
               // if(debug) printf("ApplyT(%d,%d)\n", blk, jblk);
               // BlockSpec dblk(blk, blk, m, n, cdata, &a[blk*block_size*lda+blk*block_size], lda, block_size);
               // BlockSpec cblk(blk, jblk, m, n, cdata, &a[jblk*block_size*lda+blk*block_size], lda, block_size);
               // // Apply row permutation from factorization of dblk and in
               // // the process, store a (permuted) copy for recovery in case of
               // // a failed column
               // cblk.apply_rperm_and_backup(backup);
               // // Perform elimination and determine number of rows in block
               // // passing a posteori threshold pivot test
               // int blkpass = cblk.apply_pivot_app(
               //       dblk, options.u, options.small
               //       );
               // // Update column's passed pivot count
               // cdata[blk].update_passed(blkpass);
               // END DEBUG
            }
            for(int iblk=blk+1; iblk<mblk; iblk++) {

               // BlockSpec rblk(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
               // Apply factorization on sub-diagonal block
               applyN_block_app_task(
                     /*dblk*/ blocks[blk*(mblk+1)], /*rblk*/ blocks[blk*mblk+iblk],
                     cdata, backup,
                     options);

#if defined(SPLDLT_USE_STARPU)
               starpu_task_wait_for_all();
#endif

               // DEBUG
               // if(debug) printf("ApplyN(%d,%d)\n", iblk, blk);
               // BlockSpec dblk(blk, blk, m, n, cdata, &a[blk*block_size*lda+blk*block_size], lda, block_size);
               // BlockSpec rblk(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
               // // Apply column permutation from factorization of dblk and in
               // // the process, store a (permuted) copy for recovery in case of
               // // a failed column
               // rblk.apply_cperm_and_backup(backup);
               // // Perform elimination and determine number of rows in block
               // // passing a posteori threshold pivot test
               // int blkpass = rblk.apply_pivot_app(dblk, options.u, options.small);
               // // Update column's passed pivot count
               // cdata[blk].update_passed(blkpass);
               // END DEBUG
            }

            // Adjust column once all applys have finished and we know final
            // number of passed columns.
            adjust_task(/* dblk*/blocks[blk*(mblk+1)], next_elim, cdata);

#if defined(SPLDLT_USE_STARPU)
            starpu_task_wait_for_all();
#endif

            // DEBUG
            // if(debug) printf("Adjust(%d)\n", blk);
            // cdata[blk].adjust(next_elim);
            // END DEBUG

            // Update uneliminated columns
            for(int jblk=0; jblk<blk; jblk++) {

               BlockSpec jsrc(blk, jblk, m, n, cdata, &a[jblk*block_size*lda+blk*block_size], lda, block_size);

               for(int iblk=jblk; iblk<mblk; iblk++) {

                  int isrc_row = (blk<=iblk) ? iblk : blk;
                  int isrc_col = (blk<=iblk) ? blk : iblk;
                  BlockSpec isrc(isrc_row, isrc_col, m, n, cdata, &a[isrc_col*block_size*lda+isrc_row*block_size], lda,
                                 block_size);
                  BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);

                  // Update uneliminated entries in blocks on the left
                  // of current block column
                  updateT_block_app_task(
                        // isrc, jsrc, ublk,
                        blocks[isrc_col*mblk+isrc_row], blocks[jblk*mblk+blk], 
                        blocks[jblk*mblk+iblk],
                        cdata, backup, 
                        work[0]);

#if defined(SPLDLT_USE_STARPU)
            starpu_task_wait_for_all();
#endif

                  // DEBUG
                  // if(debug) printf("UpdateT(%d,%d,%d)\n", iblk, jblk, blk);
                  // int thread_num = omp_get_thread_num();
                  // BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
                  // int isrc_row = (blk<=iblk) ? iblk : blk;
                  // int isrc_col = (blk<=iblk) ? blk : iblk;
                  // BlockSpec isrc(isrc_row, isrc_col, m, n, cdata, &a[isrc_col*block_size*lda+isrc_row*block_size], lda,
                  //                block_size);
                  // BlockSpec jsrc(blk, jblk, m, n, cdata, &a[jblk*block_size*lda+blk*block_size], lda, block_size);
                  // // If we're on the block row we've just eliminated, restore
                  // // any failed rows and release resources storing backup
                  // ublk.restore_if_required(backup, blk);
                  // // Perform actual update
                  // ublk.update(isrc, jsrc, work[0]);
                  // END DEBUG
               }
            }
            for(int jblk=blk; jblk<nblk; jblk++) {

               // Source block
               BlockSpec jsrc(jblk, blk, m, n, cdata, &a[blk*block_size*lda+jblk*block_size], lda, block_size);

               for(int iblk=jblk; iblk<mblk; iblk++) {

                  // Source block
                  BlockSpec isrc(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
                  // Destination block
                  BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);

                  // Update blocks on the right of the current block
                  // column
                  updateN_block_app_task(
                        // isrc, jsrc, ublk,
                        blocks[blk*mblk+iblk], blocks[blk*mblk+jblk],
                        blocks[jblk*mblk+iblk],
                        cdata, backup,
                        beta, upd, ldupd,
                        work[thread_num]);

#if defined(SPLDLT_USE_STARPU)
                  starpu_task_wait_for_all();
#endif

               }
            }

            // Handle update to contribution block, if required
            if(upd && mblk>nblk) {
               int uoffset = std::min(nblk*block_size, m) - n;
               T *upd2 = &upd[uoffset*(ldupd+1)];
               for(int jblk=nblk; jblk<mblk; ++jblk)
                  for(int iblk=jblk; iblk<mblk; ++iblk) {
                     T* upd_ij = &upd2[(jblk-nblk)*block_size*ldupd + 
                                       (iblk-nblk)*block_size];
                     {

                        BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
                        BlockSpec isrc(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
                        BlockSpec jsrc(jblk, blk, m, n, cdata, &a[blk*block_size*lda+jblk*block_size], lda, block_size);

                        udpate_contrib_task(
                              // isrc, jsrc, ublk,
                              blocks[blk*mblk+iblk], blocks[blk*mblk+jblk],
                              blocks[jblk*mblk+iblk],
                              beta, upd_ij, ldupd,
                              work[thread_num]
                              );
                     }
                  }
            }
         }
         // } catch(std::bad_alloc const&) {
         //    return Flag::ERROR_ALLOCATION;
         // } catch(SingularError const&) {
         //    return Flag::ERROR_SINGULAR;
         // }

         /*if(debug) {
           printf("PostElim:\n");
           print_mat(mblk, nblk, m, n, blkdata, cdata, lda);
           }*/

         /* FIXME: 
            
            If we want to return next_elim: need to wait for
            adjust_task to be completed. Alternatively, remove the
            return and retreive next_elim upon completion of
            adjsut_task.
            
          */ 
#if defined(SPLDLT_USE_STARPU)
         starpu_task_wait_for_all();
#endif

         return next_elim;
      }

      /* Permute failed entries to the back of the matrix
       */
      static
      void permute_failed(
            int m, int n, int *perm, T *a, int lda,
            int num_elim, 
            ColumnData<T,IntAlloc>& cdata, int block_size,
            Allocator const& alloc) {

         int nblk = calc_nblk(n, block_size);
         int mblk = calc_nblk(m, block_size);

         // Permute fail entries to the back of the matrix
         std::vector<int, IntAlloc> failed_perm(n-num_elim, alloc);
         for(int jblk=0, insert=0, fail_insert=0; jblk<nblk; jblk++) {
            cdata[jblk].move_back(
                  get_ncol(jblk, n, block_size), &perm[jblk*block_size],
                  &perm[insert], &failed_perm[fail_insert]
                  );
            insert += cdata[jblk].nelim;
            fail_insert += get_ncol(jblk, n, block_size) - cdata[jblk].nelim;
         }
         for(int i=0; i<n-num_elim; ++i)
            perm[num_elim+i] = failed_perm[i];

         // Extract failed entries of a
         int nfail = n-num_elim;
         std::vector<T, TAlloc> failed_diag(nfail*n, alloc);
         std::vector<T, TAlloc> failed_rect(nfail*(m-n), alloc);
         for(int jblk=0, jfail=0, jinsert=0; jblk<nblk; ++jblk) {
            // Diagonal part
            for(int iblk=jblk, ifail=jfail, iinsert=jinsert; iblk<nblk; ++iblk) {
               copy_failed_diag(
                     get_ncol(iblk, n, block_size), get_ncol(jblk, n, block_size),
                     cdata[iblk], cdata[jblk],
                     &failed_diag[jinsert*nfail+ifail],
                     &failed_diag[iinsert*nfail+jfail],
                     &failed_diag[num_elim*nfail+jfail*nfail+ifail],
                     nfail, &a[jblk*block_size*lda+iblk*block_size], lda
                     );
               iinsert += cdata[iblk].nelim;
               ifail += get_ncol(iblk, n, block_size) - cdata[iblk].nelim;
            }
            // Rectangular part
            // (be careful with blocks that contain both diag and rect parts)
            copy_failed_rect(
                  get_nrow(nblk-1, m, block_size), get_ncol(jblk, n, block_size),
                  get_ncol(nblk-1, n, block_size), cdata[jblk],
                  &failed_rect[jfail*(m-n)+(nblk-1)*block_size-n], m-n,
                  &a[jblk*block_size*lda+(nblk-1)*block_size], lda
                  );
            for(int iblk=nblk; iblk<mblk; ++iblk) {
               copy_failed_rect(
                     get_nrow(iblk, m, block_size),
                     get_ncol(jblk, n, block_size), 0, cdata[jblk],
                     &failed_rect[jfail*(m-n)+iblk*block_size-n], m-n,
                     &a[jblk*block_size*lda+iblk*block_size], lda
                     );
            }
            jinsert += cdata[jblk].nelim;
            jfail += get_ncol(jblk, n, block_size) - cdata[jblk].nelim;
         }

         // Move data up
         for(int jblk=0, jinsert=0; jblk<nblk; ++jblk) {
            // Diagonal part
            for(int iblk=jblk, iinsert=jinsert; iblk<nblk; ++iblk) {
               move_up_diag(
                     cdata[iblk], cdata[jblk], &a[jinsert*lda+iinsert],
                     &a[jblk*block_size*lda+iblk*block_size], lda
                     );
               iinsert += cdata[iblk].nelim;
            }
            // Rectangular part
            // (be careful with blocks that contain both diag and rect parts)
            move_up_rect(
                  get_nrow(nblk-1, m, block_size),
                  get_ncol(nblk-1, n, block_size), cdata[jblk],
                  &a[jinsert*lda+(nblk-1)*block_size],
                  &a[jblk*block_size*lda+(nblk-1)*block_size], lda
                  );
            for(int iblk=nblk; iblk<mblk; ++iblk)
               move_up_rect(
                     get_nrow(iblk, m, block_size), 0, cdata[jblk],
                     &a[jinsert*lda+iblk*block_size],
                     &a[jblk*block_size*lda+iblk*block_size], lda
                     );
            jinsert += cdata[jblk].nelim;
         }
         
         // Store failed entries back to correct locations
         // Diagonal part
         for(int j=0; j<n; ++j)
            for(int i=std::max(j,num_elim), k=i-num_elim; i<n; ++i, ++k)
               a[j*lda+i] = failed_diag[j*nfail+k];
         // Rectangular part
         T* arect = &a[num_elim*lda+n];
         for(int j=0; j<nfail; ++j)
            for(int i=0; i<m-n; ++i)
               arect[j*lda+i] = failed_rect[j*(m-n)+i];
      
      }

   public:
      /*
        Factorization routine for indefinite matrices implementing an
        APTP strategy. After the factorization, fail entries are permuted
        to the back of the matrix.
      */
      static
      int ldlt_app(int m, int n, int *perm, 
                   T *a, int lda, T *d, 
                   Backup& backup, 
                   struct cpu_factor_options& options/*, PivotMethod pivot_method*/, int block_size, T beta, T* upd, int ldupd, 
                   std::vector<spral::ssids::cpu::Workspace>& work, Allocator const& alloc=Allocator()) {

         /* Sanity check arguments */
         if(m < n) return -1;
         if(lda < n) return -4;

         typedef typename std::allocator_traits<Allocator>::template rebind_alloc<int> IntAlloc;
         typedef typename std::allocator_traits<Allocator>::template rebind_alloc<T> TAlloc;

         int num_elim; // return value

         /* Initialize useful quantities: */
         int nblk = calc_nblk(n, block_size);
         int mblk = calc_nblk(m, block_size);

         /* Temporary workspaces */
         ColumnData<T, IntAlloc> cdata(n, block_size, IntAlloc(alloc));

         // factorize matrix and leave fail entries in place
         num_elim = factorize_indef_app (
               m, n, perm, a, lda, d, cdata, backup,
               options, block_size,
               beta, upd, ldupd, work,
               alloc);

         // By default function calls are asynchronous, so we put a
         // barrier and wait for the task-based factorization to be
         // completed
#if defined(SPLDLT_USE_STARPU)
         starpu_task_wait_for_all();
#endif

         
         // realease all memory used for backup
         backup.release_all_memory(); 
 
         // Permute failed entries to end
         if (num_elim < n)
            permute_failed (
                  m, n, perm, a, lda,
                  num_elim, 
                  cdata, block_size,
                  alloc);

         // if(debug) {
         //    std::vector<bool> eliminated(n);
         //    for(int i=0; i<num_elim; i++) eliminated[i] = true;
         //    for(int i=num_elim; i<n; i++) eliminated[i] = false;
         //    printf("FINAL:\n");
         //    print_mat(m, n, perm, eliminated, a, lda);
         // }

         return num_elim;
      }
   };
}
