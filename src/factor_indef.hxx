/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// Sylver
#include "sylver_ciface.hxx"
#include "kernels/ldlt_app.hxx"
#include "kernels/factor_indef.hxx"
#include "kernels/assemble.hxx"
#include "factor_failed.hxx"
#include "tasks/indef.hxx"
#include "tasks/factor_failed.hxx"
#if defined(SPLDLT_USE_STARPU)
#include "StarPU/codelets.hxx"
#include "StarPU/kernels_indef.hxx"
#endif
#include "Tile.hxx"

// SSIDS
#include "ssids/cpu/kernels/ldlt_nopiv.hxx"

namespace sylver {
namespace spldlt {

   /// @brief Factor subtree
   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void factor_subtree_indef(
         NumericFront<T, PoolAlloc>& root,
         int n,
         int nnodes,
         std::vector<NumericFront<T,PoolAlloc>>& fronts,
         void** child_contrib,
         FactorAlloc& factor_alloc,
         PoolAlloc& pool_alloc,
         T const* aval,
         spral::ssids::cpu::Workspace& work,
         sylver::options_t& options
         ) {

      int const least_desc = root.symb.least_desc;
      int const rootidx = root.symb.idx;
      
      // Traverse subtree from least descenent to root
      for (int ni = least_desc; ni <= rootidx; ++ni) {

         NumericFront<T, PoolAlloc>& front = fronts[ni];         
         // Call alloc instead of activate because we don't need to
         // register StarPU handles
         alloc_front_indef(front, child_contrib, factor_alloc);
         // Assemble fully-summed from original matrix
         init_node(front, aval);
         // Assemble fully-summed from child nodes contrib 
         assemble_notask(n, front, child_contrib, pool_alloc);  
         // Compute factors and update contrib block 
         factor_front_indef_notask(front, options, work, pool_alloc);
         // Assemble cotrib block from child nodes contrib
         assemble_contrib_notask(front, child_contrib);
         // Cleanup data structures and release temp memory
         for (auto* child=front.first_child; child!=NULL; child=child->next_child) {
            fini_node(*child);
         }
      }

      
   }

   ////////////////////////////////////////////////////////////
   // factor_front_indef_notask
   
   /// @brief Perform the LDLT factorization of front as in
   /// factor_front_indef but using a STF model.
   template <typename T, typename PoolAlloc>
   void factor_front_indef_notask(
         sylver::options_t& options,
         PoolAlloc& pool_alloc,
         NumericFront<T, PoolAlloc>& front,
         // spral::ssids::cpu::Workspace& work,
         std::vector<spral::ssids::cpu::Workspace>& workspaces,
         sylver::inform_t& stats) {

      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> IntAlloc;

      // Start from first column
      front.nelim(0); // TODO add parameter from;

      // LDLT with APTP strategy
      if (options.pivot_method==sylver::PivotMethod::app_block) {

         typedef spldlt::ldlt_app_internal::CopyBackup<T, PoolAlloc> Backup;
         bool const debug = false;
         typedef
            FactorSymIndef<T, INNER_BLOCK_SIZE, Backup, debug, PoolAlloc>
            FactorSymIndefSpec;

         int& nelim = front.nelim(); 
            
         // Factor front
         FactorSymIndefSpec::factor_front_indef_app_notask(
               front, options, stats, workspaces[0], pool_alloc, nelim);
         
         if (front.nelim() < front.ncol()) { 
            // Some columns remain uneliminated after the first pass
            
            spldlt::ldlt_app_internal::
               ColumnData<T, IntAlloc> &cdata = *front.cdata;

            // Realease backup and permute failed at the back of the
            // front
            Backup& backup = *front.backup; 
            backup.release_all_memory(); // Release backup associated
                                         // to this front
            // Permute failed entries at the back of the front
            FactorSymIndefSpec::permute_failed(
                  front.nrow(), front.ncol(), front.perm,
                  front.lcol, front.ldl(),
                  front.nelim(), cdata, front.blksz(),
                  pool_alloc);
         }
         
      } // PivotMethod::app_block

      // Process uneliminated columns 
      factor_front_indef_failed(front, workspaces, options, stats);
      
   }

   
   ////////////////////////////////////////////////////////////
   // factor_front_indef

   /// @brief Perform the LDLT factorization of a front
   template <typename T, typename PoolAlloc>
   void factor_front_indef(
         NumericFront<T, PoolAlloc> &node,
         std::vector<spral::ssids::cpu::Workspace>& workspaces,
         PoolAlloc& pool_alloc,
         sylver::options_t& options,
         std::vector<sylver::inform_t>& worker_stats) {

      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> IntAlloc;
      
      /* Extract useful information about node */
      int m = node.nrow();
      int n = node.ncol();
      size_t ldl = align_lda<T>(m);
      T *lcol = node.lcol;
      T *d = &node.lcol[n*ldl];
      int *perm = node.perm;

      int ldld = m;

      int nelim = 0;

      // printf("[factor_front_indef] m = %d, n = %d\n", node.get_nrow(), node.get_ncol());

      int blksz = node.blksz();
      // node.nelim = nelim;      
      bool const debug = false;
      T *upd = nullptr;

      // Factorize from first column
      node.nelim(0); // TODO add parameter from;

      if (options.pivot_method==sylver::PivotMethod::app_block) {

         typedef spldlt::ldlt_app_internal::CopyBackup<T, PoolAlloc> Backup;
         typedef FactorSymIndef<T, INNER_BLOCK_SIZE, Backup, debug, PoolAlloc> FactorSymIndefSpec;

         // Factor fully-summed columns and form contrib blocks
         FactorSymIndefSpec::factor_front_indef_app(
               node, options, worker_stats, 0.0, upd, 0, workspaces, pool_alloc,
               node.nelim());

         // Release backup and permute failed columns
         FactorSymIndefSpec::release_permute_failed_task(
               node, pool_alloc);
      }

      // Process uneliminated columns 
      factor_front_indef_failed_task(
            node, workspaces, options, worker_stats);

// #if defined(SPLDLT_USE_STARPU)
//       ColumnData<T, IntAlloc> &cdata = *node.cdata;
//       int const nblk = calc_nblk(n, blksz);
//       insert_factor_sync(cdata[nblk-1].get_hdl(), node);
//       // insert_factor_sync(node.contrib_hdl, node);
// #endif
   }

   template <typename T, int iblksz, typename Backup, typename PoolAlloc>
   void factor_indef_init() {
#if defined(SPLDLT_USE_STARPU)
      sylver::spldlt::starpu::codelet_init_indef<T, iblksz, Backup, PoolAlloc>();
      // spldlt::starpu::codelet_init<T, FactorAllocator, PoolAllocator>();
      // spldlt::starpu::codelet_init_indef<T, iblksz, Backup, PoolAllocator>();
      // spldlt::starpu::codelet_init_factor_indef<T, PoolAllocator>();
#endif
   }

   using namespace spldlt::ldlt_app_internal;

   /* Collection of routines for the factorization of symmetric
      indefinite problems */
   template<typename T,
            int iblksz,
            typename Backup,
            bool debug,
            typename Allocator>
   class FactorSymIndef {
      /// \{
      typedef typename std::allocator_traits<Allocator>::template rebind_alloc<int> IntAlloc;
      typedef typename std::allocator_traits<Allocator>::template rebind_alloc<T> TAlloc;
      // Block type
      typedef spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> BlockSpec;
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

      ////////////////////////////////////////
      // Task priorities

#if defined(SPLDLT_USE_GPU)

      // Heteroprio
      static const int FACTOR_APP_PRIO   = 0;
      static const int ADJUST_APP_PRIO   = 0;
      static const int APPLYN_APP_PRIO   = 1;
      static const int RESTORE_APP_PRIO  = 1;
      static const int UPDATEN_APP_PRIO  = 2;
      static const int UPDATEC_APP_PRIO  = 2;
      static const int APPLYT_APP_PRIO   = 3;
      static const int UPDATET_APP_PRIO  = 3;
#else

      // LWS
      static const int FACTOR_APP_PRIO   = 3;
      static const int ADJUST_APP_PRIO   = 3;
      static const int APPLYN_APP_PRIO   = 2;
      static const int RESTORE_APP_PRIO  = 2;
      static const int UPDATEN_APP_PRIO  = 1;
      static const int UPDATEC_APP_PRIO  = 1;
      static const int APPLYT_APP_PRIO   = 0;
      static const int UPDATET_APP_PRIO  = 0;

#endif
      ////////////////////////////////////////
      // Tasks

      /* Factor task: factorize a diagonal block 
         A_kk = P_k L_kk D_k L_kk^T P_k
       */ 
      static
      void factor_block_app_task(
            BlockSpec& dblk, int& next_elim,
            int* perm, T* d,
            ColumnData<T,IntAlloc>& cdata, Backup& backup,
            sylver::options_t& options, std::vector<sylver::inform_t>& worker_stats,
            std::vector<spral::ssids::cpu::Workspace>& work,
            Allocator const& alloc) {

         int blk = dblk.get_row();

#if defined(SPLDLT_USE_STARPU)

         // printf("[factor_block_app_task] m = %d, n = %d\n", dblk.get_m(), dblk.get_n());

         sylver::spldlt::starpu::insert_factor_block_app(
               dblk.get_hdl(), cdata[blk].get_d_hdl(), cdata[blk].get_hdl(),
               dblk.get_m(), dblk.get_n(), 
               blk,
               &next_elim, perm, d,
               &cdata, &backup,
               &options, &worker_stats, &work, &alloc,
               FACTOR_APP_PRIO);
         
#else

         sylver::inform_t stats = worker_stats[0];
         
         try {
            factor_block_app(
                  dblk, next_elim, perm, d, cdata, backup, options, work[0],
                  alloc);
         } catch (SingularError const&) {
            stats.flag = sylver::Flag::ERROR_SINGULAR;
         }
            
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
            ColumnData<T,IntAlloc>& cdata, Backup& backup,
            sylver::options_t& options) {

         int blk = dblk.get_col();
         int iblk = rblk.get_row();

#if defined(SPLDLT_USE_STARPU)

         sylver::spldlt::starpu::insert_applyN_block_app(
               dblk.get_hdl(), rblk.get_hdl(),
               cdata[blk].get_hdl(),
               dblk.get_m(), dblk.get_n(), 
               blk, iblk,
               &cdata, &backup, &options,
               APPLYN_APP_PRIO);

#else

         applyN_block_app(
               dblk, rblk, cdata, backup, options);

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
            ColumnData<T,IntAlloc>& cdata, Backup& backup,
            sylver::options_t& options) {

         int blk = dblk.get_col();
         int jblk = cblk.get_col();

#if defined(SPLDLT_USE_STARPU)

         sylver::spldlt::starpu::insert_applyT_block_app (
               dblk.get_hdl(), cblk.get_hdl(),
               cdata[blk].get_hdl(),
               dblk.get_m(), dblk.get_n(), 
               blk, jblk,
               &cdata, &backup, &options,
               APPLYT_APP_PRIO);

#else

         applyT_block_app(
               dblk, cblk, cdata, backup, options);
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
            ColumnData<T,IntAlloc>& cdata, Backup& backup, 
            T const beta, T* upd, int const ldupd,
            std::vector<spral::ssids::cpu::Workspace>& work
            ) {

         int iblk = ublk.get_row();
         int jblk = ublk.get_col();
         int blk = jsrc.get_col();

#if defined(SPLDLT_USE_STARPU)

         sylver::spldlt::starpu::insert_updateN_block_app(
               isrc.get_hdl(), jsrc.get_hdl(), ublk.get_hdl(), 
               cdata[blk].get_d_hdl(), cdata[blk].get_hdl(),
               ublk.get_m(), ublk.get_n(), 
               iblk, jblk, blk,
               &cdata, &backup,
               beta, upd, ldupd,
               &work, ublk.get_blksz(),
               UPDATEN_APP_PRIO);

#else

         updateN_block_app(
               isrc, jsrc, ublk, cdata, backup,
               beta, upd, ldupd, work[0]);

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
            std::vector<spral::ssids::cpu::Workspace>& work
            ) {

         int iblk = ublk.get_row();
         int jblk = ublk.get_col();
         int blk = jsrc.get_row();

         int isrc_row = isrc.get_row();
         int isrc_col = isrc.get_col();

#if defined(SPLDLT_USE_STARPU)
         
         sylver::spldlt::starpu::insert_updateT_block_app(
               isrc.get_hdl(), jsrc.get_hdl(), ublk.get_hdl(), 
               cdata[blk].get_hdl(),
               ublk.get_m(), ublk.get_n(),
               isrc_row, isrc_col,
               iblk, jblk, blk,
               &cdata, &backup,
               &work, ublk.get_blksz(),
               UPDATET_APP_PRIO);

#else

         updateT_block_app(
               isrc, jsrc, ublk,
               backup, work[0]);
         
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

         sylver::spldlt::starpu::insert_adjust(
               cdata[blk].get_hdl(), blk,
               &next_elim, &cdata,
               ADJUST_APP_PRIO);
#else

         adjust_app(blk, next_elim, cdata);

#endif

      }

      /* update_contrib task: Udpate a block as part of the
         contribution block
       */
      static
      void udpate_contrib_task(
            BlockSpec& isrc, BlockSpec& jsrc, BlockSpec& ublk,
            T const beta, T* upd_ij, int const ldupd,
            std::vector<spral::ssids::cpu::Workspace>& work
            ) {

         int iblk =  ublk.get_row();
         int jblk =  ublk.get_col();
         int blk = jsrc.get_row();

         if(debug) printf("FormContrib(%d,%d,%d)\n", iblk, jblk, blk);

         ublk.form_contrib(
               isrc, jsrc, work[0], beta, upd_ij, ldupd
               );

      }

      /// Restore ny failed row and release backup
      static 
      void restore_failed_block_app_task(
            int elim_col,
            BlockSpec& jblk,
            BlockSpec& blk,
            ColumnData<T,IntAlloc>& cdata, Backup& backup,
            std::vector<spral::ssids::cpu::Workspace>& workspaces) {

#if defined(SPLDLT_USE_STARPU)

         sylver::spldlt::starpu::insert_restore_failed_block_app(
               jblk.get_hdl(), blk.get_hdl(), cdata[elim_col].get_hdl(),
               blk.get_m(), blk.get_n(),
               blk.get_row(), blk.get_col(), elim_col,
               &cdata, &backup, &workspaces, blk.get_blksz(),
               RESTORE_APP_PRIO);

#else

         restore_failed_block_app(elim_col, jblk, blk, cdata, backup, workspaces[0]);
         
#endif
      } 

      ////////////////////////////////////////////////////////////
      // factorize_indef_app_notask
      //
      // Sequential factorization routine for indefinite matrices
      // implementing an APTP strategy. Failed entires are left in
      // place.
      static
      int factorize_indef_app_notask (
            int const m, int const n, int* perm, T* a,
            int const lda, T* d, ColumnData<T,IntAlloc>& cdata, Backup& backup,
            sylver::options_t& options, int const block_size,
            T const beta, T* upd, int const ldupd, spral::ssids::cpu::Workspace& work,
            Allocator const& alloc, int const from_blk=0) {

         typedef spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> BlockSpec;
         // typedef ColumnData<T,IntAlloc> ColumnDataSpec;

         int const nblk = calc_nblk(n, block_size);
         int const mblk = calc_nblk(m, block_size);

         /* Setup */
         int next_elim = from_blk*block_size;
      
         int num_blocks = nblk*mblk;
         std::vector<BlockSpec> blocks;
         blocks.reserve(num_blocks);

         for(int jblk=0; jblk<nblk; jblk++) {
            for(int iblk=0; iblk<mblk; iblk++) {
               // Create and insert block at the end (column-wise storage)
               blocks.emplace_back(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
               // alternativel store pointer
               // blocks[jblk*mblk + iblk] = new BlockSpec(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
            }
         }
         
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
           
            // factor_block_app_task(
            //       blocks[blk*(mblk+1)] /*dblk*/, next_elim,
            //       perm, d,
            //       cdata, backup,
            //       options/*, block_size*/, work, alloc);

            if(debug) printf("Factor(%d)\n", blk);
            // BlockSpec dblk(blk, blk, m, n, cdata, &a[blk*block_size*lda+blk*block_size], lda, block_size);
            BlockSpec& dblk = blocks[blk*(mblk+1)];
            // Store a copy for recovery in case of a failed column
            dblk.backup(backup);
            // Perform actual factorization
            int nelim = dblk.template factor<Allocator>(
                  next_elim, perm, d, options, /*work[0],*/ work, alloc
                  );
            if(nelim<0) return nelim;
            // Init threshold check (non locking => task dependencies)
            cdata[blk].init_passed(nelim);
            // }
            
            // Loop over off-diagonal blocks applying pivot
            for(int jblk=0; jblk<blk; jblk++) {

               // BlockSpec cblk(blk, jblk, m, n, cdata, &a[jblk*block_size*lda+blk*block_size], lda, block_size);               
               // Apply factorization on uneliminated entries of
               // left-diagonal block
               // applyT_block_app_task(
               //       /*dblk*/ blocks[blk*(mblk+1)], /*cblk*/blocks[jblk*mblk+blk],
               //       cdata, backup,
               //       options);

               if(debug) printf("ApplyT(%d,%d)\n", blk, jblk);
               // BlockSpec dblk(blk, blk, m, n, cdata, &a[blk*block_size*lda+blk*block_size], lda, block_size);
               // BlockSpec cblk(blk, jblk, m, n, cdata, &a[jblk*block_size*lda+blk*block_size], lda, block_size);

               BlockSpec& cblk = blocks[jblk*mblk+blk];

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

            }
            for(int iblk=blk+1; iblk<mblk; iblk++) {

               // BlockSpec rblk(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
               // Apply factorization on sub-diagonal block
               // applyN_block_app_task(
               //       /*dblk*/ blocks[blk*(mblk+1)], /*rblk*/ blocks[blk*mblk+iblk],
               //       cdata, backup,
               //       options);

               if(debug) printf("ApplyN(%d,%d)\n", iblk, blk);
               // BlockSpec dblk(blk, blk, m, n, cdata, &a[blk*block_size*lda+blk*block_size], lda, block_size);
               // BlockSpec rblk(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);

               BlockSpec& rblk = blocks[blk*mblk+iblk];

               // Apply column permutation from factorization of dblk and in
               // the process, store a (permuted) copy for recovery in case of
               // a failed column
               rblk.apply_cperm_and_backup(backup);
               // Perform elimination and determine number of rows in block
               // passing a posteori threshold pivot test
               int blkpass = rblk.apply_pivot_app(dblk, options.u, options.small);
               // Update column's passed pivot count
               cdata[blk].update_passed(blkpass);
            }

            // Adjust column once all applys have finished and we know final
            // number of passed columns.
            // adjust_task(/* dblk*/blocks[blk*(mblk+1)], next_elim, cdata);

            if(debug) printf("Adjust(%d)\n", blk);
            cdata[blk].adjust(next_elim);

            // Update uneliminated columns
            for(int jblk=0; jblk<blk; jblk++) {

               // BlockSpec jsrc(blk, jblk, m, n, cdata, &a[jblk*block_size*lda+blk*block_size], lda, block_size);
               BlockSpec& jsrc = blocks[jblk*mblk+blk];

               for(int iblk=jblk; iblk<mblk; iblk++) {

                  int isrc_row = (blk<=iblk) ? iblk : blk;
                  int isrc_col = (blk<=iblk) ? blk : iblk;
                  // BlockSpec isrc(isrc_row, isrc_col, m, n, cdata, &a[isrc_col*block_size*lda+isrc_row*block_size], lda,
                                 // block_size);
                  // BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);

                  BlockSpec& ublk = blocks[jblk*mblk+iblk];
                  BlockSpec& isrc = blocks[isrc_col*mblk+isrc_row];

                  // Update uneliminated entries in blocks on the left
                  // of current block column
                  // updateT_block_app_task(
                  //       // isrc, jsrc, ublk,
                  //       blocks[isrc_col*mblk+isrc_row], blocks[jblk*mblk+blk], 
                  //       blocks[jblk*mblk+iblk],
                  //       cdata, backup, 
                  //       work);

                  // DEBUG
                  if(debug) printf("UpdateT(%d,%d,%d)\n", iblk, jblk, blk);
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
                  // END DEBUG
               }
            }
            for(int jblk=blk; jblk<nblk; jblk++) {

               // Source block
               // BlockSpec jsrc(jblk, blk, m, n, cdata, &a[blk*block_size*lda+jblk*block_size], lda, block_size);

               BlockSpec& jsrc = blocks[blk*mblk+jblk];

               for(int iblk=jblk; iblk<mblk; iblk++) {

                  // Source block
                  // BlockSpec isrc(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
                  // Destination block
                  // BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);

                  BlockSpec& ublk = blocks[jblk*mblk+iblk];
                  BlockSpec& isrc = blocks[blk*mblk+iblk];

                  // If we are on the current block column, restore
                  // any failed columns and release backups.
                  // Update blocks on the right of the current block column
                  // updateN_block_app_task (
                  //       // isrc, jsrc, ublk,
                  //       blocks[blk*mblk+iblk], blocks[blk*mblk+jblk],
                  //       blocks[jblk*mblk+iblk],
                  //       cdata, backup,
                  //       beta, upd, ldupd,
                  //       work);

                  // If we're on the block col we've just eliminated, restore
                  // any failed cols and release resources storing backup
                  ublk.restore_if_required(backup, blk);
                  // Perform actual update
                  ublk.update(isrc, jsrc, work,
                              beta, upd, ldupd);

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

                        // BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
                        // BlockSpec isrc(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
                        // BlockSpec jsrc(jblk, blk, m, n, cdata, &a[blk*block_size*lda+jblk*block_size], lda, block_size);

                        BlockSpec& ublk = blocks[jblk*mblk+iblk];
                        BlockSpec& isrc = blocks[blk*mblk+iblk];
                        BlockSpec& jsrc = blocks[blk*mblk+jblk];

                        // udpate_contrib_task(
                        //       // isrc, jsrc, ublk,
                        //       blocks[blk*mblk+iblk], blocks[blk*mblk+jblk],
                        //       blocks[jblk*mblk+iblk],
                        //       beta, upd_ij, ldupd,
                        //       work
                        //       );

                        ublk.form_contrib(
                              isrc, jsrc, work, beta, upd_ij, ldupd
                              );
                        
                     }
                  }
            }

// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif

         } // loop on block columns
         
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

         return next_elim;
      }

      ////////////////////////////////////////////////////////////////////////////////   
      // factorize_indef_app
      // Factorization routine for indefinite matrices implementing an
      // APTP strategy. Failed entires are left in place.
      static
      int factorize_indef_app (
            int const m, int const n, int* perm, T* a,
            int const lda, T* d, ColumnData<T,IntAlloc>& cdata, Backup& backup,
            sylver::options_t& options, std::vector<sylver::inform_t>& worker_stats, int const block_size,
            T const beta, T* upd, int const ldupd, std::vector<spral::ssids::cpu::Workspace>& work,
            Allocator const& alloc, int const from_blk=0) {

         typedef spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> BlockSpec;
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

// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif

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
                  options/*, block_size*/, worker_stats, work, alloc);

// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif

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

// #if defined(SPLDLT_USE_STARPU)
//                starpu_task_wait_for_all();
// #endif

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

// #if defined(SPLDLT_USE_STARPU)
//                starpu_task_wait_for_all();
// #endif

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

// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif

            // DEBUG
            // if(debug) printf("Adjust(%d)\n", blk);
            // cdata[blk].adjust(next_elim);
            // END DEBUG

            // Update uneliminated columns
            for(int jblk=0; jblk<blk; jblk++) {

               // BlockSpec jsrc(blk, jblk, m, n, cdata, &a[jblk*block_size*lda+blk*block_size], lda, block_size);

               for(int iblk=jblk; iblk<mblk; iblk++) {

                  int isrc_row = (blk<=iblk) ? iblk : blk;
                  int isrc_col = (blk<=iblk) ? blk : iblk;
                  // BlockSpec isrc(isrc_row, isrc_col, m, n, cdata, &a[isrc_col*block_size*lda+isrc_row*block_size], lda,
                                 // block_size);
                  // BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);

                  // Update uneliminated entries in blocks on the left
                  // of current block column
                  updateT_block_app_task(
                        // isrc, jsrc, ublk,
                        blocks[isrc_col*mblk+isrc_row], blocks[jblk*mblk+blk], 
                        blocks[jblk*mblk+iblk],
                        cdata, backup, 
                        work);

// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif

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
               // BlockSpec jsrc(jblk, blk, m, n, cdata, &a[blk*block_size*lda+jblk*block_size], lda, block_size);

               for(int iblk=jblk; iblk<mblk; iblk++) {

                  // Source block
                  // BlockSpec isrc(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
                  // Destination block
                  // BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);

                  // If we are on the current block column, restore
                  // any failed columns and release backups.
                  // Update blocks on the right of the current block column
                  updateN_block_app_task (
                        // isrc, jsrc, ublk,
                        blocks[blk*mblk+iblk], blocks[blk*mblk+jblk],
                        blocks[jblk*mblk+iblk],
                        cdata, backup,
                        beta, upd, ldupd,
                        work);

// #if defined(SPLDLT_USE_STARPU)
//                   starpu_task_wait_for_all();
// #endif

               }
            }

// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif
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
                              work
                              );

// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif

                     }
                  }
            }

// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif

         } // loop on block columns
         
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

   public:

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

      static
      void release_permute_failed_task(
            NumericFront<T, Allocator> &node,
            Allocator& alloc
            ) {

#if defined(SPLDLT_USE_STARPU)

         ColumnData<T, IntAlloc> &cdata = *node.cdata;
         int n = node.ncol();
         int const nblk = calc_nblk(n, node.blksz());

         starpu_data_handle_t *col_hdls = new starpu_data_handle_t[nblk];
         for (int c = 0; c < nblk; c++)
            col_hdls[c] = cdata[c].get_hdl();
            
         sylver::spldlt::starpu::insert_permute_failed(
               col_hdls, nblk,
               &node, &alloc);

         delete[] col_hdls;
#else
         int m = node.nrow();
         int n = node.ncol();
         size_t ldl = align_lda<T>(m);
         T *lcol = node.lcol;

         CopyBackup<T, Allocator> &backup = *node.backup; 
         ColumnData<T, IntAlloc> &cdata = *node.cdata;
         
         backup.release_all_memory(); 

         if (node.nelim() < n)
            permute_failed (
                  m, n, node.perm, lcol, ldl,
                  node.nelim(), 
                  cdata, node.blksz(),
                  alloc);         
#endif

      }

      ////////////////////////////////////////////////////////////
      // factor_front_indef_app_notask
      //
      // TODO: deactivate speculative execution and backing up as we
      // run in sequential.
      static
      void factor_front_indef_app_notask(
            NumericFront<T, Allocator> &front,
            sylver::options_t& options,
            sylver::inform_t& stats,
            spral::ssids::cpu::Workspace& work,
            Allocator const& alloc,
            int& next_elim, int const from_blk=0) {

         // Front info
         int const nblk = front.nc();
         int const mblk = front.nr();
         
         int const block_size = front.blksz();
         int const m = front.nrow();
         int const n = front.ncol();
         T *lcol = front.lcol;
         int ldl = front.ldl();
         T *d = &front.lcol[n*ldl];
         int* perm = front.perm;
         std::vector<Block<T, iblksz, IntAlloc>>& blocks = front.blocks;
         std::vector<sylver::Tile<T, Allocator>>& contrib_blocks = front.contrib_blocks;
         ColumnData<T, IntAlloc> &cdata = *front.cdata;
         CopyBackup<T, Allocator> &backup = *front.backup;

         // Dimensions of contribution block
         size_t contrib_dimn = m-n;
         int rsa = n/block_size; // index of first block in contribution blocks  

         // Loop over blocks
         for(int blk=from_blk; blk<nblk; blk++) {


            // Factor block on the diagonal
            factor_block_app(
                  blocks[blk*(mblk+1)], next_elim, perm, d, cdata, backup,
                  options, work, alloc);
            
            // Loop over off-diagonal blocks applying pivot
            for(int jblk=0; jblk<blk; jblk++) {
               
               // Apply factorization on uneliminated entries of
               // left-diagonal block
               applyT_block_app(
                     blocks[blk*(mblk+1)], blocks[jblk*mblk+blk], cdata, backup,
                     options);               
            }
            for(int iblk=blk+1; iblk<mblk; iblk++) {

               // Apply factorization on sub-diagonal block
               applyN_block_app(
                     blocks[blk*(mblk+1)], blocks[blk*mblk+iblk], cdata, backup,
                     options);
            }

            // Adjust column once all applys have finished and we know final
            // number of passed columns.
            adjust_app(blk, next_elim, cdata);

            for(int iblk=blk; iblk<mblk; iblk++) {

               // Restore failed columns in current block-column
               restore_failed_block_app(
                     blk,
                     blocks[blk*mblk+iblk], blocks[blk*(mblk+1)],
                     blocks[blk*mblk+iblk],
                     cdata, backup,
                     work);
            }

            // Update uneliminated columns
            
            // Loop over left-diagonal block-columns
            for(int jblk=0; jblk<blk; jblk++) {

               // Loop over block-rows
               for(int iblk=jblk; iblk<mblk; iblk++) {

                  int isrc_row = (blk<=iblk) ? iblk : blk;
                  int isrc_col = (blk<=iblk) ? blk : iblk;

                  // Update uneliminated entries in blocks on the left
                  // of current block column
                  updateT_block_app(
                        blocks[isrc_col*mblk+isrc_row], blocks[jblk*mblk+blk], 
                        blocks[jblk*mblk+iblk],
                        backup, work);                  
               }
            } // Loop over left-diagonal block-columns

            // Update trailing submatrix (fully-summed) 

            // Loop over trailing submatrix block-columns
            for(int jblk=blk+1; jblk<nblk; jblk++) {

               for(int iblk=jblk; iblk<mblk; iblk++) {

                  T const beta = 0.0;
                  T* upd = nullptr;
                  int const ldupd = 0;
                     
                  updateN_block_app (
                        blocks[blk*mblk+iblk], blocks[blk*mblk+jblk],
                        blocks[jblk*mblk+iblk],
                        cdata, backup,
                        beta, upd, ldupd,
                        work);

               }
            } // Loop over trailing submatrix block-columns

            // Update trailing submatrix (contribution blocks) 

            if (contrib_dimn > 0) {

               for (int jblk = rsa; jblk < mblk; ++jblk) {
                  for (int iblk = jblk;  iblk < mblk; ++iblk) {

                     sylver::Tile<T, Allocator>& upd = front.contrib_block(iblk, jblk);
                     
                     update_contrib_block_app
                        <T, IntAlloc, Allocator>
                        (front,
                         blk, iblk, jblk,
                         blocks[blk*mblk+iblk].get_a(), blocks[blk*mblk+iblk].get_lda(),
                         blocks[blk*mblk+jblk].get_a(), blocks[blk*mblk+jblk].get_lda(),
                         upd.m, upd.n, upd.a, upd.lda,
                         work);

                  }    
               }
               
            } // contrib_dimn > 0
            
            
         }
         
      }

      
      ////////////////////////////////////////////////////////////
      // factor_front_indef_app
      //
      /// @brief Perform the LDLT factorization of a matrix using a
      /// APTP pivoting strategy. 
      /// @note This call is asynchronous.
      static
      void factor_front_indef_app(
            NumericFront<T, Allocator> &node,
            sylver::options_t& options,
            std::vector<sylver::inform_t>& worker_stats,
            T const beta, T* upd, int const ldupd,
            std::vector<spral::ssids::cpu::Workspace>& workspaces,
            Allocator const& alloc, int& next_elim, int const from_blk=0
            ) {
         
         typedef spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> BlockSpec;
         
         int const block_size = node.blksz();
         int const m = node.nrow();
         int const n = node.ncol();
         T *lcol = node.lcol;
         int ldl = align_lda<T>(m);
         T *d = &node.lcol[n*ldl];
         int *perm = node.perm;
         std::vector<Block<T, iblksz, IntAlloc>>& blocks = node.blocks;
         std::vector<sylver::Tile<T, Allocator>>& contrib_blocks = node.contrib_blocks;
         ColumnData<T, IntAlloc> &cdata = *node.cdata;
         CopyBackup<T, Allocator> &backup = *node.backup;
         
         int const nblk = calc_nblk(n, block_size);
         int const mblk = calc_nblk(m, block_size);

         size_t contrib_dimn = m-n;
         int rsa = n/block_size; // index of first block in contribution blocks  
         int ncontrib = mblk-rsa;

         /* Setup */
         // int next_elim = from_blk*block_size;
      
         /* Inner loop - iterate over block columns */
         // try {
         for(int blk=from_blk; blk<nblk; blk++) {

            // Factorize block on diagonal
            factor_block_app_task(
                  blocks[blk*(mblk+1)], next_elim, perm, d, cdata, backup,
                  options, worker_stats, workspaces, alloc);
                        
            // Loop over off-diagonal blocks applying pivot
            for(int jblk=0; jblk<blk; jblk++) {

               // Apply factorization on uneliminated entries of
               // left-diagonal block
               applyT_block_app_task(
                     blocks[blk*(mblk+1)], blocks[jblk*mblk+blk],
                     cdata, backup,
                     options);

            }
            for(int iblk=blk+1; iblk<mblk; iblk++) {

               // Apply factorization on sub-diagonal block
               applyN_block_app_task(
                     blocks[blk*(mblk+1)], blocks[blk*mblk+iblk],
                     cdata, backup,
                     options);

            }

            // Adjust column once all applys have finished and we know final
            // number of passed columns.
            adjust_task(blocks[blk*(mblk+1)], next_elim, cdata);

#if defined(SPLDLT_USE_GPU)            

            // Restore failed columns in current block-column
            for(int iblk=blk; iblk<mblk; iblk++) {
               restore_failed_block_app_task(
                     blk,
                     blocks[blk*(mblk+1)],
                     blocks[blk*mblk+iblk],
                     cdata, backup,
                     workspaces);
            }

#endif

            // Update uneliminated columns
            for(int jblk=0; jblk<blk; jblk++) {

               for(int iblk=jblk; iblk<mblk; iblk++) {

                  int isrc_row = (blk<=iblk) ? iblk : blk;
                  int isrc_col = (blk<=iblk) ? blk : iblk;

                  // Update uneliminated entries in blocks on the left
                  // of current block column
                  updateT_block_app_task(
                        // isrc, jsrc, ublk,
                        blocks[isrc_col*mblk+isrc_row], blocks[jblk*mblk+blk], 
                        blocks[jblk*mblk+iblk],
                        cdata, backup, 
                        workspaces);

               }
            }

#if defined(SPLDLT_USE_GPU)            

            for(int jblk=blk+1; jblk<nblk; jblk++) {

               for(int iblk=jblk; iblk<mblk; iblk++) {

                  updateN_block_app_task (
                        blocks[blk*mblk+iblk], blocks[blk*mblk+jblk],
                        blocks[jblk*mblk+iblk],
                        cdata, backup,
                        beta, upd, ldupd,
                        workspaces);

               }
            }

#else
            for(int jblk=blk; jblk<nblk; jblk++) {

               for(int iblk=jblk; iblk<mblk; iblk++) {

                  // If we are on the current block column, restore
                  // any failed columns and release backups.
                  // Update blocks on the right of the current block column
                  updateN_block_app_task (
                        // isrc, jsrc, ublk,
                        blocks[blk*mblk+iblk], blocks[blk*mblk+jblk],
                        blocks[jblk*mblk+iblk],
                        cdata, backup,
                        beta, upd, ldupd,
                        workspaces);

               }
            }

#endif

            // Handle update to contribution block, if required
            if (contrib_dimn > 0) {
               for (int jblk = rsa; jblk < mblk; ++jblk) {
                  for (int iblk = jblk;  iblk < mblk; ++iblk) {
                     update_contrib_block_app_task(
                           blocks[blk*mblk+iblk], blocks[blk*mblk+jblk],
                           contrib_blocks[(jblk-rsa)*ncontrib+(iblk-rsa)],
                           node,
                           blk, iblk, jblk, workspaces, block_size, 
                           UPDATEC_APP_PRIO);

// #if defined(SPLDLT_USE_STARPU)
//                      starpu_task_wait_for_all();
// #endif

                  }    
               }
            }

            // #if defined(SPLDLT_USE_STARPU)
            //             starpu_task_wait_for_all();
            // #endif

         } // loop on block columns
         
         
      }

      
      ////////////////////////////////////////////////////////////////////////////////   
      // factor_indef_app_async
      //
      /// @brief Perform the LDLT factorization of a matrix using a
      /// APTP pivoting strategy. 
      /// @note This call is asynchronous.
//       static
//       void factor_indef_app_async(
//             int const m, int const n, int* perm, T* a,
//             int const lda, T* d, ColumnData<T,IntAlloc>& cdata, Backup& backup,
//             struct cpu_factor_options& options, int const block_size,
//             T const beta, T* upd, int const ldupd, std::vector<spral::ssids::cpu::Workspace>& work,
//             Allocator const& alloc, int& next_elim, int const from_blk=0
//             ) {

//          typedef spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> BlockSpec;
         

//          int const nblk = calc_nblk(n, block_size);
//          int const mblk = calc_nblk(m, block_size);

//          /* Setup */
//          // int next_elim = from_blk*block_size;
      
//          int num_blocks = nblk*mblk;
//          std::vector<BlockSpec> blocks;
//          blocks.reserve(num_blocks);

//          for(int jblk=0; jblk<nblk; jblk++) {
//             for(int iblk=0; iblk<mblk; iblk++) {
//                // Create and insert block at the end (column-wise storage)
//                blocks.emplace_back(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
//                // alternativel store pointer
//                // blocks[jblk*mblk + iblk] = new BlockSpec(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
// #if defined(SPLDLT_USE_STARPU)
//                // register handle for block (iblk, jblk)
//                blocks[jblk*mblk+iblk].register_handle(); 
// #endif
//             }
//          }

//          /* Inner loop - iterate over block columns */
//          // try {
//          for(int blk=from_blk; blk<nblk; blk++) {

//             // #if defined(SPLDLT_USE_STARPU)
//             //             starpu_task_wait_for_all();
//             // #endif

//             /*if(debug) {
//               printf("Bcol %d:\n", blk);
//               print_mat(mblk, nblk, m, n, blkdata, cdata, lda);
//               }*/

//             // Factor diagonal: depend on perm[blk*block_size] as we init npass
//             // {  
//             // BlockSpec dblk(blk, blk, m, n, cdata, a, lda, block_size);
            
//             // Factorize block on diagonal

//             // BlockSpec dblk(blk, blk, m, n, cdata, &a[blk*block_size*lda+blk*block_size], lda, block_size);
           
//             factor_block_app_task(
//                   blocks[blk*(mblk+1)] /*dblk*/, next_elim,
//                   perm, d,
//                   cdata, backup,
//                   options/*, block_size*/, work, alloc);

//             // #if defined(SPLDLT_USE_STARPU)
//             //             starpu_task_wait_for_all();
//             // #endif

//             // DEBUG
//             // {
//             //    if(debug) printf("Factor(%d)\n", blk);
//             //    BlockSpec dblk(blk, blk, m, n, cdata, &a[blk*block_size*lda+blk*block_size], lda, block_size);
//             //    // Store a copy for recovery in case of a failed column
//             //    dblk.backup(backup);
//             //    // Perform actual factorization
//             //    int nelim = dblk.template factor<Allocator>(
//             //          next_elim, perm, d, options, work[0], /*work,*/ alloc
//             //          );
//             //    if(nelim<0) return nelim;
//             //    // Init threshold check (non locking => task dependencies)
//             //    cdata[blk].init_passed(nelim);
//             // }
//             // END DEBUG

//             // }
            
//             // Loop over off-diagonal blocks applying pivot
//             for(int jblk=0; jblk<blk; jblk++) {

//                // BlockSpec cblk(blk, jblk, m, n, cdata, &a[jblk*block_size*lda+blk*block_size], lda, block_size);               
//                // Apply factorization on uneliminated entries of
//                // left-diagonal block
//                applyT_block_app_task(
//                      /*dblk*/ blocks[blk*(mblk+1)], /*cblk*/blocks[jblk*mblk+blk],
//                      cdata, backup,
//                      options);

//                // #if defined(SPLDLT_USE_STARPU)
//                //                starpu_task_wait_for_all();
//                // #endif

//                // DEBUG
//                // if(debug) printf("ApplyT(%d,%d)\n", blk, jblk);
//                // BlockSpec dblk(blk, blk, m, n, cdata, &a[blk*block_size*lda+blk*block_size], lda, block_size);
//                // BlockSpec cblk(blk, jblk, m, n, cdata, &a[jblk*block_size*lda+blk*block_size], lda, block_size);
//                // // Apply row permutation from factorization of dblk and in
//                // // the process, store a (permuted) copy for recovery in case of
//                // // a failed column
//                // cblk.apply_rperm_and_backup(backup);
//                // // Perform elimination and determine number of rows in block
//                // // passing a posteori threshold pivot test
//                // int blkpass = cblk.apply_pivot_app(
//                //       dblk, options.u, options.small
//                //       );
//                // // Update column's passed pivot count
//                // cdata[blk].update_passed(blkpass);
//                // END DEBUG
//             }
//             for(int iblk=blk+1; iblk<mblk; iblk++) {

//                // BlockSpec rblk(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
//                // Apply factorization on sub-diagonal block
//                applyN_block_app_task(
//                      /*dblk*/ blocks[blk*(mblk+1)], /*rblk*/ blocks[blk*mblk+iblk],
//                      cdata, backup,
//                      options);

//                // #if defined(SPLDLT_USE_STARPU)
//                //                starpu_task_wait_for_all();
//                // #endif

//                // DEBUG
//                // if(debug) printf("ApplyN(%d,%d)\n", iblk, blk);
//                // BlockSpec dblk(blk, blk, m, n, cdata, &a[blk*block_size*lda+blk*block_size], lda, block_size);
//                // BlockSpec rblk(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
//                // // Apply column permutation from factorization of dblk and in
//                // // the process, store a (permuted) copy for recovery in case of
//                // // a failed column
//                // rblk.apply_cperm_and_backup(backup);
//                // // Perform elimination and determine number of rows in block
//                // // passing a posteori threshold pivot test
//                // int blkpass = rblk.apply_pivot_app(dblk, options.u, options.small);
//                // // Update column's passed pivot count
//                // cdata[blk].update_passed(blkpass);
//                // END DEBUG
//             }

//             // Adjust column once all applys have finished and we know final
//             // number of passed columns.
//             adjust_task(/* dblk*/blocks[blk*(mblk+1)], next_elim, cdata);

//             // #if defined(SPLDLT_USE_STARPU)
//             //             starpu_task_wait_for_all();
//             // #endif

//             // DEBUG
//             // if(debug) printf("Adjust(%d)\n", blk);
//             // cdata[blk].adjust(next_elim);
//             // END DEBUG

//             // Update uneliminated columns
//             for(int jblk=0; jblk<blk; jblk++) {

//                // BlockSpec jsrc(blk, jblk, m, n, cdata, &a[jblk*block_size*lda+blk*block_size], lda, block_size);

//                for(int iblk=jblk; iblk<mblk; iblk++) {

//                   int isrc_row = (blk<=iblk) ? iblk : blk;
//                   int isrc_col = (blk<=iblk) ? blk : iblk;
//                   // BlockSpec isrc(isrc_row, isrc_col, m, n, cdata, &a[isrc_col*block_size*lda+isrc_row*block_size], lda,
//                   // block_size);
//                   // BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);

//                   // Update uneliminated entries in blocks on the left
//                   // of current block column
//                   updateT_block_app_task(
//                         // isrc, jsrc, ublk,
//                         blocks[isrc_col*mblk+isrc_row], blocks[jblk*mblk+blk], 
//                         blocks[jblk*mblk+iblk],
//                         cdata, backup, 
//                         work);

//                   // #if defined(SPLDLT_USE_STARPU)
//                   //             starpu_task_wait_for_all();
//                   // #endif

//                   // DEBUG
//                   // if(debug) printf("UpdateT(%d,%d,%d)\n", iblk, jblk, blk);
//                   // int thread_num = omp_get_thread_num();
//                   // BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
//                   // int isrc_row = (blk<=iblk) ? iblk : blk;
//                   // int isrc_col = (blk<=iblk) ? blk : iblk;
//                   // BlockSpec isrc(isrc_row, isrc_col, m, n, cdata, &a[isrc_col*block_size*lda+isrc_row*block_size], lda,
//                   //                block_size);
//                   // BlockSpec jsrc(blk, jblk, m, n, cdata, &a[jblk*block_size*lda+blk*block_size], lda, block_size);
//                   // // If we're on the block row we've just eliminated, restore
//                   // // any failed rows and release resources storing backup
//                   // ublk.restore_if_required(backup, blk);
//                   // // Perform actual update
//                   // ublk.update(isrc, jsrc, work[0]);
//                   // END DEBUG
//                }
//             }
//             for(int jblk=blk; jblk<nblk; jblk++) {

//                // Source block
//                // BlockSpec jsrc(jblk, blk, m, n, cdata, &a[blk*block_size*lda+jblk*block_size], lda, block_size);

//                for(int iblk=jblk; iblk<mblk; iblk++) {

//                   // Source block
//                   // BlockSpec isrc(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
//                   // Destination block
//                   // BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);

//                   // If we are on the current block column, restore
//                   // any failed columns and release backups.
//                   // Update blocks on the right of the current block column
//                   updateN_block_app_task (
//                         // isrc, jsrc, ublk,
//                         blocks[blk*mblk+iblk], blocks[blk*mblk+jblk],
//                         blocks[jblk*mblk+iblk],
//                         cdata, backup,
//                         beta, upd, ldupd,
//                         work);

//                   // #if defined(SPLDLT_USE_STARPU)
//                   //                   starpu_task_wait_for_all();
//                   // #endif

//                }
//             }

//             // #if defined(SPLDLT_USE_STARPU)
//             //             starpu_task_wait_for_all();
//             // #endif
//             // Handle update to contribution block, if required
//             //             if(upd && mblk>nblk) {
//             //                int uoffset = std::min(nblk*block_size, m) - n;
//             //                T *upd2 = &upd[uoffset*(ldupd+1)];
//             //                for(int jblk=nblk; jblk<mblk; ++jblk)
//             //                   for(int iblk=jblk; iblk<mblk; ++iblk) {
//             //                      T* upd_ij = &upd2[(jblk-nblk)*block_size*ldupd + 
//             //                                        (iblk-nblk)*block_size];
//             //                      {

//             //                         BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
//             //                         BlockSpec isrc(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
//             //                         BlockSpec jsrc(jblk, blk, m, n, cdata, &a[blk*block_size*lda+jblk*block_size], lda, block_size);

//             //                         udpate_contrib_task(
//             //                               // isrc, jsrc, ublk,
//             //                               blocks[blk*mblk+iblk], blocks[blk*mblk+jblk],
//             //                               blocks[jblk*mblk+iblk],
//             //                               beta, upd_ij, ldupd,
//             //                               work
//             //                               );

//             // // #if defined(SPLDLT_USE_STARPU)
//             // //             starpu_task_wait_for_all();
//             // // #endif

//             //                      }
//             //                   }
//             //             }

//             // #if defined(SPLDLT_USE_STARPU)
//             //             starpu_task_wait_for_all();
//             // #endif

//          } // loop on block columns
         
         
//       }


      ////////////////////////////////////////////////////////////////////////////////   
      // ldlt_app_notask
      //
      // Sequential factorization routine for indefinite matrices implementing an
      // APTP strategy. After the factorization, fail entries are permuted
      // to the back of the matrix.
      static
      int ldlt_app_notask(int m, int n, int *perm, 
                          T *a, int lda, T *d, 
                          Backup& backup, 
                          sylver::options_t& options/*, PivotMethod pivot_method*/, 
                          int block_size, T beta, T* upd, int ldupd, 
                          spral::ssids::cpu::Workspace& work, 
                          Allocator const& alloc=Allocator()) {

         /* Sanity check arguments */
         if(m < n) return -1;
         if(lda < n) return -4;
         // printf("[ldlt_app_notask]\n");
         typedef typename std::allocator_traits<Allocator>::template rebind_alloc<int> IntAlloc;
         typedef typename std::allocator_traits<Allocator>::template rebind_alloc<T> TAlloc;

         int num_elim; // return value

         /* Initialize useful quantities: */
         int nblk = calc_nblk(n, block_size);
         int mblk = calc_nblk(m, block_size);

         /* Temporary workspaces */
         ColumnData<T, IntAlloc> cdata(n, block_size, IntAlloc(alloc));

         // factorize matrix and leave fail entries in place
         num_elim = factorize_indef_app_notask (
               m, n, perm, a, lda, d, cdata, backup,
               options, block_size,
               beta, upd, ldupd, work,
               alloc);

         // By default function calls are asynchronous, so we put a
         // barrier and wait for the task-based factorization to be
         // completed
         // #if defined(SPLDLT_USE_STARPU)
         //          starpu_task_wait_for_all();
         // #endif

         
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

      /*
        Factorization routine for indefinite matrices implementing an
        APTP strategy. After the factorization, fail entries are permuted
        to the back of the matrix.
      */
      static
      int ldlt_app(
            int m, int n, 
            int *perm, 
            T *a, int lda, T *d, 
            Backup& backup, 
            sylver::options_t& options, std::vector<sylver::inform_t>& worker_stats,
            int block_size, T beta, T* upd, int ldupd, 
            std::vector<spral::ssids::cpu::Workspace>& work, 
            Allocator const& alloc=Allocator()) {
         
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
               options, worker_stats, block_size,
               beta, upd, ldupd, work,
               alloc);

         // By default function calls are asynchronous, so we put a
         // barrier and wait for the task-based factorization to be
         // completed
// #if defined(SPLDLT_USE_STARPU)
//          starpu_task_wait_for_all();
// #endif

         
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

}} // End of namespace sylver::spldlt
