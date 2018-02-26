#pragma once

#include "ssids/cpu/kernels/ldlt_nopiv.hxx"

#include "kernels/ldlt_app.hxx"
#include "kernels/factor_indef.hxx"
#include "tasks_indef.hxx"

#if defined(SPLDLT_USE_STARPU)
#include "StarPU/kernels_indef.hxx"
using namespace spldlt::starpu;
#endif

namespace spldlt {

   static const int INNER_BLOCK_SIZE = 32;


   ////////////////////////////////////////////////////////////////////////////////
   // factor_front_indef_firtstpass_nodcontrib
   // 
   /// @brief Factorize the front for the first pass (i.e. without
   /// pivoting the failed columns outside their blocks). Do not
   /// update the Shur complement.
   // template <typename T, typename PoolAlloc>
   // void factor_front_indef_firtstpass_nodcontrib(
   //       NumericFront<T, PoolAlloc> &node,
   //       struct cpu_factor_options& options,
   //       PoolAlloc& pool_alloc
   //       ) {

   //    /* Extract useful information about node */
   //    int m = node.get_nrow();
   //    int n = node.get_ncol();
   //    size_t ldl = align_lda<T>(m);
   //    int blksz = options.cpu_block_size;

   //    if (options.pivot_method==PivotMethod::app_block) {
         
   //       CopyBackup<T, PoolAlloc> backup(m, n, blksz, pool_alloc);

   //    }      
   // }


   ////////////////////////////////////////////////////////////////////////////////
   // factor_front_indef_nocontrib
   //
   // Perform the LDLT factorization of front without forming the
   // contribution block
   template <typename T, typename PoolAlloc>
   void factor_front_indef_nocontrib(
         SymbolicFront const& snode,
         NumericFront<T, PoolAlloc> &node,
         std::vector<spral::ssids::cpu::Workspace> &workspaces,
         PoolAlloc& pool_alloc,
         struct cpu_factor_options& options) {

      /* Extract useful information about node */
      int m = node.get_nrow();
      int n = node.get_ncol();
      size_t ldl = align_lda<T>(m);
      T *lcol = node.lcol;
      T *d = &node.lcol[n*ldl];
      int *perm = node.perm;

      int ldld = m;

      int nelim = 0;

      printf("[factor_front_indef_nocontrib]\n");

      int blksz = options.cpu_block_size;
      // node.nelim = nelim;      
      bool const debug = false;
      T *upd = nullptr;

      node.nelim = 0; // TODO add parameter from;

      if (options.pivot_method==PivotMethod::app_block) {

         typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> IntAlloc;
         
         // CopyBackup<T> backup(m, n, blksz);
         // CopyBackup<T, PoolAlloc> backup(m, n, blksz, pool_alloc);

         // node.alloc_backup();
         // node.alloc_cdata();

         CopyBackup<T, PoolAlloc> &backup = *node.backup; 
         ColumnData<T, IntAlloc> &cdata = *node.cdata;

         // int const nblk = calc_nblk(n, blksz);
         // int const mblk = calc_nblk(m, blksz);

         //ColumnData<T, IntAlloc> cdata(n, blksz, IntAlloc(pool_alloc));

         // node.nelim = FactorSymIndef
         //    <T, INNER_BLOCK_SIZE, CopyBackup<T, PoolAlloc>, debug, PoolAlloc>
         //    ::ldlt_app(m, n, perm, lcol, ldl, d, backup, options, blksz, 0.0, upd, 0, 
         //               workspaces, pool_alloc);
           
         FactorSymIndef
            <T, INNER_BLOCK_SIZE, CopyBackup<T, PoolAlloc>, debug, PoolAlloc>
            ::factor_indef_app_async(m, n, perm, lcol, ldl, d, cdata, backup,
                                     options, blksz, 0.0, upd, 0, workspaces,
                                     pool_alloc, node.nelim);

// #if defined(SPLDLT_USE_STARPU)
//          starpu_task_wait_for_all();
// #endif

//          printf("[factor_front_indef_nocontrib] node_nelim = %d\n", node.nelim);

         // // realease all memory used for backup
         // backup.release_all_memory(); 
         
         // // Permute failed entries to end
         // if (node.nelim < n)
         //    FactorSymIndef
         //       <T, INNER_BLOCK_SIZE, CopyBackup<T, PoolAlloc>, debug, PoolAlloc>
         //       ::permute_failed (
         //             m, n, perm, lcol, ldl,
         //             nelim, 
         //             cdata, blksz,
         //             pool_alloc);

         FactorSymIndef
            <T, INNER_BLOCK_SIZE, CopyBackup<T, PoolAlloc>, debug, PoolAlloc>
            ::release_permute_failed_task (
                  node, pool_alloc, blksz);

// #if defined(SPLDLT_USE_STARPU)
//          starpu_task_wait_for_all();
// #endif
         
//         printf("[factor_front_indef_nocontrib] first pass = %d out of %d\n", node.nelim, n);
      }

      factor_front_indef_secondpass_nocontrib_task(
            node, workspaces,options);

// #if defined(SPLDLT_USE_STARPU)
//       starpu_task_wait_for_all();
// #endif

      // if (node.nelim < n) {
      //    // Use TPP factor to eliminate the remaining columns in the following cases:
      //    // 1) options.pivot_method is set to tpp;
      //    // 2) We are at a root node;
      //    // 3) options.failed_pivot_method is set to tpp.
      //    if (m==n || options.pivot_method==PivotMethod::tpp ||
      //          options.failed_pivot_method==FailedPivotMethod::tpp) {
      //       nelim = node.nelim;
      //       T *ld = new T[2*(m-nelim)]; // TODO: workspace
      //       node.nelim += ldlt_tpp_factor(
      //             m-nelim, n-nelim, &perm[nelim], &lcol[nelim*(ldl+1)], ldl, 
      //             &d[2*nelim], ld, m-nelim, options.action, options.u, options.small, 
      //             nelim, &lcol[nelim], ldl);
      //       delete[] ld;
      //       printf("[factor_front_indef_nocontrib] second pass = %d out of %d\n", node.nelim, n);

      //    }
      // }

      // node.ndelay_out = n - node.nelim;
   }

// #if defined(SPLDLT_USE_STARPU)
//    template <typename T, typename PoolAlloc>
//    void factor_front_indef_nocontrib_cpu_func(void *buffers[], void *cl_arg) {
      
//       NumericFront<T, PoolAlloc> *node;
//       std::vector<spral::ssids::cpu::Workspace> *workspaces;
//       PoolAlloc *pool_alloc;
//       struct cpu_factor_options *options;

      
      
//    }
// #endif

   ////////////////////////////////////////////////////////////////////////////////
   // form_contrib_front
   //
   // Update contribution blocks
   template <typename T, typename PoolAlloc>
   void form_contrib_front(
         SymbolicFront const& snode,
         NumericFront<T, PoolAlloc> &node,
         int blksz) {

      int UPDATE_PRIO = 4;

      int iblksz = blksz;
      
      int nrow = snode.nrow + node.ndelay_in;
      int ncol = snode.ncol + node.ndelay_in;
      size_t contrib_dimn = nrow-ncol;
      printf("[form_contrib_front] contrib_dimn = %zu\n", contrib_dimn);
      if (contrib_dimn == 0) return; // Nothing to do

      int n = node.nelim;
      int ldl = align_lda<T>(nrow);
      T *lcol = node.lcol;
      int nr = (nrow-1) / blksz + 1; // number of block rows
      int nc = (n-1) / blksz + 1; // number of block columns
      int rsa = ncol / blksz; // index of first block in contribution blocks  
      int ncontrib = nr-rsa;
      T *d = &lcol[ncol*ldl];
      
      int nelim = 0;

      // // tests/debug
      // spldlt::Block<T, PoolAlloc>& upd = node.contrib_blocks[0];
      // nelim = node.nelim;
      // int ldld = spral::ssids::cpu::align_lda<T>(contrib_dimn);
      // T *ld = new T[nelim*ldld];
      // // calcLD<OP_N>(
      // //       contrib_dimn, nelim, &lcol[ncol], ldl, d, ld, ldld);
      // // T rbeta = 0.0;
      // // host_gemm<T>(
      // //       OP_N, OP_T, contrib_dimn, contrib_dimn, nelim,
      // //       -1.0, &lcol[ncol], ldl, ld, ldld,
      // //       rbeta, upd.a, contrib_dimn);
      
      // udpate_contrib_block(
      //       contrib_dimn, contrib_dimn, upd.a, upd.lda,  
      //       nelim, &lcol[ncol], ldl, &lcol[ncol], ldl,
      //       true, d, ld, ldld);

      // delete[] ld;


      // int ldld = spral::ssids::cpu::align_lda<T>(blksz);
      // T *ld = new T[blksz*ldld];

      for(int blk = 0; blk < nc; ++blk) {
         
         for (int jblk = rsa; jblk < nr; ++jblk) {

            for (int iblk = jblk;  iblk < nr; ++iblk) {
               
               update_contrib_indef_task(
                     snode, node, blk, iblk, jblk, blksz, UPDATE_PRIO);
            }            
         }
      }

      // delete[] ld;
   }
   

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
            BlockSpec& dblk, int& next_elim,
            int* perm, T* d,
            ColumnData<T,IntAlloc>& cdata, Backup& backup,
            struct cpu_factor_options& options,
            /*int const block_size,*/
            std::vector<spral::ssids::cpu::Workspace>& work,
            Allocator const& alloc) {

         int blk = dblk.get_row();

#if defined(SPLDLT_USE_STARPU)

         insert_factor_block_app_task (
               dblk.get_hdl(), cdata[blk].get_hdl(),
               dblk.get_m(), dblk.get_n(), 
               blk,
               &next_elim, perm, d,
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
               next_elim, perm, d, options, work[0], alloc
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
            std::vector<spral::ssids::cpu::Workspace>& work
            ) {

         int iblk = ublk.get_row();
         int jblk = ublk.get_col();
         int blk = jsrc.get_col();

#if defined(SPLDLT_USE_STARPU)

         insert_updateN_block_app(
               isrc.get_hdl(), jsrc.get_hdl(), ublk.get_hdl(), 
               cdata[blk].get_hdl(),
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
         ublk.update(isrc, jsrc, work[0],
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
            std::vector<spral::ssids::cpu::Workspace>& work
            ) {

         int iblk = ublk.get_row();
         int jblk = ublk.get_col();
         int blk = jsrc.get_row();

         int isrc_row = isrc.get_row();
         int isrc_col = isrc.get_col();

#if defined(SPLDLT_USE_STARPU)

         insert_updateT_block_app(
               isrc.get_hdl(), jsrc.get_hdl(), ublk.get_hdl(), 
               cdata[blk].get_hdl(),
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
         ublk.update(isrc, jsrc, work[0]);
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

      ////////////////////////////////////////////////////////////////////////////////   
      // factorize_indef_app_notask
      // Sequential factorization routine for indefinite matrices implementing an
      // APTP strategy. Failed entires are left in place.
      static
      int factorize_indef_app_notask (
            int const m, int const n, int* perm, T* a,
            int const lda, T* d, ColumnData<T,IntAlloc>& cdata, Backup& backup,
            struct cpu_factor_options& options, int const block_size,
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
            struct cpu_factor_options& options, int const block_size,
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
                  options/*, block_size*/, work, alloc);

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
            Allocator& alloc,
            int blksz
            ) {

#if defined(SPLDLT_USE_STARPU)

         ColumnData<T, IntAlloc> &cdata = *node.cdata;
         int n = node.get_ncol();
         int const nblk = calc_nblk(n, blksz);

         insert_permute_failed(
               cdata[nblk-1].get_hdl(),
               &node, &alloc, blksz
               );

#else
         int m = node.get_nrow();
         int n = node.get_ncol();
         size_t ldl = align_lda<T>(m);
         T *lcol = node.lcol;
         int *perm = node.perm;
         int num_elim = node.nelim;

         CopyBackup<T, Allocator> &backup = *node.backup; 
         ColumnData<T, IntAlloc> &cdata = *node.cdata;
         
         backup.release_all_memory(); 

         if (num_elim < n)
            permute_failed (
                  m, n, perm, lcol, ldl,
                  num_elim, 
                  cdata, blksz,
                  alloc);         
#endif

      }

      ////////////////////////////////////////////////////////////////////////////////   
      // factor_indef_app_async
      //
      /// @brief Perform the LDLT factorization of a matrix using a
      /// APTP pivoting strategy. 
      /// @note This call is asynchronous.
      static
      void factor_indef_app_async(
            int const m, int const n, int* perm, T* a,
            int const lda, T* d, ColumnData<T,IntAlloc>& cdata, Backup& backup,
            struct cpu_factor_options& options, int const block_size,
            T const beta, T* upd, int const ldupd, std::vector<spral::ssids::cpu::Workspace>& work,
            Allocator const& alloc, int& next_elim, int const from_blk=0
            ) {

         typedef spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> BlockSpec;
         

         int const nblk = calc_nblk(n, block_size);
         int const mblk = calc_nblk(m, block_size);

         /* Setup */
         // int next_elim = from_blk*block_size;
      
         int num_blocks = nblk*mblk;
         std::vector<BlockSpec> blocks;
         blocks.reserve(num_blocks);

         for(int jblk=0; jblk<nblk; jblk++) {
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
                  options/*, block_size*/, work, alloc);

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
            //             if(upd && mblk>nblk) {
            //                int uoffset = std::min(nblk*block_size, m) - n;
            //                T *upd2 = &upd[uoffset*(ldupd+1)];
            //                for(int jblk=nblk; jblk<mblk; ++jblk)
            //                   for(int iblk=jblk; iblk<mblk; ++iblk) {
            //                      T* upd_ij = &upd2[(jblk-nblk)*block_size*ldupd + 
            //                                        (iblk-nblk)*block_size];
            //                      {

            //                         BlockSpec ublk(iblk, jblk, m, n, cdata, &a[jblk*block_size*lda+iblk*block_size], lda, block_size);
            //                         BlockSpec isrc(iblk, blk, m, n, cdata, &a[blk*block_size*lda+iblk*block_size], lda, block_size);
            //                         BlockSpec jsrc(jblk, blk, m, n, cdata, &a[blk*block_size*lda+jblk*block_size], lda, block_size);

            //                         udpate_contrib_task(
            //                               // isrc, jsrc, ublk,
            //                               blocks[blk*mblk+iblk], blocks[blk*mblk+jblk],
            //                               blocks[jblk*mblk+iblk],
            //                               beta, upd_ij, ldupd,
            //                               work
            //                               );

            // // #if defined(SPLDLT_USE_STARPU)
            // //             starpu_task_wait_for_all();
            // // #endif

            //                      }
            //                   }
            //             }

            // #if defined(SPLDLT_USE_STARPU)
            //             starpu_task_wait_for_all();
            // #endif

         } // loop on block columns
         
         
      }


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
                          struct cpu_factor_options& options/*, PivotMethod pivot_method*/, int block_size, T beta, T* upd, int ldupd, 
                          spral::ssids::cpu::Workspace& work, Allocator const& alloc=Allocator()) {

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

} /* namespace spldlt */
