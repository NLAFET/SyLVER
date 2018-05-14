#pragma once

#include "kernels/ldlt_app.hxx"
#include "kernels/factor_indef.hxx"
// #include "StarPU/kernels.hxx"
#include "SymbolicFront.hxx"
#include "NumericFront.hxx"

#include <starpu.h>

namespace spldlt { namespace starpu {

      using namespace spldlt::ldlt_app_internal;

      /* factor_block_app StarPU task
         
       */
      
      /* factor_block_app StarPU codelet */
      // static
      extern struct starpu_codelet cl_factor_block_app;

      /* factor_block_app CPU kernel */
      template<typename T,
               int iblksz,
               typename Backup,
               typename IntAlloc,
               typename Allocator>
      void
      factor_block_app_cpu_func(void *buffers[], void *cl_arg) {

         T *a_kk = (T *)STARPU_MATRIX_GET_PTR(buffers[0]); // Get block pointer
         unsigned lda = STARPU_MATRIX_GET_LD(buffers[0]); // Get leading dimensions

         int m, n; // dimension of node
         int blk; // block index
         int *next_elim;
         int *perm;
         T *d;
         ColumnData<T,IntAlloc> *cdata = nullptr;
         // Column<T> *col = nullptr;
         Backup *backup = nullptr;
         struct cpu_factor_options *options = nullptr;
         std::vector<spral::ssids::cpu::Workspace> *work = nullptr;
         // spral::ssids::cpu::Workspace *work = nullptr;
         Allocator *alloc = nullptr;

         // printf("[factor_block_app_cpu_func]\n");
         int id = starpu_worker_get_id();
         
         starpu_codelet_unpack_args (
               cl_arg,
               &m, &n, &blk,
               &next_elim, &perm, &d,
               &cdata, &backup,
               &options, &work, &alloc);

         spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> dblk(blk, blk, m, n, *cdata, a_kk, lda, options->cpu_block_size);

         // printf("[factor_block_app_cpu_func] iblksz: %d, blksz: %d\n", iblksz, options->cpu_block_size);
         // printf("[factor_block_app_cpu_func] blk: %d, blksz: %d\n", blk, options->cpu_block_size);

         bool abort=false;
         
         dblk.backup(*backup);
         int thread_num = 0;
         // Perform actual factorization
         int nelim = dblk.template factor<Allocator>(
               *next_elim, perm, d, *options, (*work)[id], *alloc
               );
         if(nelim<0) 
            abort=true;

         // Init threshold check (non locking => task dependencies)
         (*cdata)[blk].init_passed(nelim);
      }
      
      template<typename T, 
               typename Backup, 
               typename IntAlloc, 
               typename Allocator>
      void
      insert_factor_block_app_task (
            starpu_data_handle_t a_kk_hdl,
            starpu_data_handle_t col_hdl,
            int m, int n, int blk,
            int *next_elim, int *perm, T* d,
            ColumnData<T,IntAlloc> *cdata, Backup *backup,
            struct cpu_factor_options *options,
            std::vector<spral::ssids::cpu::Workspace> *work, Allocator *alloc) {
         
         int ret;

         // printf("[insert_factor_block_app_task] %s\n", cl_factor_block_app.name);
         
         ret = starpu_task_insert(
               &cl_factor_block_app,
               STARPU_RW, a_kk_hdl,
               STARPU_RW, col_hdl,
               STARPU_VALUE, &m, sizeof(int),
               STARPU_VALUE, &n, sizeof(int),
               STARPU_VALUE, &blk, sizeof(int),
               STARPU_VALUE, &next_elim, sizeof(int*),
               STARPU_VALUE, &perm, sizeof(int*),
               STARPU_VALUE, &d, sizeof(T*),
               STARPU_VALUE, &cdata, sizeof(ColumnData<T,IntAlloc>*),
               // STARPU_VALUE, &col, sizeof(Column<T>*),
               STARPU_VALUE, &backup, sizeof(Backup*),
               STARPU_VALUE, &options, sizeof(struct cpu_factor_options *),
               STARPU_VALUE, &work, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
               STARPU_VALUE, &alloc, sizeof(Allocator*),
               0);

         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
      }

      /* applyN_block_app StarPU codelet */
      // static
      extern struct starpu_codelet cl_applyN_block_app;      

      /*  applyN_block_app StarPU task
         
       */
      template<typename T,
               int iblksz,
               typename Backup,
               typename IntAlloc>
      void
      applyN_block_app_cpu_func(void *buffers[], void *cl_arg) {

         T *a_kk = (T *)STARPU_MATRIX_GET_PTR(buffers[0]); // Get block pointer
         unsigned ld_a_kk = STARPU_MATRIX_GET_LD(buffers[0]); // Get leading dimensions

         T *a_ik = (T *)STARPU_MATRIX_GET_PTR(buffers[1]); // Get subdiagonal block pointer
         unsigned ld_a_ik = STARPU_MATRIX_GET_LD(buffers[1]); // Get leading dimensions
         
         int m, n; // node's dimensions
         int blk; // column index
         int iblk; // row index of subdiagonal block
         ColumnData<T,IntAlloc> *cdata = nullptr;
         Backup *backup = nullptr;
         struct cpu_factor_options *options = nullptr;

         // printf("[applyN_block_app_cpu_func]\n");

         starpu_codelet_unpack_args (
               cl_arg,
               &m, &n,
               &blk, &iblk,
               &cdata, &backup,
               &options);

         spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> dblk(blk, blk, m, n, *cdata, a_kk, ld_a_kk, options->cpu_block_size);
         spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> rblk(iblk, blk, m, n, *cdata, a_ik, ld_a_ik, options->cpu_block_size);
         
         // printf("[applyN_block_app_cpu_func] m = %d, n = %d\n", m, n);
         // printf("[applyN_block_app_cpu_func] ld_akk = %d, ld_aik = %d\n", ld_a_kk, ld_a_ik);

         // Apply column permutation from factorization of dblk and in
         // the process, store a (permuted) copy for recovery in case of
         // a failed column
         rblk.apply_cperm_and_backup(*backup);
         // Perform elimination and determine number of rows in block
         // passing a posteori threshold pivot test         
         int blkpass = rblk.apply_pivot_app(dblk, options->u, options->small);
                  // Update column's passed pivot count
         (*cdata)[blk].update_passed(blkpass);
      }

      /* applyT_block_app StarPU codelet */
      // static
      extern struct starpu_codelet cl_applyT_block_app;

      template<typename T, 
               typename Backup, 
               typename IntAlloc>
      void 
      insert_applyN_block_app(
            starpu_data_handle_t a_kk_hdl,
            starpu_data_handle_t a_ik_hdl,
            starpu_data_handle_t col_hdl,
            int m, int n, int blk, int iblk,            
            ColumnData<T,IntAlloc> *cdata, Backup *backup,
            struct cpu_factor_options *options) {

         int ret;

         ret = starpu_task_insert(
               &cl_applyN_block_app,
               STARPU_R, a_kk_hdl,
               STARPU_RW, a_ik_hdl,
               STARPU_R, col_hdl,
               STARPU_VALUE, &m, sizeof(int),
               STARPU_VALUE, &n, sizeof(int),
               STARPU_VALUE, &blk, sizeof(int),
               STARPU_VALUE, &iblk, sizeof(int),
               STARPU_VALUE, &cdata, sizeof(ColumnData<T,IntAlloc>*),
               STARPU_VALUE, &backup, sizeof(Backup*),
               STARPU_VALUE, &options, sizeof(struct cpu_factor_options *),
               0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");         

      }

      /*  applyT_block_app StarPU task
         
       */
      template<typename T,
               int iblksz,
               typename Backup, 
               typename IntAlloc>
      void
      applyT_block_app_cpu_func(void *buffers[], void *cl_arg) {

         T *a_kk = (T *)STARPU_MATRIX_GET_PTR(buffers[0]); // Get diagonal block pointer
         unsigned ld_a_kk = STARPU_MATRIX_GET_LD(buffers[0]); // Get leading dimensions
         
         T *a_kj = (T *)STARPU_MATRIX_GET_PTR(buffers[1]); // Get subdiagonal block pointer
         unsigned ld_a_kj = STARPU_MATRIX_GET_LD(buffers[1]); // Get leading dimensions

         int m, n; // node's dimensions
         int blk; // column index
         int jblk; // column index of leftdiagonal block     
         ColumnData<T,IntAlloc> *cdata = nullptr;
         Backup *backup = nullptr;
         struct cpu_factor_options *options = nullptr;

         starpu_codelet_unpack_args (
               cl_arg,
               &m, &n,
               &blk, &jblk,
               &cdata, &backup,
               &options);

         spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> dblk(blk, blk, m, n, *cdata, a_kk, ld_a_kk, options->cpu_block_size);
         spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> cblk(blk, jblk, m, n, *cdata, a_kj, ld_a_kj, options->cpu_block_size);

         // printf("[applyT_block_app_cpu_func] m = %d, n = %d\n", m, n);
         
         // Apply row permutation from factorization of dblk and in
         // the process, store a (permuted) copy for recovery in case of
         // a failed column
         cblk.apply_rperm_and_backup(*backup);
         // Perform elimination and determine number of rows in block
         // passing a posteori threshold pivot test
         int blkpass = cblk.apply_pivot_app(
               dblk, options->u, options->small
               );
         // Update column's passed pivot count
         (*cdata)[blk].update_passed(blkpass);
      }

      template<typename T, 
               typename Backup, 
               typename IntAlloc>
      void
      insert_applyT_block_app(
            starpu_data_handle_t a_kk_hdl,
            starpu_data_handle_t a_jk_hdl,
            starpu_data_handle_t col_hdl,
            int m, int n, int blk, int jblk,            
            ColumnData<T,IntAlloc> *cdata, Backup *backup,
            struct cpu_factor_options *options) {

         int ret;

         ret = starpu_task_insert(
               &cl_applyT_block_app,
               STARPU_R, a_kk_hdl,
               STARPU_RW, a_jk_hdl,
               STARPU_R, col_hdl,
               STARPU_VALUE, &m, sizeof(int),
               STARPU_VALUE, &n, sizeof(int),
               STARPU_VALUE, &blk, sizeof(int),
               STARPU_VALUE, &jblk, sizeof(int),
               STARPU_VALUE, &cdata, sizeof(ColumnData<T,IntAlloc>*),
               STARPU_VALUE, &backup, sizeof(Backup*),
               STARPU_VALUE, &options, sizeof(struct cpu_factor_options *),
               0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
      }

      /* restore_block_app StarPU codelet */
      extern struct starpu_codelet cl_restore_block_app;      

      /*  restore_block_app StarPU task
         
       */
      template<typename T,
               int iblksz,
               typename Backup, 
               typename IntAlloc>
      void
      restore_block_app_cpu_func(void *buffers[], void *cl_arg) {

         T *a_ij = (T *)STARPU_MATRIX_GET_PTR(buffers[0]); // Get diagonal block pointer
         unsigned ld_a_ij = STARPU_MATRIX_GET_LD(buffers[0]); // Get leading dimensions

         int m, n; // node's dimensions
         int iblk; // destination block's row index
         int jblk; // destination block's column index     
         int blk; // source block's column index     

         ColumnData<T,IntAlloc> *cdata = nullptr;
         Backup *backup = nullptr;

         int blksz;

         starpu_codelet_unpack_args (
               cl_arg,
               &m, &n,
               &iblk, &jblk, &blk,
               &cdata, &backup,
               &blksz);

         spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> ublk(iblk, jblk, m, n, *cdata, a_ij, ld_a_ij, blksz);

         ublk.restore_if_required(*backup, blk);
      }

      /* updateN_block_app StarPU codelet */
      // static
      extern struct starpu_codelet cl_updateN_block_app;      

      /*  updateN_block_app StarPU task
         
       */
      template<typename T,
               int iblksz,
               typename Backup, 
               typename IntAlloc>
      void 
      updateN_block_app_cpu_func(void *buffers[], void *cl_arg) {

         T *a_ik = (T *)STARPU_MATRIX_GET_PTR(buffers[0]); // Get diagonal block pointer
         unsigned ld_a_ik = STARPU_MATRIX_GET_LD(buffers[0]); // Get leading dimensions

         T *a_jk = (T *)STARPU_MATRIX_GET_PTR(buffers[1]); // Get diagonal block pointer
         unsigned ld_a_jk = STARPU_MATRIX_GET_LD(buffers[1]); // Get leading dimensions

         T *a_ij = (T *)STARPU_MATRIX_GET_PTR(buffers[2]); // Get diagonal block pointer
         unsigned ld_a_ij = STARPU_MATRIX_GET_LD(buffers[2]); // Get leading dimensions
         
         int id = starpu_worker_get_id();
         // printf("[updateN_block_app_cpu_func] id: %d nworker: %d\n", id, starpu_worker_get_count());

         int m, n; // node's dimensions
         int iblk; // destination block's row index
         int jblk; // destination block's column index     
         int blk; // source block's column index     

         ColumnData<T,IntAlloc> *cdata = nullptr;
         Backup *backup = nullptr;
         
         T beta;
         T* upd = nullptr;
         int ldupd;

         std::vector<spral::ssids::cpu::Workspace> *work;
         // Workspace *work;
         // struct cpu_factor_options *options = nullptr;
         int blksz;

         starpu_codelet_unpack_args (
               cl_arg,
               &m, &n,
               &iblk, &jblk, &blk,
               &cdata, &backup,
               &beta, &upd, &ldupd,
               &work, &blksz /*&options*/);

         spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> ublk(iblk, jblk, m, n, *cdata, a_ij, ld_a_ij, blksz);
         spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> isrc(iblk, blk, m, n, *cdata, a_ik, ld_a_ik, blksz);
         spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> jsrc(jblk, blk, m, n, *cdata, a_jk, ld_a_jk, blksz);

         // If we're on the block col we've just eliminated, restore
         // any failed cols and release resources storing backup
         ublk.restore_if_required(*backup, blk);
         // Perform actual update
         ublk.update(isrc, jsrc, (*work)[id],
                     beta, upd, ldupd);
      }

      template<typename T, 
               typename Backup, 
               typename IntAlloc>
      void insert_updateN_block_app(
            starpu_data_handle_t a_ik_hdl,
            starpu_data_handle_t a_jk_hdl,
            starpu_data_handle_t a_ij_hdl,
            starpu_data_handle_t col_hdl,
            int m, int n, int iblk, int jblk, int blk,
            ColumnData<T,IntAlloc> *cdata, Backup *backup,
            T beta, T* upd, int ldupd,
            std::vector<spral::ssids::cpu::Workspace> *work, int blksz/*struct cpu_factor_options *options*/) {

         // printf("[insert_updateN_block_app] blk: %d, iblk: %d, jblk: %d\n", blk, iblk, jblk);

         int ret;

         ret = starpu_task_insert(
               &cl_updateN_block_app,
               STARPU_R, a_ik_hdl,
               STARPU_R, a_jk_hdl,
               STARPU_RW, a_ij_hdl,
               STARPU_R, col_hdl,
               STARPU_VALUE, &m, sizeof(int),
               STARPU_VALUE, &n, sizeof(int),
               STARPU_VALUE, &iblk, sizeof(int),
               STARPU_VALUE, &jblk, sizeof(int),
               STARPU_VALUE, &blk, sizeof(int),
               STARPU_VALUE, &cdata, sizeof(ColumnData<T,IntAlloc>*),
               STARPU_VALUE, &backup, sizeof(Backup*),
               STARPU_VALUE, &beta, sizeof(T),
               STARPU_VALUE, &upd, sizeof(T*),
               STARPU_VALUE, &ldupd, sizeof(int),
               STARPU_VALUE, &work, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
               STARPU_VALUE, &blksz, sizeof(int),
               // STARPU_VALUE, &options, sizeof(struct cpu_factor_options *),
               0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
         
      }

      /* updateT_block_app StarPU codelet */
      // static
      extern struct starpu_codelet cl_updateT_block_app;      

      /*  updateT_block_app StarPU task
         
       */
      template<typename T,
               int iblksz,
               typename Backup, 
               typename IntAlloc>
      void 
      updateT_block_app_cpu_func(void *buffers[], void *cl_arg) {

         T *a_ik = (T *)STARPU_MATRIX_GET_PTR(buffers[0]); // Get diagonal block pointer
         unsigned ld_a_ik = STARPU_MATRIX_GET_LD(buffers[0]); // Get leading dimensions
         
         T *a_jk = (T *)STARPU_MATRIX_GET_PTR(buffers[1]); // Get diagonal block pointer
         unsigned ld_a_jk = STARPU_MATRIX_GET_LD(buffers[1]); // Get leading dimensions

         T *a_ij = (T *)STARPU_MATRIX_GET_PTR(buffers[2]); // Get diagonal block pointer
         unsigned ld_a_ij = STARPU_MATRIX_GET_LD(buffers[2]); // Get leading dimensions

         int id = starpu_worker_get_id();

         int m, n; // node's dimensions
         int iblk; // destination block's row index
         int jblk; // destination block's column index     
         int blk; // source block's column index     
         int isrc_row, isrc_col;

         ColumnData<T,IntAlloc> *cdata = nullptr;
         Backup *backup = nullptr;
         std::vector<spral::ssids::cpu::Workspace> *work; 
         // Workspace *work;
         // struct cpu_factor_options *options = nullptr;
         int blksz;

         starpu_codelet_unpack_args (
               cl_arg,
               &m, &n,
               &isrc_row, &isrc_col,
               &iblk, &jblk, &blk,
               &cdata, &backup,
               &work, &blksz /*&options*/);

         spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> ublk(iblk, jblk, m, n, *cdata, a_ij, ld_a_ij, blksz);
         spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> isrc(isrc_row, isrc_col, m, n, *cdata, a_ik, ld_a_ik, blksz);
         spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> jsrc(blk, jblk, m, n, *cdata, a_jk, ld_a_jk, blksz);

         // If we're on the block row we've just eliminated, restore
         // any failed rows and release resources storing backup
         ublk.restore_if_required(*backup, blk);
         // Perform actual update
         ublk.update(isrc, jsrc, (*work)[id]);
      }

      template<typename T, 
               typename Backup, 
               typename IntAlloc>
      void insert_updateT_block_app(
            starpu_data_handle_t a_ik_hdl,
            starpu_data_handle_t a_jk_hdl,
            starpu_data_handle_t a_ij_hdl,
            starpu_data_handle_t col_hdl,
            int m, int n, int isrc_row, int isrc_col,
            int iblk, int jblk, int blk,
            ColumnData<T,IntAlloc> *cdata, Backup *backup,
            std::vector<spral::ssids::cpu::Workspace> *work, int blksz/*struct cpu_factor_options *options*/) {

         int ret;

         ret = starpu_task_insert(
               &cl_updateT_block_app,
               STARPU_R, a_ik_hdl,
               STARPU_R, a_jk_hdl,
               STARPU_RW, a_ij_hdl,
               STARPU_R, col_hdl,
               STARPU_VALUE, &m, sizeof(int),
               STARPU_VALUE, &n, sizeof(int),
               STARPU_VALUE, &isrc_row, sizeof(int),
               STARPU_VALUE, &isrc_col, sizeof(int),
               STARPU_VALUE, &iblk, sizeof(int),
               STARPU_VALUE, &jblk, sizeof(int),
               STARPU_VALUE, &blk, sizeof(int),
               STARPU_VALUE, &cdata, sizeof(ColumnData<T,IntAlloc>*),
               STARPU_VALUE, &backup, sizeof(Backup*),
               STARPU_VALUE, &work, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
               STARPU_VALUE, &blksz, sizeof(int),
               // STARPU_VALUE, &options, sizeof(struct cpu_factor_options *),
               0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");         
      }

      /* adjust StarPU codelet */
      // static
      extern struct starpu_codelet cl_adjust;      
      
      template<typename T, typename IntAlloc>
      void 
      adjust_cpu_func(void *buffers[], void *cl_arg) {

         int blk; // column index
         int *next_elim  = nullptr;
         ColumnData<T,IntAlloc> *cdata = nullptr;

         starpu_codelet_unpack_args (
               cl_arg,
               &blk,
               &next_elim, &cdata);

         // Adjust column once all applys have finished and we know final
         // number of passed columns.

         (*cdata)[blk].adjust(*next_elim);
         
      }

      template<typename T, typename IntAlloc>
      void
      insert_adjust(
            starpu_data_handle_t col_hdl,
            int blk,
            int *next_elim,
            ColumnData<T,IntAlloc> *cdata) {

         int ret;

         ret = starpu_task_insert(
               &cl_adjust,
               STARPU_RW, col_hdl,
               STARPU_VALUE, &blk, sizeof(int),
               STARPU_VALUE, &next_elim, sizeof(int*),
               STARPU_VALUE, &cdata, sizeof(ColumnData<T,IntAlloc>*),
               0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
      }

      ////////////////////////////////////////////////////////////////////////////////      
      // Update contribution blocks
      
      template <typename T, typename IntAlloc, typename PoolAlloc>
      void update_contrib_block_app_cpu_func(void *buffers[], void *cl_arg) {

         T *upd = (T *)STARPU_MATRIX_GET_PTR(buffers[0]);
         unsigned ldupd = STARPU_MATRIX_GET_LD(buffers[0]); // Get leading dimensions
         unsigned updm = STARPU_MATRIX_GET_NX(buffers[0]);
         unsigned updn = STARPU_MATRIX_GET_NY(buffers[0]);

         const T *lik = (T *)STARPU_MATRIX_GET_PTR(buffers[1]);
         unsigned ld_lik = STARPU_MATRIX_GET_LD(buffers[1]); // Get leading dimensions

         const T *ljk = (T *)STARPU_MATRIX_GET_PTR(buffers[2]);
         unsigned ld_ljk = STARPU_MATRIX_GET_LD(buffers[2]); // Get leading dimensions
                  
         NumericFront<T, PoolAlloc> *node = nullptr;
         int k, i, j;
         std::vector<spral::ssids::cpu::Workspace> *workspaces;
         int blksz;

         starpu_codelet_unpack_args(
               cl_arg, &node, &k, &i, &j, &workspaces, &blksz);

         int workerid = starpu_worker_get_id();
         spral::ssids::cpu::Workspace &work = (*workspaces)[workerid];

         update_contrib_block_app<T, IntAlloc, PoolAlloc>(
               *node, k, i, j,
               lik, ld_lik, ljk, ld_ljk,
               updm, updn, upd, ldupd,
               work);
      }

      extern struct starpu_codelet cl_update_contrib_block_app;

      // insert_udpate_contrib_block_indef
      template <typename T, typename PoolAlloc>
      void insert_update_contrib_block_app(
            starpu_data_handle_t upd_hdl,
            starpu_data_handle_t lik_hdl,
            starpu_data_handle_t ljk_hdl,
            starpu_data_handle_t col_hdl, // Symbolic handle on block-column k
            starpu_data_handle_t contrib_hdl, // Contribution blocks symbolic handle
            NumericFront<T, PoolAlloc> *node,
            int k, int i, int j,
            std::vector<spral::ssids::cpu::Workspace> *workspaces,
            int blksz, int prio
            ) {

         // printf("[insert_udpate_contrib_block_indef]\n");

         int ret;
         
         ret = starpu_insert_task(
               &cl_update_contrib_block_app,
               STARPU_RW, upd_hdl,
               STARPU_R, lik_hdl,
               STARPU_R, ljk_hdl,
               STARPU_R, col_hdl,
               STARPU_R, contrib_hdl, // Contribution blocks symbolic handle
               STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
               STARPU_VALUE, &k, sizeof(int),
               STARPU_VALUE, &i, sizeof(int),
               STARPU_VALUE, &j, sizeof(int),
               STARPU_VALUE, &workspaces, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
               STARPU_VALUE, &blksz, sizeof(int),
               STARPU_PRIORITY, prio,
               0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

      }

      ////////////////////////////////////////////////////////////////////////////////
      // permute failed

      // CPU kernel
      template <typename T, typename IntAlloc, typename PoolAlloc>
      void permute_failed_cpu_func(void *buffers[], void *cl_arg) {
         
         NumericFront<T, PoolAlloc> *node = nullptr;
         PoolAlloc *alloc = nullptr;

         starpu_codelet_unpack_args(
               cl_arg, &node, &alloc);

         // printf("[permute_failed_cpu_func]\n");

         int n = node->get_ncol();

         if (node->nelim < n) {

            CopyBackup<T, PoolAlloc> &backup = *node->backup; 

            backup.release_all_memory(); 
         
            int m = node->get_nrow();
            int ldl = node->get_ldl();
            ColumnData<T, IntAlloc> &cdata = *node->cdata;
            bool const debug = false;
            int blksz = node->blksz;
            
            FactorSymIndef
               <T, INNER_BLOCK_SIZE, CopyBackup<T, PoolAlloc>, debug, PoolAlloc>
               ::permute_failed (
                     m, n, node->perm, node->lcol, ldl,
                     node->nelim, 
                     cdata, blksz,
                     *alloc);
         }
                  
      }

      // StarPU codelet
      extern struct starpu_codelet cl_permute_failed;

      template <typename T, typename PoolAlloc>
      void insert_permute_failed(
            starpu_data_handle_t *col_hdls, int nhdl,
            NumericFront<T, PoolAlloc> *node,
            PoolAlloc *pool_alloc
            ) {
         
         int ret;

         struct starpu_data_descr *descrs = new starpu_data_descr[nhdl];

         int nh = 0;
         for (int i=0; i<nhdl; i++) {
            descrs[nh].handle = col_hdls[i]; descrs[nh].mode = STARPU_RW;
            nh++;
         }

         ret = starpu_insert_task(
               &cl_permute_failed,
               STARPU_DATA_MODE_ARRAY, descrs, nh,
               STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
               STARPU_VALUE, &pool_alloc, sizeof(PoolAlloc*),
               0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

         delete[] descrs;
      }

      ////////////////////////////////////////////////////////////////////////////////
      // factor_front_indef_secondpass_nocontrib

      // CPU kernel      
      template <typename T, typename PoolAlloc>      
      void factor_front_indef_failed_cpu_func(void *buffers[], void *cl_arg) {
         
         NumericFront<T, PoolAlloc> *node = nullptr;
         std::vector<spral::ssids::cpu::Workspace> *workspaces = nullptr;
         struct cpu_factor_options *options = nullptr;
         std::vector<ThreadStats> *worker_stats = nullptr;

         starpu_codelet_unpack_args(
               cl_arg, &node, &workspaces, &options, &worker_stats);

         int workerid = starpu_worker_get_id();
         spral::ssids::cpu::Workspace& work = (*workspaces)[workerid];
         spral::ssids::cpu::ThreadStats& stats = (*worker_stats)[workerid];

         factor_front_indef_failed(*node, work, *options, stats);

      }

      // SarPU kernel
      extern struct starpu_codelet cl_factor_front_indef_failed;

      template <typename T, typename PoolAlloc>
      void insert_factor_front_indef_failed(
            starpu_data_handle_t col_hdl,
            starpu_data_handle_t contrib_hdl,
            NumericFront<T, PoolAlloc> *node,
            std::vector<spral::ssids::cpu::Workspace> *workspaces,
            struct cpu_factor_options *options,
            std::vector<ThreadStats> *worker_stats
            ) {

         int ret;
         
         ret = starpu_insert_task(
               &cl_factor_front_indef_failed,
               STARPU_RW, col_hdl,
               STARPU_RW, contrib_hdl,
               STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
               STARPU_VALUE, &workspaces, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
               STARPU_VALUE, &options, sizeof(struct cpu_factor_options*),
               STARPU_VALUE, &worker_stats, sizeof(std::vector<ThreadStats>*),
               0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
               
      }

      ////////////////////////////////////////////////////////////////////////////////
      // nelim_sync

      template <typename T, typename PoolAlloc>
      void nelim_sync_cpu_func(void *buffers[], void *cl_arg) {
         // printf("[nelim_sync_cpu_func]\n");
      }

      // SarPU kernel
      extern struct starpu_codelet cl_nelim_sync;

      void insert_nelim_sync(
            starpu_data_handle_t node_hdl,
            int nodeidx) {

         int ret;
         
         starpu_tag_t tag1 = (starpu_tag_t) (2*nodeidx);
         starpu_tag_t tag2 = (starpu_tag_t) (2*nodeidx+1);
         starpu_tag_declare_deps(tag2, 1, tag1);
         // starpu_tag_declare_deps(tag2, 1, 0);

         // printf("[insert_nelim_sync] nodeidx = %d, tag1 = %d, , tag2 = %d\n", 
         //        nodeidx, tag1, tag2);

         // ret = starpu_task_insert(
         //       &cl_nelim_sync,
         //       STARPU_TAG, tag2,
         //       STARPU_RW, node_hdl,
         //       0);
         // STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

         struct starpu_task *task = starpu_task_create();
         task->cl = &cl_nelim_sync; 
         task->use_tag = 1;
         task->tag_id = tag2;
         task->handles[0] = node_hdl;
         ret = starpu_task_submit(task);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

      }

      ////////////////////////////////////////////////////////////////////////////////
      // assemble_contrib_sync

      template <typename T, typename PoolAlloc>
      void assemble_contrib_sync_cpu_func(void *buffers[], void *cl_arg) {
         // printf("[assemble_contrib_sync_cpu_func]\n");
      }

      // SarPU kernel
      extern struct starpu_codelet cl_assemble_contrib_sync;

      // template <typename T, typename PoolAlloc>
      void insert_assemble_contrib_sync(
            starpu_data_handle_t contrib_hdl,
            int nodeidx) {

         int ret;

         starpu_tag_t tag1 = (starpu_tag_t) (2*nodeidx);
         starpu_tag_t tag2 = (starpu_tag_t) (2*nodeidx+1);
         // starpu_tag_declare_deps(tag2, 1, tag1);

         // printf("[insert_assemble_contrib_sync] nodeidx = %d, tag1 = %d, , tag2 = %d\n", 
         //        nodeidx, tag1, tag2);

         // ret = starpu_task_insert(
         //       &cl_assemble_contrib_sync,
         //       STARPU_TAG, tag1,
         //       STARPU_RW, contrib_hdl,
         //       0);
         // STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

         struct starpu_task *task = starpu_task_create();
         task->cl = &cl_assemble_contrib_sync; 
         task->use_tag = 1;
         task->tag_id = tag1;
         task->handles[0] = contrib_hdl;
         ret = starpu_task_submit(task);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

      }

      ////////////////////////////////////////////////////////////////////////////////
      // factor_sync

      // template <typename T, typename PoolAlloc>
      // void factor_sync_cpu_func(void *buffers[], void *cl_arg) {
      //    // printf("[factor_sync_cpu_func]\n");
      // }
      
      // // SarPU kernel
      // extern struct starpu_codelet cl_factor_sync;

      // template <typename T, typename PoolAlloc>
      // void insert_factor_sync(
      //       starpu_data_handle_t col_hdl,
      //       NumericFront<T, PoolAlloc>& node
      //       ) {

      //    // printf("[insert_factor_sync]\n");

      //    starpu_tag_t tag1 = (starpu_tag_t) (2*node.symb.idx);
      //    starpu_tag_t tag2 = (starpu_tag_t) (2*node.symb.idx+1);
      //    // starpu_tag_declare_deps(tag2, 1, tag1);
         
      //    int ret;

      //    struct starpu_task *taskA = starpu_task_create();
      //    taskA->cl = &cl_factor_sync;
      //    taskA->use_tag = 1;
      //    taskA->tag_id = tag1;
      //    taskA->handles[0] = col_hdl;
      //    ret = starpu_task_submit(taskA); 
      //    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

      //    // ret = starpu_insert_task(
      //    //       &cl_factor_sync,
      //    //       STARPU_TAG, tag1,
      //    //       STARPU_RW, contrib_hdl,
      //    //       0);
      //    // STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
         
      // }
      
      ////////////////////////////////////////////////////////////////////////////////

      /* As it is not possible to statically intialize codelet in C++,
         we do it via this function */
      template <typename T, int iblksz, 
                typename Backup, 
                typename Allocator>
      void codelet_init_indef() {
         
         // printf("[codelet_init_indef]\n");

         typedef typename std::allocator_traits<Allocator>::template rebind_alloc<int> IntAlloc;
      
         // Init codelet for posdef tasks 
         // codelet_init<T, PoolAlloc>();

         // Initialize factor_block_app StarPU codelet
         starpu_codelet_init(&cl_factor_block_app);
         cl_factor_block_app.where = STARPU_CPU;
         cl_factor_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_factor_block_app.name = "FACOTR_BLK_APP";
         cl_factor_block_app.cpu_funcs[0] = factor_block_app_cpu_func<T, iblksz, Backup, IntAlloc, Allocator>;

         // printf("[codelet_init_indef] %s\n", cl_factor_block_app.name);

         // Initialize applyN_block_app StarPU codelet
         starpu_codelet_init(&cl_applyN_block_app);
         cl_applyN_block_app.where = STARPU_CPU;
         cl_applyN_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_applyN_block_app.name = "APPLYN_BLK_APP";
         cl_applyN_block_app.cpu_funcs[0] = applyN_block_app_cpu_func<T, iblksz, Backup, IntAlloc>;
         
         // Initialize applyT_block_app StarPU codelet
         starpu_codelet_init(&cl_applyT_block_app);
         cl_applyT_block_app.where = STARPU_CPU;
         cl_applyT_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_applyT_block_app.name = "APPLYT_BLK_APP";
         cl_applyT_block_app.cpu_funcs[0] = applyT_block_app_cpu_func<T, iblksz, Backup, IntAlloc>;

         // Initialize updateN_block_app StarPU codelet
         starpu_codelet_init(&cl_updateN_block_app);
         cl_updateN_block_app.where = STARPU_CPU;
         cl_updateN_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_updateN_block_app.name = "UPDATEN_BLK_APP";
         cl_updateN_block_app.cpu_funcs[0] = updateN_block_app_cpu_func<T, iblksz, Backup, IntAlloc>;

         // Initialize updateT_block_app StarPU codelet
         starpu_codelet_init(&cl_updateT_block_app);
         cl_updateT_block_app.where = STARPU_CPU;
         cl_updateT_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_updateT_block_app.name = "UPDATET_BLK_APP";
         cl_updateT_block_app.cpu_funcs[0] = updateT_block_app_cpu_func<T, iblksz, Backup, IntAlloc>;

         // Initialize adjust StarPU codelet
         starpu_codelet_init(&cl_adjust);
         cl_adjust.where = STARPU_CPU;
         cl_adjust.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_adjust.name = "ADJUST";
         cl_adjust.cpu_funcs[0] = adjust_cpu_func<T, IntAlloc>;

         // Initialize update_contrib_block_indef StarPU codelet
         starpu_codelet_init(&cl_update_contrib_block_app);
         cl_update_contrib_block_app.where = STARPU_CPU;
         cl_update_contrib_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_update_contrib_block_app.name = "UPDATE_CONTRIB_BLOCK_APP";
         cl_update_contrib_block_app.cpu_funcs[0] = update_contrib_block_app_cpu_func<T, IntAlloc, Allocator>;

         // permute failed
         starpu_codelet_init(&cl_permute_failed);
         cl_permute_failed.where = STARPU_CPU;
         cl_permute_failed.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_permute_failed.name = "PERMUTE_FAILED";
         cl_permute_failed.cpu_funcs[0] = permute_failed_cpu_func<T, IntAlloc, Allocator>;

         // Initialize factor_front_indef_failed StarPU codelet
         starpu_codelet_init(&cl_factor_front_indef_failed);
         cl_factor_front_indef_failed.where = STARPU_CPU;
         cl_factor_front_indef_failed.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_factor_front_indef_failed.name = "FACTOR_FRONT_FAILED";
         cl_factor_front_indef_failed.cpu_funcs[0] = factor_front_indef_failed_cpu_func<T, Allocator>;

         // // Initialize factor_sync StarPU codelet
         // starpu_codelet_init(&cl_factor_sync);
         // // cl_factor_sync.where = STARPU_NOWHERE;
         // cl_factor_sync.where = STARPU_CPU;
         // cl_factor_sync.nbuffers = 1;// STARPU_VARIABLE_NBUFFERS;
         // cl_factor_sync.modes[0] = STARPU_RW;
         // // cl_factor_sync.modes[0] = STARPU_R;
         // cl_factor_sync.name = "FACTOR_SYNC";
         // cl_factor_sync.cpu_funcs[0] = factor_sync_cpu_func<T, Allocator>;

         // Initialize assemble_contrib_sync StarPU codelet
         starpu_codelet_init(&cl_assemble_contrib_sync);
         // cl_factor_sync.where = STARPU_NOWHERE;
         cl_assemble_contrib_sync.where = STARPU_CPU;
         cl_assemble_contrib_sync.nbuffers = 1;
         cl_assemble_contrib_sync.modes[0] = STARPU_RW;
         // cl_assemble_contrib_sync.modes[0] = STARPU_R;
         cl_assemble_contrib_sync.name = "ASSEMBLE_CONTRIB_SYNC";
         cl_assemble_contrib_sync.cpu_funcs[0] = assemble_contrib_sync_cpu_func<T, Allocator>;

         // Initialize factor_sync StarPU codelet
         starpu_codelet_init(&cl_nelim_sync);
         // cl_nelim_sync.where = STARPU_NOWHERE;
         cl_nelim_sync.where = STARPU_CPU;
         cl_nelim_sync.nbuffers = 1;
         // cl_nelim_sync.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_nelim_sync.modes[0] = STARPU_RW;
         // cl_nelim_sync.modes[0] = STARPU_R;
         cl_nelim_sync.name = "NELIM_SYNC";
         cl_nelim_sync.cpu_funcs[0] = nelim_sync_cpu_func<T, Allocator>;

      }
      
   }} /* namespaces spldlt::starpu  */
