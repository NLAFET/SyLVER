/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "sylver_ciface.hxx"
#if defined(SPLDLT_USE_GPU)
#include "kernels/gpu/factor_indef.hxx"
#endif
#include "kernels/ldlt_app.hxx"
#include "kernels/factor_indef.hxx"
// #include "StarPU/kernels.hxx"
#include "SymbolicFront.hxx"
#include "NumericFront.hxx"
#if defined(SPLDLT_USE_GPU)
#include "StarPU/cuda/kernels_indef.hxx"
#endif

// STD
#include <iostream>
#include <iomanip>

// StarPU
#include <starpu.h>
#if defined(SPLDLT_USE_GPU)
#include <starpu_cublas_v2.h>
#endif

namespace spldlt {
namespace starpu {
   
   using namespace spldlt::ldlt_app_internal;

   template<typename T>
   void print_block(int m, int n, const T *a, int lda) {
      for(int row=0; row<m; row++) {
         printf("%d:",row);
         for(int col=0; col<n; col++)
            printf(" %.2e", a[col*lda+row]);
         printf("\n");
      }
   }

   /* factor_block_app StarPU task
         
    */
   extern starpu_data_handle_t workspace_hdl;      
      
   /* factor_block_app StarPU codelet */
   // static
   extern struct starpu_codelet cl_factor_block_app;

   /* factor_block_app CPU kernel */
   template<typename T,
            typename Backup,
            typename IntAlloc,
            typename Allocator>
   void
   factor_block_app_cpu_func(void *buffers[], void *cl_arg) {

      T *a_kk = (T *)STARPU_MATRIX_GET_PTR(buffers[0]); // Get block pointer
      unsigned lda = STARPU_MATRIX_GET_LD(buffers[0]); // Get leading dimensions
      // Debug
      unsigned m_a_kk = STARPU_MATRIX_GET_NX(buffers[0]); // Get leading dimensions
      unsigned n_a_kk = STARPU_MATRIX_GET_NY(buffers[0]); // Get leading dimensions

      int m, n; // dimension of node
      int blk; // block index
      int *next_elim;
      int *perm;
      T *d;
      ColumnData<T,IntAlloc> *cdata = nullptr;
      // Column<T> *col = nullptr;
      Backup *backup = nullptr;
      sylver::options_t *options = nullptr;
      std::vector<sylver::inform_t> *worker_stats = nullptr;
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
            &options, &worker_stats, &work, &alloc);

      int workerid = starpu_worker_get_id();
      sylver::inform_t& inform = (*worker_stats)[workerid];

      spldlt::ldlt_app_internal::Block<T, INNER_BLOCK_SIZE, IntAlloc> dblk(blk, blk, m, n, *cdata, a_kk, lda, options->nb);

      bool abort=false;
         
      dblk.backup(*backup);
      int thread_num = 0;
      // Perform actual factorization
      int nelim = 0;
      try {
         nelim = dblk.template factor<Allocator>(
               *next_elim, perm, d, *options, (*work)[id], *alloc);
      } catch (spral::ssids::cpu::SingularError const&) {
         inform.flag = sylver::Flag::ERROR_SINGULAR;
      }
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
   insert_factor_block_app (
         starpu_data_handle_t a_kk_hdl,
         starpu_data_handle_t d_hdl,
         starpu_data_handle_t col_hdl,
         int m, int n, int blk,
         int *next_elim, int *perm, T* d,
         ColumnData<T,IntAlloc> *cdata, Backup *backup,
         sylver::options_t *options, std::vector<sylver::inform_t> *worker_stats,
         std::vector<spral::ssids::cpu::Workspace> *work, 
         Allocator *alloc,
         int prio) {

      int ret;
         
      ret = starpu_task_insert(
            &cl_factor_block_app,
            STARPU_RW, a_kk_hdl,
            STARPU_RW, d_hdl,
            STARPU_RW, col_hdl,
            STARPU_VALUE, &m, sizeof(int),
            STARPU_VALUE, &n, sizeof(int),
            STARPU_VALUE, &blk, sizeof(int),
            STARPU_VALUE, &next_elim, sizeof(int*),
            STARPU_VALUE, &perm, sizeof(int*),
            STARPU_VALUE, &d, sizeof(T*),
            STARPU_VALUE, &cdata, sizeof(ColumnData<T,IntAlloc>*),
            STARPU_VALUE, &backup, sizeof(Backup*),
            STARPU_VALUE, &options, sizeof(sylver::options_t *),
            STARPU_VALUE, &worker_stats, sizeof(std::vector<sylver::inform_t>*),
            STARPU_VALUE, &work, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
            STARPU_VALUE, &alloc, sizeof(Allocator*),
            STARPU_PRIORITY, prio,
            0);

      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
   }

   /* applyN_block_app StarPU codelet */
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
      unsigned m_a_ik = STARPU_MATRIX_GET_NX(buffers[1]); // Get leading dimensions
      unsigned n_a_ik = STARPU_MATRIX_GET_NY(buffers[1]); // Get leading dimensions
         
      int m, n; // node's dimensions
      int blk; // column index
      int iblk; // row index of subdiagonal block
      ColumnData<T,IntAlloc> *cdata = nullptr;
      Backup *backup = nullptr;
      sylver::options_t *options = nullptr;

      // printf("[applyN_block_app_cpu_func]\n");

      starpu_codelet_unpack_args (
            cl_arg,
            &m, &n,
            &blk, &iblk,
            &cdata, &backup,
            &options);

      spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> dblk(blk, blk, m, n, *cdata, a_kk, ld_a_kk, options->nb);
      spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> rblk(iblk, blk, m, n, *cdata, a_ik, ld_a_ik, options->nb);

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

   ////////////////////////////////////////
   // applyT

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
         sylver::options_t *options,
         int prio) {

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
            STARPU_VALUE, &options, sizeof(sylver::options_t *),
            STARPU_PRIORITY, prio,
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
      sylver::options_t *options = nullptr;

      starpu_codelet_unpack_args (
            cl_arg,
            &m, &n,
            &blk, &jblk,
            &cdata, &backup,
            &options);

      spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> dblk(blk, blk, m, n, *cdata, a_kk, ld_a_kk, options->nb);
      spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> cblk(blk, jblk, m, n, *cdata, a_kj, ld_a_kj, options->nb);
         
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
         sylver::options_t *options,
         int prio) {

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
            STARPU_VALUE, &options, sizeof(sylver::options_t *),
            STARPU_PRIORITY, prio,
            0);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
   }

   ////////////////////////////////////////
   // restore failed

   /* restore_block_app StarPU codelet */
   extern struct starpu_codelet cl_restore_failed_block_app;      

   /*  restore_block_app StarPU task
         
    */
   template<typename T,
            int iblksz,
            typename Backup, 
            typename IntAlloc>
   void
   restore_failed_block_app_cpu_func(void *buffers[], void *cl_arg) {

      typedef spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> BlockSpec;

      T *l_jk = (T *)STARPU_MATRIX_GET_PTR(buffers[0]); // Get diagonal block pointer
      unsigned ld_l_jk = STARPU_MATRIX_GET_LD(buffers[0]); // Get leading dimensions

      T *l_ij = (T *)STARPU_MATRIX_GET_PTR(buffers[1]); // Get diagonal block pointer
      unsigned ld_l_ij = STARPU_MATRIX_GET_LD(buffers[1]); // Get leading dimensions

      int id = starpu_worker_get_id();

      int m, n; // node's dimensions
      int iblk; // destination block's row index
      int jblk; // destination block's column index
      int elim_col; // Eliminated column      

      ColumnData<T,IntAlloc> *cdata = nullptr;
      Backup *backup = nullptr;

      std::vector<spral::ssids::cpu::Workspace> *workspaces;
      int blksz;

      starpu_codelet_unpack_args (
            cl_arg,
            &m, &n,
            &iblk, &jblk, 
            &elim_col,
            &cdata, 
            &backup,
            &workspaces, 
            &blksz);

      BlockSpec ublk(iblk, jblk, m, n, *cdata, l_ij, ld_l_ij, blksz);
      BlockSpec isrc(iblk, elim_col, m, n, *cdata, l_ij, ld_l_ij, blksz);         
      BlockSpec jsrc(jblk, elim_col, m, n, *cdata, l_jk, ld_l_jk, blksz);

      // Restore any failed cols and release resources storing
      // backup
      ublk.restore_if_required(*backup, elim_col);

      T beta = 0.0;
      T *upd = nullptr;
      int ldupd = 0;
      // Update failed cols
      ublk.update(
            isrc, jsrc, (*workspaces)[id], beta, upd, ldupd);

   }

   template<typename T, 
            typename Backup, 
            typename IntAlloc>
   void insert_restore_failed_block_app(
         starpu_data_handle_t l_jk_hdl,
         starpu_data_handle_t l_ij_hdl,
         starpu_data_handle_t col_hdl,
         int m, int n, int iblk, int jblk, int elim_col,
         ColumnData<T, IntAlloc> *cdata, 
         Backup *backup,
         std::vector<spral::ssids::cpu::Workspace> *workspaces, 
         int blksz, 
         int prio) {

      int ret;

      ret = starpu_task_insert(
            &cl_restore_failed_block_app,
            STARPU_R,  l_jk_hdl,
            STARPU_RW, l_ij_hdl,
            STARPU_R,  col_hdl,
            STARPU_VALUE, &m, sizeof(int),
            STARPU_VALUE, &n, sizeof(int),
            STARPU_VALUE, &iblk, sizeof(int),
            STARPU_VALUE, &jblk, sizeof(int),
            STARPU_VALUE, &elim_col, sizeof(int),
            STARPU_VALUE, &cdata, sizeof(ColumnData<T,IntAlloc>*),
            STARPU_VALUE, &backup, sizeof(Backup*),
            STARPU_VALUE, &workspaces, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
            STARPU_VALUE, &blksz, sizeof(int),
            STARPU_PRIORITY, prio,
            0);
         
   }

   ////////////////////////////////////////
   // updateN

   /* updateN_block_app StarPU codelet */
   extern struct starpu_codelet cl_updateN_block_app;

   /*  updateN_block_app StarPU task
         
    */
   template<typename T,
            int iblksz,
            typename Backup, 
            typename IntAlloc>
   void 
   updateN_block_app_cpu_func(void *buffers[], void *cl_arg) {

      typedef spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> BlockSpec;

      T *a_ik = (T *)STARPU_MATRIX_GET_PTR(buffers[0]); // Get diagonal block pointer
      unsigned ld_a_ik = STARPU_MATRIX_GET_LD(buffers[0]); // Get leading dimensions

      T *a_jk = (T *)STARPU_MATRIX_GET_PTR(buffers[1]); // Get diagonal block pointer
      unsigned ld_a_jk = STARPU_MATRIX_GET_LD(buffers[1]); // Get leading dimensions

      T *a_ij = (T *)STARPU_MATRIX_GET_PTR(buffers[2]); // Get diagonal block pointer
      unsigned ld_a_ij = STARPU_MATRIX_GET_LD(buffers[2]); // Get leading dimensions
      // Debug
      unsigned m_a_ij = STARPU_MATRIX_GET_NX(buffers[2]);
      unsigned n_a_ij = STARPU_MATRIX_GET_NY(buffers[2]);

         
      int id = starpu_worker_get_id();

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
      int blksz;

      starpu_codelet_unpack_args (
            cl_arg,
            &m, &n,
            &iblk, &jblk, &blk,
            &cdata, &backup,
            &beta, &upd, &ldupd,
            &work, &blksz);

      BlockSpec ublk(iblk, jblk, m, n, *cdata, a_ij, ld_a_ij, blksz);
      BlockSpec isrc(iblk, blk, m, n, *cdata, a_ik, ld_a_ik, blksz);
      BlockSpec jsrc(jblk, blk, m, n, *cdata, a_jk, ld_a_jk, blksz);

#if !defined(SPLDLT_USE_GPU)
      // If we're on the block col we've just eliminated, restore any
      // failed cols and release resources storing backup
      ublk.restore_if_required(*backup, blk);
#endif
      // Perform actual update
      ublk.update(
            isrc, jsrc, (*work)[id], beta, upd, ldupd);

   }

   template<typename T, 
            typename Backup, 
            typename IntAlloc>
   void insert_updateN_block_app(
         starpu_data_handle_t a_ik_hdl,
         starpu_data_handle_t a_jk_hdl,
         starpu_data_handle_t a_ij_hdl,
         starpu_data_handle_t d_hdl,
         starpu_data_handle_t col_hdl,
         int m, int n, int iblk, int jblk, int blk,
         ColumnData<T,IntAlloc> *cdata, Backup *backup,
         T beta, T* upd, int ldupd,
         std::vector<spral::ssids::cpu::Workspace> *work, 
         int blksz,
         int prio) {

      int ret;

      ret = starpu_task_insert(
            &cl_updateN_block_app,
            STARPU_R, a_ik_hdl,
            STARPU_R, a_jk_hdl,
            STARPU_RW | STARPU_COMMUTE, a_ij_hdl,
            STARPU_R, d_hdl,
            STARPU_SCRATCH, workspace_hdl,
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
            STARPU_PRIORITY, prio,
            0);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
         
   }

   /* updateT_block_app StarPU codelet */
   extern struct starpu_codelet cl_updateT_block_app;      

   /*  updateT_block_app StarPU task
         
    */
   template<typename T,
            int iblksz,
            typename Backup, 
            typename IntAlloc>
   void 
   updateT_block_app_cpu_func(void *buffers[], void *cl_arg) {

      typedef spldlt::ldlt_app_internal::Block<T, iblksz, IntAlloc> BlockSpec;
         
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
      std::vector<spral::ssids::cpu::Workspace> *workspaces; 
      int blksz;

      starpu_codelet_unpack_args (
            cl_arg,
            &m, &n,
            &isrc_row, &isrc_col,
            &iblk, &jblk, &blk,
            &cdata, &backup,
            &workspaces, &blksz /*&options*/);


      BlockSpec ublk(iblk, jblk, m, n, *cdata, a_ij, ld_a_ij, blksz);
      BlockSpec isrc(isrc_row, isrc_col, m, n, *cdata, a_ik, ld_a_ik, blksz);
      BlockSpec jsrc(blk, jblk, m, n, *cdata, a_jk, ld_a_jk, blksz);

      // If we're on the block row we've just eliminated, restore
      // any failed rows and release resources storing backup
      ublk.restore_if_required(*backup, blk);
      // Perform actual update
      ublk.update(isrc, jsrc, (*workspaces)[id]);
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
         std::vector<spral::ssids::cpu::Workspace> *work, 
         int blksz,
         int prio) {

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
            STARPU_PRIORITY, prio,
            0);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");         
   }

   /* adjust StarPU codelet */
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
         ColumnData<T,IntAlloc> *cdata,
         int prio) {

      int ret;

      ret = starpu_task_insert(
            &cl_adjust,
            STARPU_RW, col_hdl,
            STARPU_VALUE, &blk, sizeof(int),
            STARPU_VALUE, &next_elim, sizeof(int*),
            STARPU_VALUE, &cdata, sizeof(ColumnData<T,IntAlloc>*),
            STARPU_PRIORITY, prio,
            0);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
   }

   ////////////////////////////////////////////////////////////
   // Update contribution blocks

#if defined(SPLDLT_USE_PROFILING)

   template <typename T, typename IntAlloc, typename PoolAlloc>
   size_t update_contrib_block_app_size_base(struct starpu_task *task, unsigned nimpl) {
         
      size_t flops_task = 0;

      starpu_data_handle_t upd_hdl = STARPU_TASK_GET_HANDLE(task, 0);
      size_t updm = (size_t) starpu_matrix_get_nx(upd_hdl);
      size_t updn = (size_t) starpu_matrix_get_ny(upd_hdl);
         
      NumericFront<T, PoolAlloc> *node = nullptr;
      int k, i, j;
      std::vector<spral::ssids::cpu::Workspace> *workspaces;

      starpu_codelet_unpack_args(
            task->cl_arg, &node, &k, &i, &j, &workspaces);

      spldlt::ldlt_app_internal::ColumnData<T, IntAlloc> *cdata = node->cdata;
      int cnelim = (*cdata)[k].nelim;
         
      flops_task = 2*updm*updn*cnelim;

      task->flops = (double)flops_task;

      return flops_task;
   }

   template <typename T, typename IntAlloc, typename PoolAlloc>
   uint32_t update_contrib_block_app_footprint(struct starpu_task *task) {
         
      uint32_t footprint = 0;
      size_t flops;
         
      flops = update_contrib_block_app_size_base<T, IntAlloc, PoolAlloc>(task, 0);

      footprint = starpu_hash_crc32c_be_n(&flops, sizeof(flops), footprint);

      return footprint;
   }

   extern struct starpu_perfmodel update_contrib_block_app_perfmodel;

   // struct starpu_perfmodel update_contrib_block_app_perfmodel = {
   //    .type = STARPU_HISTORY_BASED,
   //    .size_base = update_contrib_block_app_size_base,
   //    .footprint = update_contrib_block_app_footprint
   // };
#endif

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

      starpu_codelet_unpack_args(
            cl_arg, &node, &k, &i, &j, &workspaces);

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
         starpu_data_handle_t d_hdl,
         starpu_data_handle_t col_hdl, // Symbolic handle on block-column k
         starpu_data_handle_t contrib_hdl, // Contribution blocks symbolic handle
         NumericFront<T, PoolAlloc> *node,
         int k, int i, int j,
         std::vector<spral::ssids::cpu::Workspace> *workspaces,
         int prio) {

      int ret;

      // #if defined(SPLDLT_USE_PROFILING)

      // Tile<T, PoolAlloc>& upd = cnode->get_contrib_block(i, j);
      // spldlt::ldlt_app_internal::Block<T, INNER_BLOCK_SIZE, IntAlloc>

      // #endif

      ret = starpu_insert_task(
            &cl_update_contrib_block_app,
            STARPU_RW | STARPU_COMMUTE, upd_hdl,
            STARPU_R, lik_hdl,
            STARPU_R, ljk_hdl,
            STARPU_R, d_hdl,
            STARPU_SCRATCH, workspace_hdl,
            STARPU_R, col_hdl,
            STARPU_R, contrib_hdl, // Contribution blocks symbolic handle
            STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
            STARPU_VALUE, &k, sizeof(int),
            STARPU_VALUE, &i, sizeof(int),
            STARPU_VALUE, &j, sizeof(int),
            STARPU_VALUE, &workspaces, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
            STARPU_PRIORITY, prio,
            0);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

   }

   ////////////////////////////////////////////////////////////
   // permute_failed StarPU task

   // CPU kernel
   template <typename T, typename IntAlloc, typename PoolAlloc>
   void permute_failed_cpu_func(void *buffers[], void *cl_arg) {
         
      NumericFront<T, PoolAlloc> *node = nullptr;
      PoolAlloc *alloc = nullptr;

      starpu_codelet_unpack_args(
            cl_arg, &node, &alloc);

      int n = node->get_ncol();

      if (node->nelim < n) {

         CopyBackup<T, PoolAlloc> &backup = *node->backup; 

         backup.release_all_memory(); 
         
         int m = node->get_nrow();
         int ldl = node->get_ldl();
         ColumnData<T, IntAlloc> &cdata = *node->cdata;
         bool const debug = false;
         int blksz = node->blksz();
            
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
      
   ////////////////////////////////////////////////////////////
   // form_contrib StarPU task

   // CPU kernel      
   template <typename T, typename PoolAlloc>
   void form_contrib_cpu_func(void *buffers[], void *cl_arg) {

      NumericFront<T, PoolAlloc> *node = nullptr;
      // spral::ssids::cpu::Workspace *work = nullptr;
      std::vector<spral::ssids::cpu::Workspace> *workspaces = nullptr;
      int nelim_from;
      int nelim_to;

      starpu_codelet_unpack_args(
            cl_arg, &node, &workspaces, &nelim_from, &nelim_to);

      int workerid = starpu_worker_get_id();
      spral::ssids::cpu::Workspace& work = (*workspaces)[workerid];

      form_contrib_notask(*node, work, nelim_from, nelim_to);
   }

   // SarPU codelet
   extern struct starpu_codelet cl_form_contrib;

   template <typename T, typename PoolAlloc>
   void insert_form_contrib(
         starpu_data_handle_t *hdls, int nhdl,
         NumericFront<T, PoolAlloc> *node,
         std::vector<spral::ssids::cpu::Workspace> *workspaces, 
         int nelim_from, int nelim_to) {

      int ret;
      struct starpu_data_descr *descrs = new starpu_data_descr[nhdl];
         
      int nh = 0;
      for (int i=0; i<nhdl; i++) {
         descrs[nh].handle = hdls[i]; descrs[nh].mode = STARPU_RW;
         nh++;
      }

      ret = starpu_insert_task(
            &cl_form_contrib,
            STARPU_DATA_MODE_ARRAY, descrs, nh,
            STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
            STARPU_VALUE, &workspaces, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
            STARPU_VALUE, &nelim_from, sizeof(int),
            STARPU_VALUE, &nelim_to, sizeof(int),
            0);
         
      delete[] descrs;
         
   }

   ////////////////////////////////////////////////////////////
   // zero_contrib_blocks

   // CPU kernel      
   template <typename T, typename PoolAlloc>
   void zero_contrib_blocks_cpu_func(void *buffers[], void *cl_arg) {

      printf("[zero_contrib_blocks_cpu_func]\n");
         
      NumericFront<T, PoolAlloc> *node = nullptr;

      starpu_codelet_unpack_args(
            cl_arg, &node);

      node->zero_contrib_blocks();
     
   }

   // SarPU codelet
   extern struct starpu_codelet cl_zero_contrib_blocks;

   template <typename T, typename PoolAlloc>
   void insert_zero_contrib_blocks(
         starpu_data_handle_t *hdls, int nhdl,
         NumericFront<T, PoolAlloc> *node) {

      int ret;
      struct starpu_data_descr *descrs = new starpu_data_descr[nhdl];
         
      int nh = 0;
      for (int i=0; i<nhdl; i++) {
         descrs[nh].handle = hdls[i]; descrs[nh].mode = STARPU_RW;
         nh++;
      }

      ret = starpu_insert_task(
            &cl_zero_contrib_blocks,
            STARPU_DATA_MODE_ARRAY, descrs, nh,
            STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
            0);
         
      delete[] descrs;

   }
      
   ////////////////////////////////////////////////////////////
   // nelim_sync

   // CPU kernel
   extern void nelim_sync_cpu_func(void *buffers[], void *cl_arg);

   // SarPU kernel
   extern struct starpu_codelet cl_nelim_sync;

   void insert_nelim_sync(starpu_data_handle_t node_hdl, int nodeidx);

   ////////////////////////////////////////////////////////////
   // assemble_contrib_sync

   extern void assemble_contrib_sync_cpu_func(void *buffers[], void *cl_arg);

   // StarPU kernel
   extern struct starpu_codelet cl_assemble_contrib_sync;

   void insert_assemble_contrib_sync(
         starpu_data_handle_t contrib_hdl,int nodeidx);

   ////////////////////////////////////////////////////////////
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

   ////////////////////////////////////////////////////////////
   // factor_failed_sync

   extern void factor_failed_sync_cpu_func(void *buffers[], void *cl_arg);

   // StarPU codelet
   extern struct starpu_codelet cl_factor_failed_sync;

   void insert_factor_failed_sync(int nodeidx);

   ////////////////////////////////////////////////////////////
   // assemble_delays_subtree
      
   // CPU kernel      
   template <typename T, typename PoolAlloc>      
   void assemble_delays_subtree_cpu_func(void *buffers[], void *cl_arg) {

      NumericFront<T, PoolAlloc> *node = nullptr;
      sylver::SymbolicFront *csnode = nullptr;
      void **child_contrib;
      int contrib_idx;
      int delay_col;

      starpu_codelet_unpack_args(
            cl_arg,
            &node, &csnode, &child_contrib, &contrib_idx, &delay_col);
         
      assemble_delays_subtree(
            *node, *csnode, child_contrib, contrib_idx, delay_col);
         
   }

   // StarPU codelet
   extern struct starpu_codelet cl_assemble_delays_subtree;

   template <typename T, typename PoolAlloc>
   void insert_assemble_delays_subtree(
         starpu_data_handle_t *hdls, int nhdl,
         starpu_data_handle_t root_hdl,
         NumericFront<T, PoolAlloc> *node,
         sylver::SymbolicFront *csnode,
         void **child_contrib, int contrib_idx,
         int delay_col) {

      assert(delay_col > 0);
      assert(nhdl > 0);

      int ret;
      int nh = 0;

      struct starpu_data_descr *descrs = new starpu_data_descr[nhdl+1];
         
      for (int i=0; i<nhdl; i++) {
         descrs[nh].handle = hdls[i]; descrs[nh].mode = STARPU_RW;
         nh++;
      }

      assert(nh > 0);

      // Handle on subtree
      descrs[nh].handle = root_hdl; descrs[nh].mode = STARPU_R;
      nh++;

      ret = starpu_task_insert(
            &cl_assemble_delays_subtree,
            STARPU_DATA_MODE_ARRAY, descrs, nh,
            STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
            STARPU_VALUE, &csnode, sizeof(sylver::SymbolicFront*),
            STARPU_VALUE, &child_contrib, sizeof(void**),
            STARPU_VALUE, &contrib_idx, sizeof(int),
            STARPU_VALUE, &delay_col, sizeof(int),
            0);

      delete[] descrs;
   }
         
   ////////////////////////////////////////////////////////////
   // assemble_delays
      
   // CPU kernel      
   template <typename T, typename PoolAlloc>      
   void assemble_delays_cpu_func(void *buffers[], void *cl_arg) {

      NumericFront<T, PoolAlloc> *node = nullptr;
      NumericFront<T, PoolAlloc> *cnode = nullptr;
      int delay_col;

      starpu_codelet_unpack_args(
            cl_arg, &cnode, &delay_col, &node);

      assemble_delays(*cnode, delay_col, *node);
   }

   // StarPU kernel
   extern struct starpu_codelet cl_assemble_delays;

   template <typename T, typename PoolAlloc>
   void insert_assemble_delays(
         starpu_data_handle_t *chdls, int nchdl,
         starpu_data_handle_t *hdls, int nhdl,
         NumericFront<T, PoolAlloc> *cnode,
         int delay_col,
         NumericFront<T, PoolAlloc> *node) {

      assert(nchdl > 0);
      assert(nhdl > 0);
      assert(delay_col > 0);

      int ret;

      struct starpu_data_descr *descrs = new starpu_data_descr[nchdl+nhdl];

      int nh = 0;
      // Add handles from cnode in R mode
      for (int i=0; i<nchdl; i++) {
         descrs[nh].handle = chdls[i]; descrs[nh].mode = STARPU_R;
         nh++;
      }

      // Add handles from node in RW mode
      for (int i=0; i<nhdl; i++) {
         descrs[nh].handle = hdls[i]; descrs[nh].mode = STARPU_RW;
         nh++;
      }

      ret = starpu_insert_task(
            &cl_assemble_delays,
            STARPU_VALUE, &cnode, sizeof(NumericFront<T, PoolAlloc>*),
            STARPU_VALUE, &delay_col, sizeof(int),
            STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
            STARPU_DATA_MODE_ARRAY, descrs, nh,               
            0);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

      delete[] descrs;
   }
      
}} /* namespaces spldlt::starpu  */
