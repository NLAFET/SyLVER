/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "NumericFront.hxx"
#include "kernels/ldlt_app.hxx"
#include "sylver/kernels/ColumnData.hxx"
#include "sylver/Tile.hxx"

// STD
#include <assert.h>
#include <chrono>
// SSIDS
#include "ssids/cpu/Workspace.hxx"
#include "ssids/cpu/kernels/calc_ld.hxx"

namespace sylver {
namespace spldlt {

   /// @brief Performs update of contribution block
   template <typename T>
   void update_contrib_block(
         int m, int n,
         T *upd, int ldupd,
         int k,
         const T * const lik, int ld_lik, 
         const T * const ljk, int ld_ljk,
         bool zero_upd,
         const T * const dk, // Diagonal
         T *ld, int ldld // Workspace
         ) {

      T rbeta = zero_upd ? 0.0 : 1.0;
      
      // Compute L_ik * D_k in workspace W
      spral::ssids::cpu::calcLD<spral::ssids::cpu::OP_N>(
            m, k, lik, ld_lik, dk, ld, ldld);
      
      // Compute U = U - W L^{T}
      host_gemm(
            OP_N, OP_T, m, n, k,
            // -1.0, ljk, ld_ljk, ld, ldld,
            -1.0, ld, ldld, ljk, ld_ljk,
            rbeta, upd, ldupd
            );

   }

   ////////////////////////////////////////////////////////////
   // APTP factorization kernels
   
   /// @brief Performs factorization of a block on the diagonal in the
   /// LDLT factorization with APTP pivoting
   template<
      typename BlockSpec, typename T, typename IntAlloc, typename Backup,
      typename Allocator>
   void factor_block_app(
         BlockSpec& dblk, int& next_elim,
         int* perm, T* d,
         sylver::ColumnData<T,IntAlloc>& cdata,
         Backup& backup,
         sylver::options_t& options,
         spral::ssids::cpu::Workspace& work,
         Allocator const& alloc) {

      int blk = dblk.get_row();

      dblk.backup(backup);
            
      // Perform actual factorization
      int nelim = dblk.template factor<Allocator>(
            next_elim, perm, d, options, work, alloc);
            
      // Init threshold check (non locking => task dependencies)
      cdata[blk].init_passed(nelim);
   }

   /// @brief Applies pivots on sub-diagonal block in the LDLT
   /// factorization with APTP pivoting
   template<
      typename BlockSpec, typename T, typename IntAlloc, typename Backup>
   void applyN_block_app(
         BlockSpec& dblk, BlockSpec& rblk,
         sylver::ColumnData<T,IntAlloc>& cdata,
         Backup& backup,
         sylver::options_t& options) {

      int blk = dblk.get_col();

      // Apply column permutation from factorization of dblk and in
      // the process, store a (permuted) copy for recovery in case of
      // a failed column
      rblk.apply_cperm_and_backup(backup);
      // Perform elimination and determine number of columns in block
      // passing a posteori threshold pivot test
      int blkpass = rblk.apply_pivot_app(dblk, options.u, options.small);
      // Update column's passed pivot count
      cdata[blk].update_passed(blkpass);
   }

   
   /// @brief Applies pivots on left-diagonal block in the LDLT
   /// factorization with APTP pivoting
   template<
      typename BlockSpec, typename T, typename IntAlloc, typename Backup>
   void applyT_block_app(
         BlockSpec& dblk, BlockSpec& cblk,
         sylver::ColumnData<T,IntAlloc>& cdata,
         Backup& backup,
         sylver::options_t& options) {

      int blk = dblk.get_col();

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

   /// @brief Calculate the number of pivots passed in the LDLT
   /// factorization with APTP pivoting
   template<typename T, typename IntAlloc>
   void adjust_app(
         int blk,
         int& next_elim,
         sylver::ColumnData<T,IntAlloc>& cdata) {

      // Adjust column once all applys have finished and we know final
      // number of passed columns.
      cdata[blk].adjust(next_elim);

   }

   /// @brief Restore failed clomuns in the LDLT factorization with
   /// APP
   template<typename BlockSpec, typename T, typename IntAlloc, typename Backup>
   void restore_failed_block_app(
         int elim_col,
         BlockSpec& isrc,
         BlockSpec& jsrc,
         BlockSpec& ublk,
         sylver::ColumnData<T,IntAlloc>& cdata,
         Backup& backup, spral::ssids::cpu::Workspace& work) {

      ublk.restore_if_required(backup, elim_col);

      T beta = 0.0;
      T *upd = nullptr;
      int ldupd = 0;
      // Update failed cols
      ublk.update(
            isrc, jsrc, work, beta, upd, ldupd);

   }

   /// @brief Update uneliminated block in the left-diagonal in the
   /// LDLT factorization with APP
   template<typename BlockSpec, typename Backup>
   void updateT_block_app(
         BlockSpec& isrc, BlockSpec& jsrc, BlockSpec& ublk,
         Backup& backup,
         spral::ssids::cpu::Workspace& work) {

      int blk = jsrc.get_row();
      
      // If we're on the block row we've just eliminated, restore
      // any failed rows and release resources storing backup
      ublk.restore_if_required(backup, blk);
      // Perform actual update
      ublk.update(isrc, jsrc, work);
   }
   
   /// @brief Update block in the trailing submatrix in the LDLT
   /// factorization with APP
   template<typename BlockSpec, typename T, typename IntAlloc, typename Backup>
   void updateN_block_app(
         BlockSpec& isrc, BlockSpec& jsrc, BlockSpec& ublk,
         sylver::ColumnData<T,IntAlloc>& cdata,
         Backup& backup,
         T const beta, T* upd, int const ldupd,
         spral::ssids::cpu::Workspace& work) {

      // Perform actual update
      ublk.update(isrc, jsrc, work, beta, upd, ldupd);

   }

   
   /// @brief Performs update of contribution block in the LDLT with
   /// APP factorization
   template <typename T, typename IntAlloc, typename PoolAlloc>
   void update_contrib_block_app(
         NumericFront<T, PoolAlloc>& node,
         int k, int i, int j,
         const T *lik, int ld_lik,
         const T *ljk, int ld_ljk,
         int updm, int updn, T *upd, int ldupd, 
         spral::ssids::cpu::Workspace& work
         ) {

      int blksz = node.blksz();

      int nrow = node.nrow();
      int ncol = node.ncol();
      int ldl = align_lda<T>(nrow);

      T *lcol = node.lcol;
      T *d = &lcol[ncol*ldl];

      int ljk_first_row = std::max(0, ncol-j*blksz);
      int lik_first_row = std::max(0, ncol-i*blksz);
      // printf("[udpate_contrib_block_app_cpu_func] lik_first_row = %d, ljk_first_row = %d\n", lik_first_row, ljk_first_row);

      sylver::ColumnData<T, IntAlloc> *cdata = node.cdata;
      int cnelim = (*cdata)[k].nelim;
      bool first_elim = (*cdata)[k].first_elim;
      T *dk = (*cdata)[k].d;
         
      // printf("[udpate_contrib_block_app_cpu_func] k = %d, i = %d, j = %d, cnelim = %d, first_elim = %d\n", k, i, j, cnelim, first_elim);

      if (cnelim <= 0) return; // No factors to update in current block-column

      int ldld = spral::ssids::cpu::align_lda<T>(blksz);
      // T *ld = new T[blksz*ldld];
      T *ld = work.get_ptr<T>(blksz*ldld);

      update_contrib_block(
            updm, updn, upd, ldupd,
            cnelim,
            &lik[lik_first_row], ld_lik,
            &ljk[ljk_first_row], ld_ljk,
            first_elim,
            dk,
            ld, ldld);
   
   }

   /// @brief Form cotribution blocks using the the columns nelim_from
   /// to nelim_to in the factors
   template <typename T, typename PoolAlloc>
   void form_contrib_notask(
         NumericFront<T, PoolAlloc>& node,
         spral::ssids::cpu::Workspace& work,
         int nelim_from, // First column in factors
         int nelim_to // Last column in factors
         ) {

      assert(nelim_to >= nelim_from);
      assert(nelim_from >= 0);
      // node.zero_contrib_blocks();

      int m = node.nrow();
      int n = node.ncol();
      
      assert(nelim_to < n);
      
      size_t ldl = node.ldl();
      T *lcol = node.lcol;
      T *d = &lcol[n*ldl];
      int blksz = node.blksz();

      int fc = nelim_from/blksz; // First block-column
      int lc = nelim_to/blksz; // Last block-column
      int nr = node.nr();
      int rsa = n/blksz;
      // int ncontrib = nr-rsa;

      // printf("[form_contrib] fc = %d, lc = %d\n", fc, lc);

      // for (int k = fc; k <= lc; ++k) {

      // int first_col = std::max(k*blksz, nelim_from); // first column in current block-column of L
      // int last_col = std::min(((k+1)*blksz)-1, nelim_to); // last column in current block-column of L

      int first_col = nelim_from; 
      int last_col = nelim_to;

      //int nelim_col = 0;
      int nelim_col = last_col-first_col+1;
      // if (k==fc) nelim_col = nelim_col+1; // debug
      T *dk = &d[2*first_col];
      // printf("[form_contrib] first_col = %d, last_col = %d, nelim_from = %d, nelim_to = %d, nelim_col = %d\n",
      // first_col, last_col, nelim_from, nelim_to, nelim_col);
      // printf("[form_contrib] k = %d, first_col = %d, last_col = %d, nelim_from = %d, nelim_to = %d, nelim_col = %d\n",
      //        k, first_col, last_col, nelim_from, nelim_to, nelim_col);
         
      // printf("k = %d, first_col = %d, last_col = %d\n", k, first_col, last_col);
      // return;

      for (int j = rsa; j < nr; ++j) {

         int ljk_first_row = std::max(j*blksz, n);
         T *ljk = &lcol[first_col*ldl+ljk_first_row];
         //T *ljk = &lcol[k*blksz*ldl+j*blksz];

         for (int i = j; i < nr; ++i) {
                           
            int lik_first_row = std::max(i*blksz, n);
            T *lik = &lcol[first_col*ldl+lik_first_row];

            // Tile<T, PoolAlloc>& upd = node.contrib_blocks[(j-rsa)*ncontrib+(i-rsa)];
            sylver::Tile<T, PoolAlloc>& upd = node.contrib_block(i, j);
                           
            int ldld = spral::ssids::cpu::align_lda<T>(blksz);
            T *ld = work.get_ptr<T>(blksz*ldld);
            // T *ld = new T[blksz*ldld];

            // printf("[form_contrib] updm = %d, updn = %d\n", upd.m, upd.n);
            // printf("[form_contrib] nelim_from = %d\n", nelim_from);

            update_contrib_block(
                  upd.m, upd.n, upd.a, upd.lda,
                  nelim_col, lik, ldl, ljk, ldl,
                  (nelim_from==0), dk, ld, ldld);
               
            // spral::ssids::cpu::calcLD<OP_N>(
            //       upd.m, (k==fc) ? nelim_col+1 : nelim_col, lik, ldl, dk, ld, ldld);
      
            // // Compute U = U - W L^{T}
            // host_gemm(
            //       OP_N, OP_T, upd.m, upd.n, (k==fc) ? nelim_col+1 : nelim_col,
            //       // -1.0, ljk, ld_ljk, ld, ldld,
            //       -1.0, ld, ldld, ljk, ldl,
            //       1.0, upd.a, upd.lda
            //       );

         }
      }
      // }
   }

}} // End of namespace sylver::spldlt
