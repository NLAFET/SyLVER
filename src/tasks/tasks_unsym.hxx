#pragma once

// SyVLER
#include "Block.hxx"
#include "kernels/factor_unsym.hxx"

// SSIDS
#include "ssids/cpu/Workspace.hxx"

namespace spldlt {

   ////////////////////////////////////////////////////////////
   // APTP factorization tasks

   /// @brief Factor diagonal block dblk for APTP
   /// @param dblk Block to be factorized
   /// @param rperm Global row permutation
   /// @param cperm Global columns permutation
   /// @param cdata Column data
   template <typename T, typename IntAlloc>
   void factor_block_unsym_app_task(
         BlockUnsym<T>& dblk, int *rperm, int *cperm, 
         ColumnData<T, IntAlloc>& cdata) {

      int blk = dblk.get_row(); // Block row index
      dblk.backup(); // Backup dblk
      
      int nelim_block = dblk.factor(rperm, cperm);
      // int nelim = dblk.factor_lu_pp(rperm);
      // Init threshold check (non locking => task dependencies)
      cdata[blk].init_passed(nelim_block); // Init npass
      cdata[blk].nelim = nelim_block; // Init nelim
      
      printf("[factor_block_unsym_app_task] blk = %d, nelim_block = %d, npass = %d\n",
             blk, nelim_block, cdata[blk].get_npass());

   }

   /// @param u Threshold parameter
   template <typename T, typename IntAlloc>
   void appyU_block_app_task(
         BlockUnsym<T>& dblk, T u, BlockUnsym<T>& lblk, 
         ColumnData<T, IntAlloc>& cdata) {
      
      int elim_col = dblk.get_col();

      // int *rperm = dblk.get_lrperm();
      // lblk.backup_perm(rperm);
      
      lblk.backup(); // Note: assume there is no column pivoting

      // printf("[appyU_block_app_task] elim_col = %d, npass = %d\n",
      //        elim_col, cdata[elim_col].get_npass());

      int blkpass = lblk.applyU_app(dblk, u, cdata);
      // Update column's passed pivot count
      cdata[elim_col].update_passed(blkpass);      

      printf("[appyU_block_app_task] elim_col = %d, blkpass = %d, npass = %d\n",
             elim_col, blkpass, cdata[elim_col].get_npass());

   }

   template <typename T, typename IntAlloc>
   void applyL_block_app_task(
         BlockUnsym<T>& dblk, BlockUnsym<T>& ublk,
         ColumnData<T, IntAlloc>& cdata, 
         std::vector<spral::ssids::cpu::Workspace>& workspaces) {

      printf("[appyL_block_app_task]\n");

      spral::ssids::cpu::Workspace& workspace = workspaces[0]; 
      
      ublk.apply_rperm(dblk, workspace);
      
      ublk.applyL_app(dblk, cdata);
   }
   
   template <typename T, typename IntAlloc>
   void update_block_unsym_app_task(
         BlockUnsym<T>& lblk, BlockUnsym<T>& ublk, BlockUnsym<T>& blk,
         ColumnData<T, IntAlloc>& cdata) {

      // printf("[update_block_unsym_app_task]\n");
      
      blk.update_app(lblk, ublk, cdata);
   }

   template <typename T, typename IntAlloc>
   void restore_block_unsym_app_task(
         int elim_col, BlockUnsym<T>& blk, ColumnData<T, IntAlloc>& cdata) {
      
      blk.restore_failed(elim_col, cdata);
   }

   template <typename T, typename PoolAlloc, typename IntAlloc>
   void update_cb_block_unsym_app_task(
         BlockUnsym<T>& lblk, BlockUnsym<T>& ublk, Tile<T, PoolAlloc>& blk, 
         ColumnData<T, IntAlloc>& cdata) {

      int m = blk.m;
      int n = blk.m;

      int elim_col = lblk.get_row();
      int nelim = cdata[elim_col].nelim;

      int ml = lblk.m;
      int nl = lblk.n;
      T *l = &lblk.a[m-ml];
      int ldl = lblk.lda;
      
      int nu = ublk.n;
      T *u = (nu==n) ? lblk.a : lblk.b;
      int ldu = (nu==n) ? lblk.lda : lblk.ldb;
      
      update_block_lu(
            m, n, blk.a, blk.lda,
            nelim,
            l, ldl,
            u, ldu);
   }

   /// @brief Ajust the number of eliminated columns whitin
   /// block-colum elim_col
   /// @param nelim Golbal number of eliminated columns
   template <typename T, typename IntAlloc>
   void adjust_unsym_app_task(
         int elim_col,
         ColumnData<T, IntAlloc>& cdata,
         int& nelim) {

      printf("[adjust_unsym_app_task] elim_col = %d, nelim = %d, npass = %d\n",
             elim_col, cdata[elim_col].nelim, cdata[elim_col].get_npass());
      
      cdata[elim_col].nelim = cdata[elim_col].get_npass();

      nelim += cdata[elim_col].nelim;
   }

   ////////////////////////////////////////////////////////////

   /// @brief Perfom LU factorization on block dblk using partial
   /// pivoting
   template <typename T>
   void factor_block_lu_pp_task(BlockUnsym<T>& dblk, int *perm) {

      // dblk.alloc_lrperm();
      dblk.alloc_init_lrperm();
      int *lrperm = dblk.get_lrperm(); // Local row permutation
      // Number of fully-summed rows/columns in dblk
      int nfs = dblk.get_nfs();
      
      // Note: lrperm is 0-indexed in factor_block_lu_pp 
      factor_block_lu_pp(
            dblk.m, nfs, lrperm, dblk.a, dblk.lda, dblk.b, dblk.ldb);

      // printf("nfs = %d\n", nfs);
      // printf("lrperm\n");          
      // for (int i=0; i < nfs; ++i) printf(" %d ", lrperm[i]);
      // printf("\n");            

      // Update perm using local permutation lrperm
      int *temp = new int[nfs];
      for (int i = 0; i < nfs; ++i)
         temp[i] = perm[lrperm[i]];
      for (int i = 0; i < nfs; ++i)
         perm[i] = temp[i]; 

      delete[] temp;
   }

   /// @brief Apply row permutation perm on block rblk
   template <typename T>
   void apply_rperm_block_task(
         BlockUnsym<T>& dblk, BlockUnsym<T>& rblk,
         std::vector<spral::ssids::cpu::Workspace>& workspaces) {

      // Get local row permutation from dblk
      int *lrperm = dblk.get_lrperm();
      
      // Block dimensions
      int m = rblk.m;
      int n = rblk.n;
      
      spral::ssids::cpu::Workspace& workspace = workspaces[0]; 
      int ldw = spral::ssids::cpu::align_lda<T>(m);
      T* work = workspace.get_ptr<T>(ldw*n);
      apply_rperm_block(m, n, lrperm, rblk.a, rblk.lda, work, ldw);
   }

   /// @brief Compute U factor in lblk resulting from the
   /// factorization of block dblk
   template <typename T>
   void applyL_block_task(
         BlockUnsym<T>& dblk, BlockUnsym<T>& ublk,
         std::vector<spral::ssids::cpu::Workspace>& workspaces) {

      // Get local row permutation from dblk
      int *lrperm = dblk.get_lrperm();
      
      // ublk block might be split between lcol and ucol
      
      // Block dimensions
      int m = ublk.m;
      int n = ublk.n;
      int na = ublk.get_na();
      spral::ssids::cpu::Workspace& workspace = workspaces[0]; 
      int ldw = spral::ssids::cpu::align_lda<T>(m);
      T* work = workspace.get_ptr<T>(ldw*n);
      int nb = ublk.get_nb();

      applyL_block(
         m, na, dblk.a, dblk.lda, lrperm, ublk.a, ublk.lda, work, ldw);

      if (nb > 0) {
         // Apply L in parts stored un ucol
         applyL_block(
               m, nb, dblk.a, dblk.lda, lrperm, ublk.b, ublk.ldb, work, ldw);
      }

   }

   /// @brief Compute L factor in lblk resulting from the
   /// factorization of block dblk
   template <typename T>
   void applyU_block_task(BlockUnsym<T>& dblk, BlockUnsym<T>& lblk) {

      int m = lblk.m;
      int n = lblk.n;
      
      applyU_block(
            m, n, dblk.a, dblk.lda, lblk.a, lblk.lda);
   }

   template <typename T>
   void update_block_lu_task(BlockUnsym<T>& lblk, BlockUnsym<T>& ublk, BlockUnsym<T>& blk) {

      int ma = blk.get_ma();
      int na = blk.get_na();
      int k = lblk.n;
      
      update_block_lu(
            ma, na, blk.a, blk.lda,
            k,
            lblk.a, lblk.lda,
            ublk.a, ublk.lda);

      int nb = blk.get_nb();   

      if (nb > 0) {

         int mb = blk.get_mb();   

         update_block_lu(
            mb, nb, blk.b, blk.ldb,
            k,
            lblk.a, lblk.lda,
            ublk.b, ublk.ldb);
         
      }
   }

   template <typename T, typename PoolAlloc>
   void update_cb_block_lu_task(
         BlockUnsym<T>& lblk, BlockUnsym<T>& ublk, Tile<T, PoolAlloc>& blk) {

      int m = blk.m;
      int n = blk.m;

      int ml = lblk.m;
      int nl = lblk.n;
      T *l = &lblk.a[m-ml];
      int ldl = lblk.lda;
      
      int nu = ublk.n;
      T *u = (nu==n) ? lblk.a : lblk.b;
      int ldu = (nu==n) ? lblk.lda : lblk.ldb;
      
      update_block_lu(
            m, n, blk.a, blk.lda,
            nl,
            l, ldl,
            u, ldu);
      
   }
}
