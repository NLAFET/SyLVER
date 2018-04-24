#pragma once

// SpLDLT
#include "SymbolicFront.hxx"
#include "kernels/assemble.hxx"
#include "kernels/factor.hxx"
#if defined(SPLDLT_USE_STARPU)
#include "StarPU/kernels.hxx"
#include "StarPU/kernels_indef.hxx"
#endif

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"

#if defined(SPLDLT_USE_STARPU)
// StarPU
#include <starpu.h>
#endif

namespace spldlt {

   const int ASSEMBLE_PRIO = 4;   

   ////////////////////////////////////////////////////////////////////////////////
   /// @brief Launches a task for activating a node.
   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void activate_front_task(
         bool posdef,
         SymbolicFront& snode,
         NumericFront<T, PoolAlloc>& node,
         void** child_contrib,
         int blksz,
         FactorAlloc& factor_alloc) {

      // printf("[activate_front_task]\n");

#if defined(SPLDLT_USE_STARPU)

      int nchild = 0;
      for (auto* child=node.first_child; child!=NULL; child=child->next_child)
         nchild++;

      starpu_data_handle_t *cnode_hdls = new starpu_data_handle_t[nchild];
      
      int i = 0;
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {
         cnode_hdls[i] = child->symb.hdl;
         ++i;
      }
      
      spldlt::starpu::insert_activate_node(
            snode.hdl, cnode_hdls, nchild,
            posdef, &snode, &node, child_contrib, blksz, &factor_alloc); 

      delete[] cnode_hdls;

#else
      activate_front(
            posdef, snode, node, child_contrib, blksz, factor_alloc);
#endif
      
   }

   ////////////////////////////////////////////////////////////////////////////////
   // Initialize node.
   template <typename T, typename PoolAlloc>
   void init_node_task(
         SymbolicFront &sfront, 
         NumericFront<T, PoolAlloc> &front,
         T *aval, int prio) {

#if defined(SPLDLT_USE_STARPU)
      
      spldlt::starpu::insert_init_node(
            &sfront, &front,
            sfront.hdl,
            aval, prio);
#else

      init_node(sfront, front, aval);

#endif

   }

   ////////////////////////////////////////////////////////////////////////////////
   // Terminate node
   template <typename T, typename PoolAlloc>
   void fini_node_task(
         NumericFront<T, PoolAlloc> &node, 
         int blksz, 
         int prio) {

#if defined(SPLDLT_USE_STARPU)
      spldlt::starpu::insert_fini_node(node.get_hdl(), &node, prio);
#else
      fini_node(node);
#endif

   }

   ////////////////////////////////////////////////////////////////////////////////
   /// @brief Lauches a task that activate and init a front.
   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void activate_init_front_task(
         bool posdef,
         SymbolicFront& snode,
         NumericFront<T, PoolAlloc>& node,
         void** child_contrib,
         int blksz,
         FactorAlloc& factor_alloc,
         PoolAlloc& pool_alloc,
         T *aval
         ){

#if defined(SPLDLT_USE_STARPU)

      int nchild = 0;
      for (auto* child=node.first_child; child!=NULL; child=child->next_child)
         nchild++;

      starpu_data_handle_t *cnode_hdls = new starpu_data_handle_t[nchild];
      
      int i = 0;
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {
         cnode_hdls[i] = child->symb.hdl;
         ++i;
      }
      // printf("[activate_init_front_task] node = %d, nchild = %d\n", snode.idx+1, nchild);
      spldlt::starpu::insert_activate_init_node(
            snode.hdl, cnode_hdls, nchild,
            posdef, &snode, &node, child_contrib, blksz, &factor_alloc, 
            &pool_alloc, aval);
      
      delete[] cnode_hdls;
      
#else
      // Allocate data structures
      activate_front(
            posdef, snode, node, child_contrib, blksz, factor_alloc); 

      // Add coefficients from original matrix
      init_node(snode, node, aval);
#endif

   }

   ////////////////////////////////////////////////////////////////////////////////
   // Factorize block on the diagonal

   // TODO create and use BLK structure
   template <typename T, typename PoolAlloc>
   void factorize_diag_block_task(
         SymbolicFront const& snode,
         NumericFront<T, PoolAlloc> &node,
         int kk, // block  column (and row) index
         int blksz, int prio) {

      int m = snode.nrow + node.ndelay_in;
      int n = snode.ncol + node.ndelay_in;

      int nr = (m-1) / blksz + 1; // number of block rows
      int nc = (n-1) / blksz + 1; // number of block columns
      
      // printf("[factorize_diag_block_task] nr: %d, nc: %d, kk: %d\n", 
      //        nr, nc, kk);

      // printf("[factorize_diag_block_task] \n");

      int blkm = std::min(blksz, m - kk*blksz);
      int blkn = std::min(blksz, n - kk*blksz);

      int lda = align_lda<T>(m);
      T *a = node.lcol;
      T *contrib = node.contrib;
      int ldcontrib = m-n;

#if defined(SPLDLT_USE_STARPU)

      // starpu_data_handle_t node_hdl = NULL;
      // if (kk==0) node_hdl = snode.hdl;
      starpu_data_handle_t node_hdl = snode.hdl;

      if ((blkm > blkn) && (ldcontrib > 0)) {
         // factorize_diag_block(blkm, blkn,
         //                      &a[kk*blksz*(lda+1)], lda,
         //                      contrib, ldcontrib,
         //                      kk==0);

         spldlt::starpu::insert_factorize_block(
               kk, snode.handles[kk*nr + kk], node.contrib_blocks[0].hdl,
               snode.hdl, prio);
      }
      else {
         // printf("blkm: %d, blkn: %d, ldcontrib:%d\n", blkm, blkn, ldcontrib);
         spldlt::starpu::insert_factorize_block(
               snode.handles[kk*nr + kk], snode.hdl, prio);
      }

#else

      factorize_diag_block(blkm, blkn,
                           &a[kk*blksz*(lda+1)], lda,
                           contrib, ldcontrib,
                           kk==0);
#endif
   }

   ////////////////////////////////////////////////////////////////////////////////
   // Solve block (on subdaig) task

   template <typename T, typename PoolAlloc>
   void solve_block_task(
         SymbolicFront const& snode,
         NumericFront<T, PoolAlloc> &node,
         int k, // Column index
         int i, // Row index
         T *a, int lda, 
         T *a_ik, int lda_ik,         
         int blksz, int prio) {
      
      int m = snode.nrow + node.ndelay_in;
      int n = snode.ncol + node.ndelay_in;
      int ldcontrib = m-n;

      int nr = (m-1) / blksz + 1; // number of block rows
      int nc = (n-1) / blksz + 1; // number of block columns

      int rsa = n / blksz; // Row/Col index of first block in contrib 

      int blkn = std::min(blksz, n - k*blksz);
      int blkm = std::min(blksz, m - i*blksz);

#if defined(SPLDLT_USE_STARPU)
      
      if((blkn<blksz) && (ldcontrib>0)) {

         spldlt::starpu::insert_solve_block(
               k, blksz,
               snode.handles[k*nr + k], // diag block handle 
               snode.handles[k*nr + i], // subdiag block handle
               // snode.contrib_handles[i-rsa], // subdiag block handle
               node.contrib_blocks[i-rsa].hdl, // subdiag block handle
               snode.hdl,
               prio);
      }
      else {

         spldlt::starpu::insert_solve_block(
               snode.handles[k*nr + k], // diag block handle 
               snode.handles[k*nr + i], // subdiag block handle
               snode.hdl,
               prio);
      }
#else
      
      solve_block(blkm, blkn, a, lda, a_ik, lda_ik);

#endif   
   }

   ////////////////////////////////////////////////////////////////////////////////
   // Update block on subdaig task
   
   template <typename T, typename PoolAlloc>
   void update_block_task(
         SymbolicFront const& snode,
         NumericFront<T, PoolAlloc> &node,
         int k, /* block column index of A_ik and A_jk blocks */
         int i, /* block row index of A_ik and A_ij blocks  */
         int j, /* block row index of A_jk and block column index of
                   A_ij blocks */
         T *a_ij, int lda_ij,
         T *a_ik, int lda_ik,
         T *a_jk, int lda_jk,
         int blksz, int prio) {

      int m = snode.nrow + node.ndelay_in;
      int n = snode.ncol + node.ndelay_in;
      int ldcontrib = m-n;

      int blkm = std::min(blksz, m - i*blksz);
      int blkn = std::min(blksz, n - j*blksz);
      int blkk = std::min(blksz, n - k*blksz);

#if defined(SPLDLT_USE_STARPU)

      int nr = (m-1)/blksz + 1; // number of block rows
      int nc = (n-1)/blksz + 1; // number of block columns

      // TODO doen't work in supernodal mode
      if ((ldcontrib>0) && (blkn<blksz)) {

         int rsa = n/blksz; // Row/Col index of first block in contrib 

         spldlt::starpu::insert_update_block(
               k, blksz,
               snode.handles[j*nr + i], // A_ij block handle 
               snode.handles[k*nr + i], // A_ik block handle
               snode.handles[k*nr + j],  // A_jk block handle
               // snode.contrib_handles[i-rsa],
               node.contrib_blocks[i-rsa].hdl,
               snode.hdl,
               prio);
      }
      else {
         spldlt::starpu::insert_update_block(
               snode.handles[j*nr + i], // A_ij block handle 
               snode.handles[k*nr + i], // A_ik block handle
               snode.handles[k*nr + j],  // A_jk block handle
               snode.hdl,
               prio);
      }
#else

      update_block(blkm, blkn, a_ij, lda_ij,
                   blkk,
                   a_ik, lda_ik,
                   a_jk, lda_jk);

#endif
   }

//    template <typename T>
//    void update_diag_block_task(
//          SymbolicSNode const& snode,
//          int kk, /* block column index of A_ik and A_jk blocks */
//          int ii, /* block row index of A_ik and A_ij blocks  */
//          int jj, /* block row index of A_jk and block column index of
//                     A_ij blocks */
//          T *a_ij, int lda_ij,
//          T *a_ik, int lda_ik,
//          T *a_jk, int lda_jk,
//          int blksz, int prio) {

//       int m = snode.nrow;
//       int n = snode.ncol;

// #if defined(SPLDLT_USE_STARPU)

//       int nr = (m-1) / blksz + 1; // number of block rows
//       int nc = (n-1) / blksz + 1; // number of block columns

//       insert_update_diag_block(
//             snode.handles[jj*nr + ii], // A_ij block handle 
//             snode.handles[kk*nr + ii], // A_ik block handle
//             snode.handles[kk*nr + jj],  // A_jk block handle
//             prio);

// #else

//       int blkm = std::min(blksz, m - ii*blksz);
//       int blkn = std::min(blksz, n - jj*blksz);
//       int blkk = std::min(blksz, n - kk*blksz);

//       update_diag_block(blkm, blkn, a_ij, lda_ij,
//                         blkk,
//                         a_ik, lda_ik,
//                         a_jk, lda_jk);

// #endif
//    }


   ////////////////////////////////////////////////////////////////////////////////
   // Update contrib block task

   template <typename T, typename PoolAlloc>
   void update_contrib_task(
         SymbolicFront const& snode,
         NumericFront<T, PoolAlloc> &node,
         int k, int i, int j,
         int blksz, int prio) {

      int m = snode.nrow + node.ndelay_in;
      int n = snode.ncol + node.ndelay_in;

      int blkm = std::min(blksz, m - i*blksz);
      int blkn = std::min(blksz, m - j*blksz);
      int blkk = std::min(blksz, n - k*blksz);

      int lda = align_lda<T>(m);
      T *a = node.lcol;
      int ldcontrib = m-n;
      T *contrib = node.contrib;

      int nr = (m-1)/blksz + 1; // number of block rows
      int nc = (n-1)/blksz + 1; // number of block columns

#if defined(SPLDLT_USE_STARPU)

      int rsa = n/blksz;
      int ncontrib = nr-rsa;

      spldlt::starpu::insert_update_contrib(k,
                            // snode.contrib_handles[(i-rsa)+(j-rsa)*ncontrib],
                            node.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].hdl,
                            snode.handles[k*nr + i],
                            snode.handles[k*nr + j],
                            snode.hdl,
                            prio);

      // update_block(blkm, blkn,
      //              &contrib[((j*blksz-n)*ldcontrib) + (i*blksz)-n], ldcontrib,
      //              blkk,
      //              &a[(k*blksz*lda) + (i*blksz)], lda, 
      //              &a[(k*blksz*lda) + (j*blksz)], lda,
      //              k==0);

#else

      update_block(blkm, blkn,
                   &contrib[((j*blksz-n)*ldcontrib) + (i*blksz)-n], ldcontrib,
                   blkk,
                   &a[(k*blksz*lda) + (i*blksz)], lda, 
                   &a[(k*blksz*lda) + (j*blksz)], lda,
                   k==0);

#endif      
   }   
   
   ////////////////////////////////////////////////////////////////////////////////
   // Factor subtree task

   template <typename T>
   inline void factor_subtree_task(
         const void *akeep,
         void *fkeep,
         SymbolicFront const& root,
         const T *aval,
         int p, void **child_contrib,
         const struct spral::ssids::cpu::cpu_factor_options *options,
         std::vector<ThreadStats>& worker_stats) {

#if defined(SPLDLT_USE_STARPU)

      spldlt::starpu::insert_factor_subtree(
            root.hdl, akeep, fkeep, p, aval, child_contrib, options, &worker_stats);

#else
      ThreadStats& stats = worker_stats[0];
      spldlt_factor_subtree_c(
            akeep, fkeep, p, aval, child_contrib, options, &stats);
#endif
   }

   ////////////////////////////////////////////////////////////////////////////////
   // Get contrib task

//    // Kernel
//    extern "C" void spldlt_get_contrib_c(void *akeep, void *fkeep, int p, void **child_contrib);

//    void get_contrib_task(void *akeep, void *fkeep, SymbolicSNode const& root, int p, void **child_contrib) {

// #if defined(SPLDLT_USE_STARPU)

//       insert_get_contrib(root.hdl, akeep, fkeep, p, child_contrib);

// #else
//       spldlt_get_contrib_c(akeep, fkeep, p, child_contrib)
// #endif      
//    }


   ////////////////////////////////////////////////////////////////////////////////
   // Assemble subtree task
   template <typename T, typename PoolAlloc>   
   void assemble_subtree_task(
         SymbolicFront const& sfront, // Destination node (Symbolic)
         NumericFront<T,PoolAlloc>& front, // Destination node (Numeric) 
         SymbolicFront &csfront, // Root of the subtree
         void** child_contrib, 
         int contrib_idx, // Index of subtree to assemble
         int *cmap, // row/column mapping array 
         int blksz, int prio) {

#if defined(SPLDLT_USE_STARPU)

      int nrow = front.get_nrow();
      int ncol = front.get_ncol();
      int nr = front.get_nr(); // Number of block rows in destination node
      int nc = front.get_nc(); // Number of block columns in destination node
      int cc = -1; // Block column index in destination node
      int rr = -1; // Block row index in destination node

      starpu_data_handle_t *hdls = new starpu_data_handle_t[nr*nc];
      int nh = 0;

      int cn = csfront.nrow - csfront.ncol;
      
      for(int j = 0; j < cn; ++j) {

         int c = cmap[ j ]; // Destination column

         if (cc == (c/blksz)) continue;
         if (c < ncol) {

            cc = c / blksz;
            rr = -1;

            for (int i = j ; i < cn; ++i) {

               int r = cmap[ i ];
               if (rr==(r/blksz)) continue;
               rr = r/blksz;
               
               hdls[nh] = sfront.handles[cc*nr+rr];
               nh++;            
            }
         }         
      }

      // Insert assembly tasks if there are any contributions
      if (nh>0) {
         spldlt::starpu::insert_subtree_assemble(
               &front, &csfront, sfront.hdl, csfront.hdl, hdls, nh, 
               child_contrib, contrib_idx);
      }

      delete[] hdls;

#else      
      // Retreive contribution block from subtrees
      int cn, ldcontrib, ndelay, lddelay;
      double const *cval, *delay_val;
      int const *crlist, *delay_perm;
      spral_ssids_contrib_get_data(
            child_contrib[contrib_idx], &cn, &cval, &ldcontrib, &crlist,
            &ndelay, &delay_perm, &delay_val, &lddelay
            );

      printf("[subtree_assemble_task] contrib_idx: %d, cn: %d, ndelay: %d\n", 
             contrib_idx+1, cn, ndelay);
      // continue;
      if(!cval) return; // child was all delays, nothing more to do
      // int* cache = work[omp_get_thread_num()].get_ptr<int>(cn);
      for(int j = 0; j < cn; ++j) {
               
         int c = cmap[ j ]; // Destination column
                  
         T const* src = &cval[j*ldcontrib];

         if (c < sfront.ncol) {

            int ldd = front.get_ldl();
            T *dest = &front.lcol[c*ldd];

            for (int i = j ; i < cn; ++i) {
               // Assemble destination block
               dest[ cmap[ i ]] += src[i];
            }
         }
      }
#endif
   }
   
   ////////////////////////////////////////////////////////////////////////////////
   // Subtree assemble contrib task
   template <typename T, typename PoolAlloc>   
   void assemble_contrib_subtree_task(
         NumericFront<T,PoolAlloc>& node, // Destination node
         SymbolicFront& csnode, // Root of the subtree
         void** child_contrib, 
         int contrib_idx, // Index of subtree to assemble
         int *cmap, // row/column mapping array 
         int prio) {

      int blksz = node.blksz;

#if defined(SPLDLT_USE_STARPU)
// #if 0
      SymbolicFront const& snode = node.symb;

      int ncol = node.get_ncol();
      int nr = node.get_nr();
      int rsa = ncol / blksz; // Rows/Cols index of first block in contrib 
      int ncontrib = nr-rsa; // Number of block rows/cols in contrib

      int cc = -1; // Block column index in destination node
      int rr = -1; // Block row index in destination node

      starpu_data_handle_t *hdls = new starpu_data_handle_t[ncontrib*ncontrib];
      int nh = 0;

      int cn = csnode.nrow - csnode.ncol;

      for(int j = 0; j < cn; ++j) {

         int c = cmap[ j ]; // Destination column

         if (cc == (c/blksz)) continue;

         if (c >= snode.ncol) {

            cc = c / blksz; // Destination block column
            rr = -1;

            for (int i = j; i < cn; ++i) {

               int r = cmap[ i ]; // Destination row in parent front
               if (rr == (r/blksz)) continue;
               rr = r / blksz; // Destination block row

               hdls[nh] = node.contrib_blocks[(cc-rsa)*ncontrib+(rr-rsa)].hdl;
               nh++;
               
            }
         }
      }

      // printf("[assemble_contrib_subtree_task] nh = %d\n", nh);
      // Insert assembly tasks if there are any contributions
      if (nh > 0) {
         spldlt::starpu::insert_subtree_assemble_contrib(
               &node, &csnode, snode.hdl, csnode.hdl, hdls, nh, child_contrib,
               contrib_idx, blksz, prio);
      }
      
      delete[] hdls;
      
#else

      assemble_contrib_subtree(node, csnode, child_contrib, contrib_idx, blksz);

#endif
   }

   ////////////////////////////////////////////////////////////////////////////////
   // Assemble block task
   // ii: Row index in frontal matrix node
   // jj: Col index in frontal matrix node
   template <typename T, typename PoolAlloc>   
   void assemble_block_task(
         SymbolicFront const& snode, 
         NumericFront<T,PoolAlloc>& node, 
         SymbolicFront const& csnode, 
         NumericFront<T,PoolAlloc> const& cnode, 
         int ii, int jj, int *cmap, int blksz, int prio) {

#if defined(SPLDLT_USE_STARPU)

      int nrow = node.get_nrow();
      int ncol = node.get_ncol();
      int nr = node.get_nr(); // Number of block-rows in destination node
      int nc = node.get_nc(); // Number of block-columns in destination node

      starpu_data_handle_t *hdls = new starpu_data_handle_t[nr*nc];
      int nh = 0;
      int cm = csnode.nrow - csnode.ncol;
      // colum indexes
      int c_sa = (csnode.ncol > jj*blksz) ? 0 : (jj*blksz-csnode.ncol); // first col in block
      int c_en = std::min((jj+1)*blksz-csnode.ncol, cm); // last col in block
      // row indexes
      int r_en = std::min((ii+1)*blksz-csnode.ncol, cm); // last row in block

      int cc = -1; // Block column index in destination node
      int rr = -1; // Block row index in destination node

      // loop over column in block
      for (int j=c_sa; j<c_en; j++) {
         
         // Column index in parent node.
         // int c = map[ csnode.rlist[csnode.ncol+j] ];
         int c = cmap[ j ];

         if (cc == (c/blksz)) continue;

         if (c < ncol) {

            cc = c/blksz;
            rr = -1;

            int r_sa = (ii == jj) ? j : (ii*blksz-csnode.ncol); // first row in block

            for (int i=r_sa; i<r_en; i++) {

               // int r = map[ csnode.rlist[csnode.ncol+i] ];
               int r = cmap[ i ];
               if (rr == (r / blksz)) continue;
               rr = r/blksz;
               
               hdls[nh] = snode.handles[cc*nr+rr];
               nh++;
            }
         }
      }
      
      // Insert assembly tasks if there are contributions
      if (nh > 0) {
         
         int cnr = (csnode.nrow-1)/blksz+1;  
         int crsa = csnode.ncol/blksz;
         int cncontrib = cnr-crsa;
      
         spldlt::starpu::insert_assemble_block(&node, &cnode, ii, jj, cmap, 
                               // csnode.contrib_handles[(jj-crsa)*cncontrib+(ii-crsa)],
                               cnode.contrib_blocks[(jj-crsa)*cncontrib+(ii-crsa)].hdl,
                               hdls, nh, snode.hdl, csnode.hdl,
                               prio);
      }

      delete[] hdls;

      // assemble_block(node, cnode, ii, jj, cmap, blksz);
#else

      assemble_block(node, cnode, ii, jj, cmap);
#endif
   }

   ////////////////////////////////////////////////////////////////////////////////
   // Assemble contrib block task.   
   // ii: Row index in frontal matrix.
   // jj: Col index in frontal matrix.
   template <typename T, typename PoolAlloc>   
   void assemble_contrib_block_task(
         NumericFront<T,PoolAlloc>& node, 
         NumericFront<T,PoolAlloc>& cnode, 
         int ii, int jj, int *cmap, int prio) {

      int blksz = node.blksz;

#if defined(SPLDLT_USE_STARPU)

      SymbolicFront const& snode = node.symb;
      SymbolicFront const& csnode = cnode.symb;
   
      int nrow = node.get_nrow();
      int ncol = node.get_ncol();
      int nr = node.get_nr(); // Number of block-rows in destination node
      int nc = node.get_nc(); // Number of block-columns in destination node
      int rsa = ncol/blksz; // rows/cols index of first block in contrib 
      int ncontrib = nr-rsa; // number of block rows/cols in contrib

      // Array of block handles in parent front
      starpu_data_handle_t *hdls = new starpu_data_handle_t[ncontrib*ncontrib];
      int nh = 0;

      int cm = csnode.nrow - csnode.ncol;

      // colum indexes
      int c_sa = (csnode.ncol > jj*blksz) ? 0 : (jj*blksz-csnode.ncol); // first col in block
      int c_en = std::min((jj+1)*blksz-csnode.ncol, cm); // last col in block
      // row indexes
      int r_en = std::min((ii+1)*blksz-csnode.ncol, cm); // last row in block

      int cc = -1;
      int rr = -1;

      // loop over column in block
      for (int j = c_sa; j < c_en; j++) {

         // Column index in parent node
         int c = cmap[ j ];

         if (cc == (c/blksz)) continue;

         if (c >= ncol) {

            cc = c/blksz;
            rr = -1;

            int r_sa = (ii == jj) ? j : (ii*blksz-csnode.ncol); // first row in block

            for (int i = r_sa; i < r_en; i++) {

               int r = cmap[ i ];
               if (rr == (r/blksz)) continue;
               rr = r/blksz;
               
               // hdls[nh] = snode.contrib_handles[(cc-rsa)*ncontrib+(rr-rsa)];
               hdls[nh] = node.contrib_blocks[(cc-rsa)*ncontrib+(rr-rsa)].hdl;
               nh++;
            }
         }
      }

      // Insert assembly tasks if there are contribution
      if (nh>0) {
         
         int cnr = (csnode.nrow-1)/blksz+1;  
         int crsa = csnode.ncol/blksz;
         int cncontrib = cnr-crsa;

         spldlt::starpu::insert_assemble_contrib_block(&node, &cnode, ii, jj, cmap, blksz, 
                                       // csnode.contrib_handles[(jj-crsa)*cncontrib+(ii-crsa)], 
                                       cnode.contrib_blocks[(jj-crsa)*cncontrib+(ii-crsa)].hdl,
                                       hdls, nh, snode.hdl, csnode.hdl,
                                       prio);
      }
      delete[] hdls;

      // assemble_contrib_block(node, cnode, ii, jj, cmap, blksz);
#else

      assemble_contrib_block(node, cnode, ii, jj, cmap, blksz);
#endif
   }   

   ////////////////////////////////////////////////////////////////////////////////
   // factor_front_posdef
   //
   // Factorize a supernode using a multifrontal mode
   template <typename T, typename PoolAlloc>
   void factor_front_posdef(
         SymbolicFront const& snode,
         NumericFront<T, PoolAlloc> &node,
         struct cpu_factor_options const& options
         ) {

      /* Extract useful information about node */
      int m = node.get_nrow();
      int n = node.get_ncol();
      int lda = align_lda<T>(m);
      T *lcol = node.lcol;
      T *contrib = node.contrib;
      int ldcontrib = m-n;

      int blksz = options.cpu_block_size;
      int nr = (m-1) / blksz + 1; // number of block rows
      int nc = (n-1) / blksz + 1; // number of block columns
   
      // printf("[factorize_node_posdef_mf] contrib: %p\n", contrib);

      int FACTOR_PRIO = 3;
      int SOLVE_PRIO = 2;
      int UPDATE_PRIO = 1;

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

// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif

         /* Column Solve Tasks */
         for(int i = j+1; i < nr; ++i) {

            int blkm = std::min(blksz, m - i*blksz);

            // printf("[factorize_node_posdef_mf] contrib start: %d\n", (i*blksz)-n);
            // TODO fix STF version
            solve_block_task(
                  snode, node, j, i,
                  &lcol[j*blksz*(lda+1)], lda,
                  &lcol[(j*blksz*lda) + (i*blksz)], lda,
                  blksz, SOLVE_PRIO);

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

   ////////////////////////////////////////////////////////////////////////////////
   // assemble front
   template <typename T, typename PoolAlloc>
   void assemble_task(
         int n,
         SymbolicFront& snode,
         NumericFront<T, PoolAlloc>& node,
         void** child_contrib,
         PoolAlloc& pool_alloc,
         int blksz
         ) {

#if defined(SPLDLT_USE_STARPU)

      int nchild = 0;
      for (auto* child=node.first_child; child!=NULL; child=child->next_child)
         nchild++;

      starpu_data_handle_t *cnode_hdls = new starpu_data_handle_t[nchild];
      
      int i = 0;
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {
         cnode_hdls[i] = child->symb.hdl;
         ++i;
      }
      // printf("[assemble_task] node = %d, nchild = %d\n", snode.idx+1, nchild);
      spldlt::starpu::insert_assemble(
            snode.hdl, cnode_hdls, nchild,
            n, &node, child_contrib, &pool_alloc, blksz);

      delete[] cnode_hdls;
      
#else
      assemble(n, node, child_contrib, pool_alloc, blksz);
#endif
      
   }

   // // TODO: error managment
   // template <typename T, typename PoolAlloc>
   // void factorize_node_posdef(
   //       SymbolicSNode const& snode,
   //       spldlt::NumericNode<T, PoolAlloc> &node,
   //       struct cpu_factor_options const& options
   //       ) {

   //    /* Extract useful information about node */
   //    int m = snode.nrow;
   //    int n = snode.ncol;
   //    int lda = align_lda<T>(m);
   //    T *a = node.lcol;
   //    // T *contrib = nodes_[ni].contrib;

   //    int blksz = options.cpu_block_size;
   //    int nr = (m-1) / blksz + 1; // number of block rows
   //    int nc = (n-1) / blksz + 1; // number of block columns

   //    /* Task priorities:
   //       init: 4
   //       facto: 3
   //       solve: 2
   //       udpate: 1
   //       udpate_between: 0
   //    */
   
   //    for(int j = 0; j < nc; ++j) {
         
   //       /* Diagonal Block Factorization Task */
   //       int blkm = std::min(blksz, m - j*blksz);
   //       int blkn = std::min(blksz, n - j*blksz);
         
   //       factorize_diag_block_task(snode, node, j, blksz, 3);
         
   //       // factorize_diag_block(blkm, blkn, &a[j*blksz*(lda+1)], lda);


   //       /* Column Solve Tasks */
   //       for(int i = j+1; i < nr; ++i) {

   //          int blkm = std::min(blksz, m - j*blksz);
            
   //          solve_block_task(
   //                snode,
   //                j, i,
   //                &a[j*blksz*(lda+1)], lda,
   //                &a[(j*blksz*lda) + (i*blksz)], lda,
   //                blksz, 2);

   //          // solve_block(
   //          //       blkm, blkn, 
   //          //       &a[j*blksz*(lda+1)], lda, 
   //          //       &a[(j*blksz*lda) + (i*blksz)], lda);
   //       }

   //       /* Schur Update Tasks: mostly internal */
   //       for(int k = j+1; k < nc; ++k) {

   //          update_diag_block_task(
   //                snode,
   //                j, /* block column index of A_ik and A_jk blocks */
   //                k, /* block row index of A_ik and A_ij blocks  */
   //                k, /* block row index of A_jk and block column index of
   //                      A_ij blocks */
   //                &a[ (k*blksz*lda) + (k*blksz)], lda,
   //                &a[(j*blksz*lda) + (k*blksz)], lda,
   //                &a[(j*blksz*lda) + (k*blksz)], lda,
   //                blksz, 1);

   //          // int blkm = std::min(blksz, m - k*blksz);
   //          // int blkn = std::min(blksz, n - k*blksz);
   //          // int blkk = std::min(blksz, n - j*blksz);

   //          // update_diag_block(blkm, blkn, 
   //          //                   &a[(k*blksz*lda) + (k*blksz)], lda,
   //          //                   blkk,
   //          //                   &a[(j*blksz*lda) + (k*blksz)], lda,
   //          //                   &a[(j*blksz*lda) + (k*blksz)], lda);


   //          for(int i = k+1;  i < nr; ++i) {
               
   //             update_block_task(
   //                   snode,
   //                   j, /* block column index of A_ik and A_jk blocks */
   //                   i, /* block row index of A_ik and A_ij blocks  */
   //                   k, /* block row index of A_jk and block column index of
   //                         A_ij blocks */
   //                   &a[ (k*blksz*lda) + (i*blksz)], lda,
   //                   &a[(j*blksz*lda) + (i*blksz)], lda,
   //                   &a[(j*blksz*lda) + (k*blksz)], lda,
   //                   blksz, 1);
   //          }
   //       }         
   //    }
   // }

   // Serial version

   // // TODO: error managment
   // template <typename T, typename PoolAlloc>
   // void factorize_node_posdef_notask(
   //       SymbolicSNode const& snode,
   //       spldlt::NumericNode<T, PoolAlloc> &node,
   //       struct cpu_factor_options const& options
   //       ) {

   //    /* Extract useful information about node */
   //    int m = snode.nrow;
   //    int n = snode.ncol;
   //    int lda = align_lda<T>(m);
   //    T *lcol = node.lcol;
   //    // T *contrib = nodes_[ni].contrib;

   //    int blksz = options.cpu_block_size;
   //    int nr = (m-1) / blksz + 1; // number of block rows
   //    int nc = (n-1) / blksz + 1; // number of block columns
   
   //    // int flag;
   //    // cholesky_factor(
   //    //       m, n, lcol, ldl, 0.0, NULL, 0, options.cpu_block_size, &flag
   //    //       );
   //    // int *info = new int(-1);

   //    for(int j = 0; j < nc; ++j) {

   //       int blkn = std::min(blksz, n - j*blksz);
         
   //       /* Diagonal Block Factorization Task */

   //       int blkm = std::min(blksz, m - j*blksz);
   //       // int flag = lapack_potrf(FILL_MODE_LWR, blkn, &a[j*(lda+1)], lda);
   //       // if(blkm>blkn) {
   //       //    // Diagonal block factored OK, handle some rectangular part of block
   //       //    host_trsm(SIDE_RIGHT, FILL_MODE_LWR, OP_T, DIAG_NON_UNIT,
   //       //              blkm-blkn, blkn, 1.0, &a[j*(lda+1)], lda,
   //       //              &a[j*(lda+1)+blkn], lda);

   //       //    // if(upd) {
   //       //    //    double rbeta = (j==0) ? beta : 1.0;
   //       //    //    host_syrk(FILL_MODE_LWR, OP_N, blkm-blkn, blkn, -1.0,
   //       //    //              &a[j*(lda+1)+blkn], lda, rbeta, upd, ldupd);
   //       //    // }
   //       // }
         
   //       factorize_diag_block(blkm, blkn, &lcol[j*blksz*(lda+1)], lda);
         
   //       /* Column Solve Tasks */
   //       for(int i = j+1; i < nr; ++i) {
            
   //          int blkm = std::min(blksz, m - i*blksz);
            
   //          solve_block(blkm, blkn, 
   //                      &lcol[j*blksz*(lda+1)], lda, 
   //                      &lcol[(j*blksz*lda) + (i*blksz)], lda);
            
   //          // host_trsm(SIDE_RIGHT, FILL_MODE_LWR, OP_T, DIAG_NON_UNIT,
   //          //           blkm, blkn, 1.0, &a[j*(lda+1)], lda, &a[j*lda+i], lda);

   //          // if(blkn<blksz && upd) {
   //          //    double rbeta = (j==0) ? beta : 1.0;
   //          //    host_gemm(OP_N, OP_T, blkm, blksz-blkn, blkn, -1.0,
   //          //              &a[j*lda+i], lda, &a[j*(lda+1)+blkn], lda,
   //          //              rbeta, &upd[i-n], ldupd);
   //          // }
   //       }

   //       /* Schur Update Tasks: mostly internal */
   //       for(int k = j+1; k < nc; ++k) {

   //          int blkk = std::min(blksz, n - k*blksz);

   //          for(int i = k;  i < nr; ++i) {
               
   //             int blkm = std::min(blksz, m - i*blksz);

   //             update_block(blkm, blkk, &lcol[ (k*blksz*lda) + (i*blksz)], lda,
   //                          blkn,
   //                          &lcol[(j*blksz*lda) + (i*blksz)], lda, 
   //                          &lcol[(j*blksz*lda) + (k*blksz)], lda);

   //             // host_gemm(OP_N, OP_T, blkm, blkk, blkn, -1.0, &a[j*lda+i], lda,
   //             //           &a[j*lda+k], lda, 1.0, &a[k*lda+i], lda);

   //             // if(blkk < blksz && upd) {
   //             //    double rbeta = (j==0) ? beta : 1.0;
   //             //    int upd_width = (m<k+blksz) ? blkm - blkk
   //             //       : blksz - blkk;
   //             //    if(i-n < 0) {
   //             //       // Special case for first block of contrib
   //             //       host_gemm(OP_N, OP_T, blkm+i-n, upd_width, blkn, -1.0,
   //             //                 &a[j*lda+n], lda, &a[j*lda+k+blkk], lda, rbeta,
   //             //                 upd, ldupd);
   //             //    } else {
   //             //       host_gemm(OP_N, OP_T, blkm, upd_width, blkn, -1.0,
   //             //                 &a[j*lda+i], lda, &a[j*lda+k+blkk], lda, rbeta,
   //             //                 &upd[i-n], ldupd);
   //             //    }
   //             // }
   //          }
   //       }
         
   //    }
   // }

   /* Update between task */

//    template <typename T, typename PoolAlloc>
//    void update_between_block_task(
//          SymbolicSNode &snode, // symbolic source node
//          spldlt::NumericNode<T, PoolAlloc> &node, // numeric source node
//          int cptr, int cptr2, // pointer to the first and last row of
//                               // A_jk block
//          int rptr, int rptr2, // pointer to the first and last row of
//                               // A_ik block
//          int kk, // column index of A_jk and A_ik blocks
//          SymbolicSNode &asnode, // symbolic destination node
//          spldlt::NumericNode<T, PoolAlloc> &anode, // numeric destination node
//          int ii, /* block row index of A_ij block in ancestor node  */
//          int jj, /* block column index of A_ij block in ancestor node */
//          int blksz, // blocking size
//          Workspace &work,
//          Workspace &rowmap, Workspace& colmap,
//          int prio) {

//       /* Extract useful information about node */
//       int m = snode.nrow;
//       int n = snode.ncol;

//       int nr = (m-1) / blksz + 1; // number of block rows
//       int nc = (n-1) / blksz + 1; // number of block columns
//       // printf("kk: %d, ii: %d, jj: %d\n", kk, ii, jj);
// #if defined(SPLDLT_USE_STARPU)

//       /* Extract useful information about anode */
//       int a_m = asnode.nrow;
//       int a_n = asnode.ncol;
//       int a_nr = (a_m-1) / blksz +1; // number of block rows
//       int a_nc = (a_n-1) / blksz +1; // number of block columns
      
//       // A_ik
//       // First block in A_ik
//       int blk_sa = rptr  / blksz;
//       int blk_en = rptr2 / blksz;
//       int nblk_ik = blk_en - blk_sa +1;
//       // printf("rptr: %d, rptr2: %d, blk_sa: %d, blk_en: %d, nblk_ik: %d\n", rptr, rptr2, blk_sa, blk_en, nblk_ik);
//       starpu_data_handle_t *bc_ik_hdls = new starpu_data_handle_t[nblk_ik];
//       for (int i = 0; i < nblk_ik; ++i) {
//          // printf("[update_between_block_task] a_ik handle %d ptr: %p\n", i, snode.handles[kk*nr + blk_sa + i]);
//          bc_ik_hdls[i] = snode.handles[kk*nr + blk_sa + i];
//       }

//       // A_jk
//       // First block in A_jk
//       blk_sa = cptr  / blksz;
//       blk_en = cptr2 / blksz;
//       int nblk_jk = blk_en - blk_sa +1;
//       // printf("cptr: %d, cptr2: %d, blk_sa: %d, blk_en: %d, nblk_jk: %d\n", cptr, cptr2, blk_sa, blk_en, nblk_jk);      
//       starpu_data_handle_t *bc_jk_hdls = new starpu_data_handle_t[nblk_jk];
//       for (int i = 0; i < nblk_jk; ++i) {
//          // printf("[update_between_block_task] a_jk handle %d ptr: %p\n", i, snode.handles[kk*nr + blk_sa + i]);
//          bc_jk_hdls[i] = snode.handles[kk*nr + blk_sa + i];
//       }
//       // printf("[update_between_block_task] a_nr: %d, ii: %d, jj: %d\n", a_nr, ii, jj);
//       // printf("[update_between_block_task] a_ij handle ptr: %p\n", asnode.handles[jj*a_nr + ii]);
//       insert_update_between(
//             &snode, &node,
//             bc_ik_hdls, nblk_ik,
//             bc_jk_hdls, nblk_jk,
//             kk,
//             &asnode,
//             asnode.hdl,
//             asnode.handles[jj*a_nr + ii],
//             ii, jj,
//             work.hdl, rowmap.hdl, colmap.hdl,
//             blksz,
//             cptr, cptr2,
//             rptr, rptr2,
//             prio);

//       // int blkn = std::min(blksz, n - kk*blksz);
//       // T *a_lcol = anode.lcol;
//       // int a_ldl = align_lda<T>(asnode.nrow);

//       // int mc = cptr2-cptr+1; // number of rows in Ajk
//       // int *clst = colmap.get_ptr<int>(mc);
//       // int mr = rptr2-rptr+1; // number of rows in Aik
//       // int *rlst = rowmap.get_ptr<int>(mr); // Get ptr on rowmap array
//       // T *buffer = work.get_ptr<T>(mr*mc);

//       // update_between_block(blkn, 
//       //                      kk, ii, jj, 
//       //                      blksz, 
//       //                      cptr, cptr2, rptr, rptr2,
//       //                      snode, node,
//       //                      asnode, &a_lcol[(jj*blksz*a_ldl)+(ii*blksz)], a_ldl,
//       //                      buffer, rlst, clst);
      
//       delete[] bc_ik_hdls;
//       delete[] bc_jk_hdls;

// #else

//       int blkn = std::min(blksz, n - kk*blksz);
//       T *a_lcol = anode.lcol;
//       int a_ldl = align_lda<T>(asnode.nrow);

//       int mc = cptr2-cptr+1; // number of rows in Ajk
//       int *clst = colmap.get_ptr<int>(mc);
//       int mr = rptr2-rptr+1; // number of rows in Aik
//       int *rlst = rowmap.get_ptr<int>(mr); // Get ptr on rowmap array
//       T *buffer = work.get_ptr<T>(mr*mc);
//       // T *buffer = work.get_ptr<T>(blksz*blksz);
//       // printf("[update_between_block_task] kk: %d, ii: %d, jj: %d\n", kk, ii, jj);
//       // printf("[update_between_block_task] cptr: %d, cptr2: %d, rptr: %d, rptr2: %d\n", cptr, cptr2, rptr, rptr2);

//       update_between_block(blkn,
//                            kk, ii, jj,
//                            blksz, 
//                            cptr, cptr2, rptr, rptr2,
//                            snode, node,
//                            asnode, &a_lcol[(jj*blksz*a_ldl)+(ii*blksz)], a_ldl,
//                            buffer, rlst, clst);

// #endif

//    }

   /* Apply factorization to ancestor node */
   
   // template <typename T, typename PoolAlloc>
   // void apply_node(
   //       SymbolicSNode &snode, // symbolic node to be applied 
   //       spldlt::NumericNode<T, PoolAlloc> &node, // numeric node to be applied
   //       int nnodes,
   //       SymbolicTree &stree, // symbolic tree
   //       std::vector<spldlt::NumericNode<T,PoolAlloc>> &nodes, // list of nodes in the tree
   //       int blksz, // blocking size
   //       // // struct cpu_factor_options const& options
   //       Workspace &work,
   //       Workspace &rowmap, Workspace& colmap
   //       ) {

   //    // return;
   //    // printf("node idx: %d\n", snode.idx);
   //    // return;

   //    /* Extract useful information about node */
   //    int m = snode.nrow; // number of row in node
   //    int n = snode.ncol; // number of column in node
   //    int ldl = align_lda<T>(m); // leading dimensions
   //    T *lcol = node.lcol; // numerical values
   //    int nc = (n-1)/blksz +1; // number of block column

   //    Workspace map(stree.n); // maxfront size
   //    int *amap = map.get_ptr<int>(stree.n); // row to block row mapping array 
   //    bool map_done = false;
      
   //    int cptr = n; // point to first row below diag in node
   //    int cptr2 = 0; 
      
   //    int parent = snode.parent;

   //    int prio = 0;

   //    while (parent < nnodes) {
   //       // printf("[apply_node] parent: %d\n", parent);
   //       // NumericNode<T,PoolAllocator> &anode = nodes_[parent];

   //       SymbolicSNode &asnode = stree[parent]; // parent symbolic node 
   //       int sa = asnode.sa;
   //       int en = asnode.en;


   //       T *a_lcol = nodes[asnode.idx].lcol;
   //       int a_ldl = align_lda<T>(asnode.nrow);

   //       map_done = false;
               
   //       while (cptr < m) {
   //          if (snode.rlist[cptr] >= sa) break;
   //          cptr++;
   //       }
   //       if (cptr >= m) break;

   //       while(cptr < m) {
   //          if (snode.rlist[cptr] > en) break;
                  
   //          // determine last column index of current block column 
   //          int cb = (snode.rlist[cptr] - sa) / blksz;
   //          int jlast = std::min(sa + (cb+1)*blksz-1, en);

   //          // printf("[NumericTree] cb: %d\n", cb);

   //          // find cptr2
   //          cptr2 = cptr;
   //          while (cptr2 < m) {
   //             if (snode.rlist[cptr2] > jlast) break;
   //             cptr2++;
   //          }
   //          cptr2--;

   //          // printf("cptr: %d, rlist[cptr]: %d, sa: %d, cptr2: %d, rlist[cptr2]: %d, en: %d\n", 
   //          //        cptr, snode.rlist[cptr], sa, cptr2, snode.rlist[cptr2], en);
                  
   //          if (!map_done) {
   //             // int a_blksz = blksz;
   //             int r = 0; // row index
   //             int rr = 0; // block row index
   //             for (int row = 0; row < asnode.nrow; ++row) {
   //                rr = row / blksz;
   //                r = asnode.rlist[row];
   //                amap[r] = rr;
   //             }
   //             map_done = true;
   //          }

   //          int mc = cptr2-cptr+1; // number of rows in Ajk
   //          int *clst = colmap.get_ptr<int>(mc);

   //          int rptr = cptr;
   //          int ii = amap[snode.rlist[cptr]]; // first block row in anode

   //          int rptr2 = 0;
   //          for (rptr2 = cptr; rptr2 < m; rptr2++) {
   //             int k = amap[snode.rlist[rptr2]];
                     
   //             if (k != ii) {
                  
   //                int mr = rptr2-rptr; // number of rows in Aik
   //                int *rlst = rowmap.get_ptr<int>(mr); // Get ptr on rowmap array
   //                T *buffer = work.get_ptr<T>(mr*mc);
                        
   //                for (int kk = 0; kk < nc; ++kk) {

   //                   int blkn = std::min(blksz, n-(kk*blksz));

   //                   update_between_block_task(
   //                         snode, node,
   //                         cptr, cptr2, rptr, rptr2-1,
   //                         kk,
   //                         asnode, nodes[asnode.idx],
   //                         ii, cb,
   //                         blksz,
   //                         work, rowmap, colmap,
   //                         prio);

   //                   // update_between_block(blkn,
   //                   //                      kk, ii, cb,
   //                   //                      blksz, 
   //                   //                      cptr, cptr2, rptr, rptr2-1,
   //                   //                      snode, node,
   //                   //                      asnode, 
   //                   //                      &a_lcol[(cb*blksz*a_ldl)+(ii*blksz)], a_ldl,
   //                   //                      buffer, rlst, clst);

   //                }


   //                ii = k;
   //                rptr = rptr2;
   //             }
   //          }

   //          int mr = rptr2-rptr+1; // number of rows in Aik
   //          int *rlst = rowmap.get_ptr<int>(mr); // Get ptr on rowmap array
   //          T *buffer = work.get_ptr<T>(mr*mc);

   //          for (int kk = 0; kk < nc; ++kk) {

   //             int blkn = std::min(blksz, n-(kk*blksz));

   //             update_between_block_task(
   //                   snode, node,
   //                   cptr, cptr2, rptr, rptr2-1,
   //                   kk, 
   //                   asnode, nodes[asnode.idx],
   //                   ii, cb,
   //                   blksz,
   //                   work, rowmap, colmap,
   //                   prio);                     

   //             // update_between_block(blkn,
   //             //                      kk, ii, cb,
   //             //                      blksz, 
   //             //                      cptr, cptr2, rptr, rptr2-1,
   //             //                      snode, node,
   //             //                      asnode, 
   //             //                      &a_lcol[(cb*blksz*a_ldl)+(ii*blksz)], a_ldl,
   //             //                      buffer, rlst, clst);

   //          }
            
   //          cptr = cptr2 + 1; // move cptr
   //       }

   //       parent = stree[asnode.idx].parent; // move up the tree
   //    }
   // }

   // Serial version

   // template <typename T, typename PoolAlloc>
   // void apply_node_notask(
   //       SymbolicSNode const& snode, // symbolic node to be applied 
   //       spldlt::NumericNode<T, PoolAlloc> &node, // numeric node to be applied
   //       int nnodes,
   //       SymbolicTree const& stree, // symbolic tree
   //       std::vector<spldlt::NumericNode<T,PoolAlloc>> &nodes, // list of nodes in the tree
   //       int nb, // blocking size
   //       // struct cpu_factor_options const& options
   //       Workspace &work,
   //       Workspace &rowmap, Workspace& colmap
   //       ) {
      
   //    /* Extract useful information about node */
   //    int m = snode.nrow; // number of row in node
   //    int n = snode.ncol; // number of column in node
   //    int ldl = align_lda<T>(m); // leading dimensions
   //    T *lcol = node.lcol; // numerical values
   //    int nc = (n-1)/nb +1; // number of block column

   //    Workspace map(stree.n); // maxfront size
   //    int *amap = map.get_ptr<int>(stree.n); // row to block row mapping array 
   //    bool map_done = false;
      
   //    int cptr = n; // point to first row below diag in node
   //    int cptr2 = 0; 
      
   //    int parent = snode.parent;
   //    while (parent < nnodes) {
   //       // NumericNode<T,PoolAllocator> &anode = nodes_[parent];
   //       SymbolicSNode const& asnode = stree[parent]; // parent symbolic node 
   //       int sa = asnode.sa;
   //       int en = asnode.en;


   //       T *a_lcol = nodes[asnode.idx].lcol;
   //       int a_ldl = align_lda<T>(asnode.nrow);

   //       map_done = false;
   //       // printf("[NumericTree] node: %d, parent: %d, sa: %d, en: %d\n", ni, asnode.idx, sa, en);

   //       // printf("cptr: %d, rlist[cptr]: %d, cptr2: %d, rlist[cptr2]: %d\n", 
   //       //        cptr, snode.rlist[cptr], cptr2, snode.rlist[cptr2]);
               
   //       // for (int i = 0; i < m; i++) 
   //       //    printf(" %d ", snode.rlist[i]);
   //       // printf("\n");
               
   //       while (cptr < m) {
   //          if (snode.rlist[cptr] >= sa) break;
   //          cptr++;
   //       }
   //       if (cptr >= m) break;

   //       while(cptr < m) {
   //          if (snode.rlist[cptr] > en) break;
                  
   //          // determine last column index of current block column 
   //          int cb = (snode.rlist[cptr] - sa) / nb;
   //          int jlast = std::min(sa + (cb+1)*nb, en);

   //          // printf("[NumericTree] cb: %d\n", cb);

   //          // find cptr2
   //          cptr2 = cptr;
   //          while (cptr2 < m) {
   //             if (snode.rlist[cptr2] > jlast) break;
   //             cptr2++;
   //          }
   //          cptr2--;

   //          // printf("cptr: %d, rlist[cptr]: %d, sa: %d, cptr2: %d, rlist[cptr2]: %d, en: %d\n", 
   //          //        cptr, snode.rlist[cptr], sa, cptr2, snode.rlist[cptr2], en);
                  
   //          if (!map_done) {
   //             // int a_nb = nb;
   //             int r = 0; // row index
   //             int rr = 0; // block row index
   //             for (int row = 0; row < asnode.nrow; ++row) {
   //                rr = row / nb;
   //                r = asnode.rlist[row];
   //                amap[r] = rr;
   //             }
   //             map_done = true;
   //          }

   //          int rptr = cptr;
   //          int ii = amap[snode.rlist[cptr]]; // first block row in anode

   //          int i = 0;
   //          for (i = cptr; i < m; i++) {
   //             int k = amap[snode.rlist[i]];
                     
   //             if (k != ii) {
                        
   //                for (int kk = 0; kk < nc; ++kk) {
                     
   //                   int blkn = std::min(nb, n-(kk*nb));
                     
   //                   update_between_block(blkn, kk*nb, cptr, cptr2, rptr, i-1,
   //                                        snode, node,
   //                                        asnode, nodes[asnode.idx],
   //                                        work, rowmap, colmap);
   //                }


   //                ii = k;
   //                rptr = i;
   //             }
   //          }

   //          for (int kk = 0; kk < nc; ++kk) {
                     
   //             int blkn = std::min(nb, n-(kk*nb));
                     
   //             update_between_block(blkn, kk*nb, cptr, cptr2, rptr, i-1, // cptr, m-1,
   //                                  snode, node,
   //                                  asnode, nodes[asnode.idx],
   //                                  work, rowmap, colmap);
   //          }

   //          cptr = cptr2 + 1; // move cptr
   //       }

   //       parent = stree[asnode.idx].parent; // move up the tree
   //    }
   // }

} /* end of namespace spldlt */
