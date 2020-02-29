/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// STD
#include <vector>

// SpLDLT
#include "sylver_ciface.hxx"
#include "kernels/assemble.hxx"
#include "kernels/factor.hxx"
#if defined(SPLDLT_USE_GPU)
#include "kernels/gpu/wrappers.hxx"
#include "StarPU/cuda/kernels.hxx"
#endif

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/contrib.h"
// StarPU
#include <starpu.h>
#if defined(SPLDLT_USE_GPU)
#include <starpu_cublas_v2.h>
#endif
// OMP
#if defined(SPLDLT_USE_OMP)
#include "omp.hxx"
#endif

namespace spldlt {
namespace starpu {

   // unregister handles for a node in StarPU
   template <typename T, typename PoolAlloc>
   void unregister_node_submit(
         NumericFront<T, PoolAlloc> &node
         ) {

      // printf("[unregister_node_submit]\n");
         
      // Get node info
      sylver::SymbolicFront &snode = node.symb();
      int blksz = node.blksz();
      int m = node.nrow();
      int n = node.ncol();
      int nr = node.nr(); // Number of block rows
      int nc = node.nc(); // Number of block columns

      // Unregister block handles in the factors
      for(int j = 0; j < nc; ++j) {
         for(int i = j; i < nr; ++i) {
            starpu_data_unregister_submit(snode.handles[i + j*nr]);
         }
      }

      // Unregister block handles in the contribution blocks
      int ldcontrib = m-n;

      if (ldcontrib>0) {
         // Index of first block in contrib
         int rsa = n/blksz;
         // Number of block in contrib
         int ncontrib = nr-rsa;

         for(int j = rsa; j < nr; j++) {
            for(int i = j; i < nr; i++) {
               // Register block in StarPU
               // node.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].unregister_handle();
               node.contrib_block(i, j).unregister_handle();
            }
         }

      }
   }

   /* Unregister handles in StarPU*/
   // void unregister

   ////////////////////////////////////////////////////////////////////////////////      
   // init_node StarPU task
   // CPU task
   template <typename T, typename PoolAlloc>
   void init_node_cpu_func(void *buffers[], void *cl_arg) {

      NumericFront<T, PoolAlloc> *front = nullptr;
      T *aval;
      T *scaling;

      starpu_codelet_unpack_args(
            cl_arg,
            &front,
            &aval,
            &scaling);

      init_node(*front, aval, scaling);
   }

   // init_node StarPU codelet
   extern struct starpu_codelet cl_init_node;      

   /// @brief Launch StarPU task for intializing a front
   ///
   /// @param front Frontal martix
   /// @param node_hdl StarPU handle (symbolic) associated with the
   /// front
   /// @param aval Numerical values in the original matrix,
   /// orginised according to a CSC format
   /// @param prio Task priority
   template <typename T, typename PoolAlloc>
   void insert_init_node(
         NumericFront<T, PoolAlloc> *front,
         starpu_data_handle_t node_hdl,
         T *aval, T *scaling, int prio) {
                  
      int ret;

      int nr = front->nr();
      int nc = front->nc();

      int nhdls = nr*nc;
      struct starpu_data_descr *descrs = new starpu_data_descr[nhdls+1];
         
      int nh = 0;
      for(int j = 0; j < nc; ++j) {
         for(int i = j; i < nr; ++i) {
            descrs[nh].handle = front->get_block(i,j).get_hdl();
            descrs[nh].mode = STARPU_RW;
            ++nh;
         }
      }

      descrs[nh].handle = node_hdl; descrs[nh].mode = STARPU_RW;
      ++nh;
         
      ret = starpu_insert_task(
            &cl_init_node,
            STARPU_DATA_MODE_ARRAY, descrs, nh,
            // STARPU_RW, node_hdl,
            STARPU_VALUE, &front, sizeof(NumericFront<T, PoolAlloc>*),
            STARPU_VALUE, &aval, sizeof(T*),
            STARPU_VALUE, &scaling, sizeof(T*),
            STARPU_PRIORITY, prio,
            0);

      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

      delete[] descrs;
   }

   ////////////////////////////////////////////////////////////////////////////////
   // fini_node StarPU kernels
   // fini_node CPU kernel
   template <typename T, typename PoolAlloc>
   void fini_node_cpu_func(void *buffers[], void *cl_arg) {
         
      NumericFront<T, PoolAlloc> *node = nullptr;
      bool posdef;
         
      starpu_codelet_unpack_args(cl_arg, &node, &posdef);

      // printf("[fini_node_cpu_func] node idx = %d\n", node->symb.idx+1);
      if (posdef) {
         unregister_node_posdef<T, PoolAlloc, true>(*node);
      }
      else {
         unregister_node_indef<T, PoolAlloc, true>(*node);
         // unregister_node_indef<T, PoolAlloc, false>(*node);
      }
         
      fini_node(*node);
   }

   // fini_node codelet
   extern struct starpu_codelet cl_fini_node;
      
   template <typename T, typename PoolAlloc>
   void insert_fini_node(
         starpu_data_handle_t node_hdl,
         starpu_data_handle_t *hdls, int nhdl, // Children node's symbolic handles
         NumericFront<T, PoolAlloc> *node,
         bool posdef,
         int prio) {

      int ret;
      int nh = 0;
         
      struct starpu_data_descr *descrs = new starpu_data_descr[nhdl+1];

      for (int i=0; i<nhdl; i++) {
         descrs[nh].handle = hdls[i]; descrs[nh].mode = STARPU_RW;
         ++nh;
      }

      descrs[nh].handle = node_hdl; descrs[nh].mode = STARPU_RW;
      ++nh;
         
      ret = starpu_insert_task(
            &cl_fini_node,
            STARPU_DATA_MODE_ARRAY, descrs, nh,
            STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
            STARPU_VALUE, &posdef, sizeof(bool),
            STARPU_PRIORITY, prio,
            0);

      delete[] descrs;
   }

   ////////////////////////////////////////////////////////////////////////////////
   // factorize_block StarPU task

   // CPU kernel
   template<typename T>
   void factorize_block_cpu_func(void *buffers[], void *cl_arg) {
         
      T *blk = (T *)STARPU_MATRIX_GET_PTR(buffers[0]);
      unsigned m = STARPU_MATRIX_GET_NX(buffers[0]);
      unsigned n = STARPU_MATRIX_GET_NY(buffers[0]);
      unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);

      std::vector<sylver::inform_t> *worker_stats;      
      starpu_codelet_unpack_args(cl_arg, &worker_stats);
      int workerid = starpu_worker_get_id();
      sylver::inform_t& inform = (*worker_stats)[workerid];
      
      int flag = factorize_diag_block(m, n, blk, ld);

      if (flag > 0)
         inform.flag = sylver::Flag::ERROR_NOT_POS_DEF;
         
      // std::cout << "[factorize_block_cpu_func] Error, flag = " << flag << std::endl;
   }

   // CPU kernel
   template<typename T>      
   void factorize_contrib_block_cpu_func(void *buffers[], void *cl_arg) {
         
      T *blk = (T *)STARPU_MATRIX_GET_PTR(buffers[0]);
      unsigned m = STARPU_MATRIX_GET_NX(buffers[0]);
      unsigned n = STARPU_MATRIX_GET_NY(buffers[0]);
      unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);

      T *contrib = (T *)STARPU_MATRIX_GET_PTR(buffers[1]);
      unsigned mcontrib = STARPU_MATRIX_GET_NX(buffers[1]);
      unsigned ncontrib = STARPU_MATRIX_GET_NY(buffers[1]);
      unsigned ldcontrib = STARPU_MATRIX_GET_LD(buffers[1]);

      int k;
      std::vector<sylver::inform_t> *worker_stats;
      starpu_codelet_unpack_args(
            cl_arg,
            &k,
            &worker_stats);
      int workerid = starpu_worker_get_id();
      sylver::inform_t& inform = (*worker_stats)[workerid];

      int flag = factorize_diag_block(m, n, blk, ld, contrib, ldcontrib,
                                      k==0);

      if (flag > 0) 
         inform.flag = sylver::Flag::ERROR_NOT_POS_DEF;
      

   }
      
   /* FIXME: although it would be better to statically initialize
      the codelet, it is not well suported by g++ */

   // struct starpu_codelet cl_factorize_block = {
   //    // .where = w,
   //    // .cpu_funcs = {factorize_block_cpu_func, NULL},
   //    .nbuffers = STARPU_VARIABLE_NBUFFERS,
   //    // .name = "FACTO_BLK"
   // };

   // factorize block codelet
   extern struct starpu_codelet cl_factorize_contrib_block;

   void insert_factor_block(
         int k, 
         starpu_data_handle_t bc_hdl,
         starpu_data_handle_t contrib_hdl,
         starpu_data_handle_t node_hdl, // Symbolic node handle
         int prio,
         std::vector<sylver::inform_t> *worker_stats);
      
   ////////////////////////////////////////////////////////////////////////////////   
   // factorize_block

   // factorize block codelet
   extern struct starpu_codelet cl_factorize_block;      

   void insert_factor_block(
         starpu_data_handle_t bc_hdl,
         starpu_data_handle_t node_hdl, // Symbolic node handle
         int prio,
         std::vector<sylver::inform_t> *worker_stats);
      
   ////////////////////////////////////////////////////////////////////////////////
   // solve_block

   // CPU kernel
   template<typename T>
   void solve_block_cpu_func(void *buffers[], void *cl_arg) {

      /* Get diag block pointer and info */
      T *blk = (T *)STARPU_MATRIX_GET_PTR(buffers[0]);
      unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);

      /* Get sub diag block pointer and info */
      T *blk_ik = (T *)STARPU_MATRIX_GET_PTR(buffers[1]);
      unsigned m = STARPU_MATRIX_GET_NX(buffers[1]);
      unsigned n = STARPU_MATRIX_GET_NY(buffers[1]);
      unsigned ld_ik = STARPU_MATRIX_GET_LD(buffers[1]);

      /* Call kernel function */
      solve_block(m, n, blk, ld, blk_ik, ld_ik);
   }

   // solve_block codelet
   extern struct starpu_codelet cl_solve_block;

   // Insert solve of subdiag block into StarPU
   void insert_solve_block(
         starpu_data_handle_t bc_kk_hdl, /* diag block handle */
         starpu_data_handle_t bc_ik_hdl, /* sub diag block handle */
         starpu_data_handle_t node_hdl,
         int prio
         );
      
   ////////////////////////////////////////////////////////////////////////////////
   // solve_contrib_block

   // CPU task
   template<typename T>
   void solve_contrib_block_cpu_func(void *buffers[], void *cl_arg) {

      // Get diag block pointer and info
      T *blk = (T *)STARPU_MATRIX_GET_PTR(buffers[0]);
      unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);

      // Get sub diag block pointer and info
      T *blk_ik = (T *)STARPU_MATRIX_GET_PTR(buffers[1]);
      unsigned m = STARPU_MATRIX_GET_NX(buffers[1]);
      unsigned n = STARPU_MATRIX_GET_NY(buffers[1]);
      unsigned ld_ik = STARPU_MATRIX_GET_LD(buffers[1]);

      T *contrib = (T *)STARPU_MATRIX_GET_PTR(buffers[2]);
      unsigned ldcontrib = STARPU_MATRIX_GET_LD(buffers[2]);

      int k, nb;

      starpu_codelet_unpack_args(
            cl_arg, &k, &nb);

      solve_block(m, n, blk, ld, blk_ik, ld_ik,
                  contrib, ldcontrib, k==0, nb);
   }

   extern struct starpu_codelet cl_solve_contrib_block;

   // Insert solve of subdiag block into StarPU
   void insert_solve_block(
         int k, int nb,
         starpu_data_handle_t bc_kk_hdl, // Diag block handle
         starpu_data_handle_t bc_ik_hdl, // Sub diag block handle
         starpu_data_handle_t contrib_hdl, // Contrib block handle
         starpu_data_handle_t node_hdl,
         int prio);
      
   ////////////////////////////////////////////////////////////
   // update_block StarPU task

   // CPU kernel
   template<typename T>
   void update_block_cpu_func(void *buffers[], void *cl_arg) {

      /* Get A_ij pointer */
      T *blk_ij = (T *)STARPU_MATRIX_GET_PTR(buffers[0]);
      unsigned m = STARPU_MATRIX_GET_NX(buffers[0]);
      unsigned n = STARPU_MATRIX_GET_NY(buffers[0]);
      unsigned ld_ij = STARPU_MATRIX_GET_LD(buffers[0]);
                  
      /* Get A_ik pointer */
      T *blk_ik = (T *)STARPU_MATRIX_GET_PTR(buffers[1]);
      unsigned k = STARPU_MATRIX_GET_NY(buffers[1]);
      unsigned ld_ik = STARPU_MATRIX_GET_LD(buffers[1]); 

      T *blk_jk = (T *)STARPU_MATRIX_GET_PTR(buffers[2]);
      unsigned ld_jk = STARPU_MATRIX_GET_LD(buffers[2]); 
        
      update_block(m, n, blk_ij, ld_ij,
                   k,
                   blk_ik, ld_ik, 
                   blk_jk, ld_jk);
   }

   // update_block codelet
   extern struct starpu_codelet cl_update_block;      

   void insert_update_block(
         starpu_data_handle_t bc_ij_hdl, /* A_ij block handle */
         starpu_data_handle_t bc_ik_hdl, /* A_ik block handle */
         starpu_data_handle_t bc_jk_hdl, /* A_jk block handle */
         starpu_data_handle_t node_hdl,
         int prio);

   ////////////////////////////////////////////////////////////
   // update_contrib_block StarPU task
   
   // CPU kernel
   template<typename T>
   void update_contrib_block_cpu_func(void *buffers[], void *cl_arg) {

      /* Get A_ij pointer */
      T *blk_ij = (T *)STARPU_MATRIX_GET_PTR(buffers[0]);
      unsigned m = STARPU_MATRIX_GET_NX(buffers[0]);
      unsigned n = STARPU_MATRIX_GET_NY(buffers[0]);
      unsigned ld_ij = STARPU_MATRIX_GET_LD(buffers[0]);
                  
      /* Get A_ik pointer */
      T *blk_ik = (T *)STARPU_MATRIX_GET_PTR(buffers[1]);
      unsigned k = STARPU_MATRIX_GET_NY(buffers[1]);
      unsigned ld_ik = STARPU_MATRIX_GET_LD(buffers[1]);

      T *blk_jk = (T *)STARPU_MATRIX_GET_PTR(buffers[2]);
      unsigned ld_jk = STARPU_MATRIX_GET_LD(buffers[2]);

      T *contrib = (T *)STARPU_MATRIX_GET_PTR(buffers[3]);
      unsigned cbm = STARPU_MATRIX_GET_NX(buffers[3]);
      unsigned cbn = STARPU_MATRIX_GET_NY(buffers[3]);
      unsigned ldcontrib = STARPU_MATRIX_GET_LD(buffers[3]);

      int kk, blksz;

      starpu_codelet_unpack_args(
            cl_arg, &kk, &blksz);

      update_block(m, n, blk_ij, ld_ij,
                   k,
                   blk_ik, ld_ik, 
                   blk_jk, ld_jk,
                   contrib, ldcontrib,
                   cbm, cbn,
                   kk==0, blksz);
   }

   extern struct starpu_codelet cl_update_contrib_block;      

   void insert_update_block(
         int k, int nb,
         starpu_data_handle_t bc_ij_hdl, /* A_ij block handle */
         starpu_data_handle_t bc_ik_hdl, /* A_ik block handle */
         starpu_data_handle_t bc_jk_hdl, /* A_jk block handle */
         starpu_data_handle_t contrib_hdl, /* A_ij block handle */
         starpu_data_handle_t node_hdl,
         int prio);
      
   // update_diag_block StarPU task

   // void update_diag_block_cpu_func(void *buffers[], void *cl_arg) {

   //    /* Get A_ij pointer */
   //    double *blk_ij = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
   //    unsigned m = STARPU_MATRIX_GET_NX(buffers[0]);
   //    unsigned n = STARPU_MATRIX_GET_NY(buffers[0]);
   //    unsigned ld_ij = STARPU_MATRIX_GET_LD(buffers[0]);
                  
   //    /* Get A_ik pointer */
   //    double *blk_ik = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
   //    unsigned k = STARPU_MATRIX_GET_NY(buffers[1]);
   //    unsigned ld_ik = STARPU_MATRIX_GET_LD(buffers[1]); 

   //    double *blk_jk = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
   //    unsigned ld_jk = STARPU_MATRIX_GET_LD(buffers[2]); 
        
   //    update_diag_block(m, n, blk_ij, ld_ij,
   //                      k,
   //                      blk_ik, ld_ik, 
   //                      blk_jk, ld_jk);

   // }

   // /* update diag block codelet */
   // struct starpu_codelet cl_update_diag_block;      

   // void insert_update_diag_block(
   //       starpu_data_handle_t bc_ij_hdl, /* A_ij block handle */
   //       starpu_data_handle_t bc_ik_hdl, /* A_ik block handle */
   //       starpu_data_handle_t bc_jk_hdl, /* A_jk block handle */
   //       int prio) {

   //    int ret;

   //    ret = starpu_insert_task(
   //          &cl_update_diag_block,
   //          STARPU_RW, bc_ij_hdl,
   //          STARPU_R, bc_ik_hdl,
   //          STARPU_R, bc_jk_hdl,
   //          STARPU_PRIORITY, prio,
   //          0);

   //    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");         
   // }

   ////////////////////////////////////////////////////////////////////////////////
   // update_contrib StarPU task

   // CPU task
   template<typename T>
   void update_contrib_cpu_func(void *buffers[], void *cl_arg) {

      // Get A_ij pointer
      T *blk_ij = (T *)STARPU_MATRIX_GET_PTR(buffers[0]);
      unsigned m = STARPU_MATRIX_GET_NX(buffers[0]);
      unsigned n = STARPU_MATRIX_GET_NY(buffers[0]);
      unsigned ld_ij = STARPU_MATRIX_GET_LD(buffers[0]);
                  
      // Get A_ik pointer
      T *blk_ik = (T *)STARPU_MATRIX_GET_PTR(buffers[1]);
      unsigned k = STARPU_MATRIX_GET_NY(buffers[1]);
      unsigned ld_ik = STARPU_MATRIX_GET_LD(buffers[1]); 

      T *blk_jk = (T *)STARPU_MATRIX_GET_PTR(buffers[2]);
      unsigned ld_jk = STARPU_MATRIX_GET_LD(buffers[2]); 
        
      int kk;

      starpu_codelet_unpack_args(cl_arg, &kk);

      update_block(m, n, blk_ij, ld_ij,
                   k,
                   blk_ik, ld_ik, 
                   blk_jk, ld_jk,
                   kk==0);
   }

   // update_contrib codelet
   extern struct starpu_codelet cl_update_contrib;      

   void insert_update_contrib(
         int k,
         starpu_data_handle_t bc_ij_hdl, /* A_ij block handle */
         starpu_data_handle_t bc_ik_hdl, /* A_ik block handle */
         starpu_data_handle_t bc_jk_hdl, /* A_jk block handle */
         starpu_data_handle_t node_hdl,
         int prio);
      
   ////////////////////////////////////////////////////////////////////////////////
   // update_between StarPU task

   /* CPU task */

   // template <typename T, typename PoolAlloc>
   // void update_between_cpu_func(void *buffers[], void *cl_arg) {

   //    /* Get workspace */
   //    T *buffer = (T *)STARPU_MATRIX_GET_PTR(buffers[0]);
   //    // unsigned buf_m = STARPU_MATRIX_GET_NX(buffers[0]);
   //    // unsigned buf_n = STARPU_MATRIX_GET_NY(buffers[0]);
   //    // unsigned buf_ld = STARPU_MATRIX_GET_LD(buffers[0]);
   //    // printf("[update_between_cpu_func] buf_m: %d, buf_n: %d, buf_ld: %d\n", buf_m, buf_n, buf_ld);
   //    /* Get rowmap workspace */
   //    int *rlst = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
   //    /* Get colmap workspace */
   //    int *clst = (int *)STARPU_VECTOR_GET_PTR(buffers[2]);
   //    /* Get A_ij block */
   //    T *a_ij = (T *)STARPU_MATRIX_GET_PTR(buffers[3]);
   //    unsigned lda = STARPU_MATRIX_GET_LD(buffers[3]);
   //    /* Get any A_ik block width */ 
   //    // unsigned n = STARPU_MATRIX_GET_NY(buffers[4]);
        
   //    int kk, ii, jj;
   //    int blksz;
   //    int cptr, cptr2;
   //    int rptr, rptr2;

   //    SymbolicSNode *snode = nullptr, *asnode = nullptr;
   //    spldlt::NumericNode<T, PoolAlloc> *node = nullptr;

   //    starpu_codelet_unpack_args(
   //          cl_arg,
   //          &snode,
   //          &node,
   //          &kk,
   //          &asnode,
   //          &ii, &jj,
   //          &blksz,
   //          &cptr, &cptr2,
   //          &rptr, &rptr2
   //          );

   //    // printf("[update_between_cpu_func] n: %d\n", n);
   //    // printf("[update_between_cpu_func] kk: %d, ii: %d, jj: %d\n", kk, ii, jj);
   //    // printf("[update_between_block_task] cptr: %d, cptr2: %d, rptr: %d, rptr2: %d\n", cptr, cptr2, rptr, rptr2);

   //    // printf("[update_between_cpu_func] snode: %p\n", snode);
   //    // printf("[update_between_cpu_func]  snode nrow: %d\n", snode->nrow);
   //    // printf("[update_between_cpu_func] asnode nrow: %d\n", asnode->nrow);

   //    // int mc = cptr2-cptr+1; // number of rows in Ajk
   //    // int mr = rptr2-rptr+1; // number of rows in Aik
   //    // printf("[update_between_cpu_func] mr: %d, mc: %d, n: %d, blksz: %d\n", mr, mc, n, blksz);
   //    // rlst = new int[mr]; // Get ptr on rowmap array
   //    // clst = new int[mc];
   //    // buffer = new T[mr*mc];
   //    // printf("[update_between_cpu_func] buffer: %p\n", buffer);
   //    // for (int j = 0; j < blksz; j++)
   //    //    for (int i = 0; i < blksz; i++)
   //    //       buffer[j*blksz+i] = 0.0;

   //    // Determine A_ik and A_kj column width 
   //    int blkn = std::min(snode->ncol - kk*blksz, blksz);
   //    // printf("[update_between_cpu_func] blkn: %d\n", blkn);

   //    update_between_block(
   //          blkn, // blcok column width
   //          kk, ii, jj, // block row and block column index of A_ij block in destination node
   //          blksz, // block size
   //          cptr, cptr2, // local row indexes of a_kj elements 
   //          // in source node
   //          rptr, rptr2, // local row indexes of a_ik elements 
   //          // in source node
   //          *snode, // symbolic source node  
   //          *node, // numeric source node
   //          *asnode,  // symbolic destination node
   //          a_ij, lda, // block to be updated in destination node  
   //          buffer, // workspace
   //          rlst, clst // workpaces for col and row mapping
   //          );
   // }

   // /* update beween codelet */
   // struct starpu_codelet cl_update_between;      
      
   // /* Note that bc_ik_hdls and bc_jk_hdls hadles are used for coherency */
   // template <typename T, typename PoolAlloc>
   // void insert_update_between(
   //       SymbolicSNode *snode,
   //       spldlt::NumericNode<T, PoolAlloc> *node,
   //       starpu_data_handle_t *bc_ik_hdls, int nblk_ik, /* A_ij block handle */
   //       starpu_data_handle_t *bc_jk_hdls, int nblk_jk, /* A_ik block handle */
   //       int kk,
   //       SymbolicSNode *asnode,
   //       starpu_data_handle_t anode_hdl,
   //       starpu_data_handle_t bc_ij_hdl,
   //       int ii, int jj,
   //       starpu_data_handle_t work_hdl,
   //       starpu_data_handle_t row_list_hdl,
   //       starpu_data_handle_t col_list_hdl,
   //       int blksz,
   //       int cptr, int cptr2,
   //       int rptr, int rptr2,
   //       int prio) {
         
   //    struct starpu_data_descr *descrs = 
   //       new starpu_data_descr[nblk_ik+nblk_jk+5];

   //    int ret;
   //    int nh = 0;
         
   //    // worksapce 
   //    descrs[nh].handle =  work_hdl; descrs[nh].mode = STARPU_SCRATCH;
   //    nh = nh + 1;
   //    // row_list
   //    descrs[nh].handle =  row_list_hdl; descrs[nh].mode = STARPU_SCRATCH;
   //    nh = nh + 1;
   //    // col_list
   //    descrs[nh].handle =  col_list_hdl; descrs[nh].mode = STARPU_SCRATCH;
   //    nh = nh + 1;
   //    // A_ij
   //    // printf("[insert_update_between] a_ij handle ptr: %p\n", bc_ij_hdl);
   //    descrs[nh].handle =  bc_ij_hdl; descrs[nh].mode = /* STARPU_RW;*/  (starpu_data_access_mode) (STARPU_RW | STARPU_COMMUTE);
   //    nh = nh + 1;
   //    // printf("[insert_update_between] nblk_ik: %d, nblk_jk: %d\n", nblk_ik, nblk_jk);
   //    // A_ik handles
   //    // DEBUG
   //    // printf("[insert_update_between] A_ik handle %d ptr: %p\n", 0, bc_ik_hdls[0]);
   //    // descrs[nh].handle = bc_ik_hdls[0];  descrs[nh].mode = STARPU_R;
   //    // nh = nh + 1;
   //    for(int i = 0; i < nblk_ik; i++) {
   //       // printf("[insert_update_between] a_ik handle %d, ptr: %p\n", i, bc_ik_hdls[i]);
   //       descrs[nh].handle = bc_ik_hdls[i];  descrs[nh].mode = STARPU_R;
   //       nh = nh + 1;
   //    }
   //    // A_jk handles
   //    for(int i = 0; i < nblk_jk; i++){
   //       // printf("[insert_update_between] a_jk handle %d, ptr: %p\n", i, bc_jk_hdls[i]);
   //       descrs[nh].handle = bc_jk_hdls[i];  descrs[nh].mode = STARPU_R;
   //       nh = nh + 1;
   //    }
         
   //    // Make sure that destination node as been initialized
   //    descrs[nh].handle =  anode_hdl; descrs[nh].mode = STARPU_R;
   //    nh = nh + 1;         

   //    ret = starpu_task_insert(&cl_update_between,
   //                             STARPU_DATA_MODE_ARRAY, descrs,   nh,
   //                             STARPU_VALUE, &snode, sizeof(SymbolicSNode*),
   //                             STARPU_VALUE, &node, sizeof(spldlt::NumericNode<T, PoolAlloc>*),
   //                             STARPU_VALUE, &kk, sizeof(int),
   //                             STARPU_VALUE, &asnode, sizeof(SymbolicSNode*),
   //                             STARPU_VALUE, &ii, sizeof(int),
   //                             STARPU_VALUE, &jj, sizeof(int),
   //                             STARPU_VALUE, &blksz, sizeof(int),
   //                             STARPU_VALUE, &cptr, sizeof(int),
   //                             STARPU_VALUE, &cptr2, sizeof(int),
   //                             STARPU_VALUE, &rptr, sizeof(int),
   //                             STARPU_VALUE, &rptr2, sizeof(int),
   //                             STARPU_PRIORITY, prio,
   //                             0);

   //    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

   //    delete[] descrs;
   // }

   ////////////////////////////////////////////////////////////////////////////////
   // Factor subtree task

   // CPU kernel

#if defined(SPLDLT_USE_OMP)
   template <typename T>
   void factor_subtree_cpu_func(void *buffers[], void *cl_arg) {

      // bool posdef;
      void *akeep;
      void *fkeep;
      int p;
      T *aval;
      T *scaling;
      void **child_contrib;
      sylver::options_t *options;
      std::vector<sylver::inform_t> *worker_stats;

      starpu_codelet_unpack_args(
            cl_arg,
            &akeep,
            &fkeep,
            &p,
            &aval,
            &scaling,
            &child_contrib,
            &options,
            &worker_stats);

      // printf("[factor_subtree_cpu_func] failed_pivot_method: %d\n",
      // options->failed_pivot_method);

      struct spral::ssids::cpu::cpu_factor_options subtree_opts;
      // Setup options for SSIDS
      options->copy(subtree_opts);
      // subtree_opts = *options;
      subtree_opts.cpu_block_size = 256; // TODO add new parameter for blksz in subtrees
      // printf("[factor_subtree_cpu_func] cpu_block_size = %d\n", subtree_opts.cpu_block_size);

      int workerid = starpu_worker_get_id();
      // printf("[factor_subtree_cpu_func] workerid: %d\n", workerid);
      sylver::inform_t& inform = (*worker_stats)[workerid];
      ThreadStats stats;

      // options->failed_pivot_method = FailedPivotMethod::tpp;
         
      // printf("[factor_subtree_cpu_func] akeep = %p, fkeep = %p\n", akeep, fkeep);
#pragma omp parallel default(shared)
      {
         int nth = 0;
         nth = omp_get_num_threads();
#pragma omp single
         {
            // printf("[factor_subtree_cpu_func] nth: %d\n", nth);
            factor_subtree(akeep, fkeep, p, aval, scaling, child_contrib, &subtree_opts, &stats);
         }
      }

      inform += stats;
   }

#else

   template <typename T>
   void factor_subtree_cpu_func(void *buffers[], void *cl_arg) {

      // bool posdef;
      void *akeep;
      void *fkeep;
      int p;
      T *aval;
      T *scaling;
      void **child_contrib;
      // struct spral::ssids::cpu::cpu_factor_options *options;
      sylver::options_t *options;
      // std::vector<ThreadStats> *worker_stats;
      std::vector<sylver::inform_t> *worker_stats;
      
      starpu_codelet_unpack_args(
            cl_arg,
            &akeep, 
            &fkeep,
            &p,
            &aval,
            &scaling,
            &child_contrib,
            &options,
            &worker_stats);

      // printf("[factor_subtree_cpu_func] failed_pivot_method: %d\n",
      // options->failed_pivot_method);

      int workerid = starpu_worker_get_id();
      // printf("[factor_subtree_cpu_func] workerid: %d\n", workerid);
      // ThreadStats& stats = (*worker_stats)[workerid];
      sylver::inform_t& stats = (*worker_stats)[workerid];
      
      // options->failed_pivot_method = FailedPivotMethod::tpp;

      struct spral::ssids::cpu::cpu_factor_options subtree_opts;
      // Setup options for SSIDS
      options->copy(subtree_opts);
      subtree_opts.cpu_block_size = 256; // TODO add new parameter for blksz in subtrees
      ThreadStats subtree_stats;
      // printf("[factor_subtree_cpu_func] akeep = %p, fkeep = %p\n", akeep, fkeep);
      // spldlt_factor_subtree_c(akeep, fkeep, p, aval, child_contrib, options, &stats);
      factor_subtree(akeep, fkeep, p, aval, scaling, child_contrib, &subtree_opts, &subtree_stats);
      stats += subtree_stats;

   }
#endif
   // factor_subtree StarPU codelet
   extern struct starpu_codelet cl_factor_subtree;
   // factor_subtree StarPU codelet
   extern struct starpu_codelet cl_factor_subtree_gpu;

   /// @brief Launch StarPU task for factorizing a subtree on a specific worker
   template <typename T>
   void insert_factor_subtree_worker(
         starpu_data_handle_t root_hdl, // Symbolic handle on root node
         void *akeep,
         void *fkeep,
         int p, // Subtree index
         T *aval,
         T *scaling,
         void **child_contrib,
         sylver::options_t *options,
         std::vector<sylver::inform_t> *worker_stats,
         int workerid // Worker index
         ) {

      int ret;

      enum starpu_worker_archtype archtype = starpu_worker_get_type(workerid);

      ret = starpu_task_insert(
            (archtype==STARPU_CUDA_WORKER) ? &cl_factor_subtree_gpu : &cl_factor_subtree,
            STARPU_EXECUTE_ON_WORKER, workerid, 
            STARPU_RW, root_hdl,
            STARPU_VALUE, &akeep, sizeof(void*),
            STARPU_VALUE, &fkeep, sizeof(void*),
            STARPU_VALUE, &p, sizeof(int),
            STARPU_VALUE, &aval, sizeof(T*),
            STARPU_VALUE, &scaling, sizeof(T*),
            STARPU_VALUE, &child_contrib, sizeof(void**),
            STARPU_VALUE, &options, sizeof(sylver::options_t*),
            STARPU_VALUE, &worker_stats, sizeof(std::vector<sylver::inform_t>*),
            0);
      
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");  
   }

   /// @brief Launch StarPU task for factorizing a subtree
   ///
   /// @param root_hdl
   template <typename T>
   void insert_factor_subtree(
         starpu_data_handle_t root_hdl, // Symbolic handle on root node
         void *akeep,
         void *fkeep,
         int p, // Subtree index
         T *aval,
         T *scaling,
         void **child_contrib,
         sylver::options_t *options,
         std::vector<sylver::inform_t> *worker_stats,
         int loc // Locality index
         ) {

      int ret;

      if (loc > 0) {
         ret = starpu_task_insert(
               &cl_factor_subtree,
               STARPU_RW, root_hdl,
               STARPU_VALUE, &akeep, sizeof(void*),
               STARPU_VALUE, &fkeep, sizeof(void*),
               STARPU_VALUE, &p, sizeof(int),
               STARPU_VALUE, &aval, sizeof(T*),
               STARPU_VALUE, &scaling, sizeof(T*),
               STARPU_VALUE, &child_contrib, sizeof(void**),
               STARPU_VALUE, &options, sizeof(sylver::options_t*),
               STARPU_VALUE, &worker_stats, sizeof(std::vector<sylver::inform_t>*),
               STARPU_SCHED_CTX, loc,
               0);

      }
      else {
         ret = starpu_task_insert(
               &cl_factor_subtree,
               STARPU_RW, root_hdl,
               STARPU_VALUE, &akeep, sizeof(void*),
               STARPU_VALUE, &fkeep, sizeof(void*),
               STARPU_VALUE, &p, sizeof(int),
               STARPU_VALUE, &aval, sizeof(T*),
               STARPU_VALUE, &scaling, sizeof(T*),
               STARPU_VALUE, &child_contrib, sizeof(void**),
               STARPU_VALUE, &options, sizeof(sylver::options_t*),
               STARPU_VALUE, &worker_stats, sizeof(std::vector<sylver::inform_t>*),
               0);
      }
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
   }

   // Get contrib task

   // extern "C" void spldlt_get_contrib_c(void *akeep, void *fkeep, int p, void **child_contrib);

   // // CPU kernel
   // void get_contrib_cpu_func(void *buffers[], void *cl_arg) {

   //    void *akeep;
   //    void *fkeep;
   //    int p;
   //    void **child_contrib;
         
   //    starpu_codelet_unpack_args(
   //          cl_arg,
   //          &akeep,
   //          &fkeep,
   //          &p,
   //          &child_contrib);

   //    // printf("[get_contrib_cpu_func] p = %d\n", p);

   //    spldlt_get_contrib_c(akeep, fkeep, p, child_contrib);
   // }

   // StarPU codelet
   // struct starpu_codelet cl_get_contrib;

   // void insert_get_contrib(
   //       starpu_data_handle_t root_hdl, // Symbolic handle on root node
   //       void *akeep, 
   //       void *fkeep,
   //       int p,
   //       void **child_contrib) {

   //    int ret;
         
   //    ret = starpu_task_insert(&cl_get_contrib,
   //                             STARPU_R, root_hdl,
   //                             STARPU_VALUE, &akeep, sizeof(void*),
   //                             STARPU_VALUE, &fkeep, sizeof(void*),
   //                             STARPU_VALUE, &p, sizeof(int),
   //                             STARPU_VALUE, &child_contrib, sizeof(void**),
   //                             0);

   //    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
   // }

   ////////////////////////////////////////////////////////////////////////////////
   // Assemble subtree task

   // CPU kernel
   template <typename T, typename PoolAlloc>
   void subtree_assemble_cpu_func(void *buffers[], void *cl_arg) {

      NumericFront<T, PoolAlloc> *node = nullptr;
      sylver::SymbolicFront *csnode;
      void **child_contrib;
      int contrib_idx;

      // printf("[subtree_assemble_cpu_func]\n");

      starpu_codelet_unpack_args(cl_arg,
                                 &node, &csnode,
                                 &child_contrib, &contrib_idx);

      assemble_subtree(*node, *csnode, child_contrib, contrib_idx);

   }

   // subtree_assemble StarPU codelet
   extern struct starpu_codelet cl_subtree_assemble;

   template <typename T, typename PoolAlloc>
   void insert_subtree_assemble(
         spldlt::NumericFront<T, PoolAlloc> *node,
         sylver::SymbolicFront *csnode,
         starpu_data_handle_t node_hdl,
         starpu_data_handle_t root_hdl,
         starpu_data_handle_t *dest_hdls, int ndest,
         void **child_contrib, int contrib_idx
         ) {

      int ret;
      int nh = 0;
         
      struct starpu_data_descr *descrs = new starpu_data_descr[ndest+2];

      for (int i=0; i<ndest; i++) {
         descrs[nh].handle = dest_hdls[i]; descrs[nh].mode = STARPU_RW;
         nh++;
      }

      // Handle on subtree
      descrs[nh].handle = root_hdl; descrs[nh].mode = STARPU_R;
      nh++;

      // // Handle on node to be assembled
      // descrs[nh].handle = node_hdl; descrs[nh].mode = STARPU_R;
      // nh++;

      ret = starpu_task_insert(&cl_subtree_assemble,
                               STARPU_DATA_MODE_ARRAY, descrs, nh,
                               STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
                               STARPU_VALUE, &csnode, sizeof(sylver::SymbolicFront*),
                               STARPU_VALUE, &child_contrib, sizeof(void**),
                               STARPU_VALUE, &contrib_idx, sizeof(int),
                               0);
       
      delete[] descrs;
   }

   ////////////////////////////////////////////////////////////
   // Subtree assemble contrib task

   // CPU kernel
   template <typename T, typename PoolAlloc>
   void subtree_assemble_contrib_cpu_func(void *buffers[], void *cl_arg) {

      NumericFront<T, PoolAlloc> *node = nullptr;
      sylver::SymbolicFront *csnode = nullptr;
      void **child_contrib;
      int contrib_idx;

      // printf("[subtree_assemble_contrib_cpu_func]\n");

      starpu_codelet_unpack_args(cl_arg,
                                 &node, &csnode,
                                 &child_contrib, &contrib_idx);

      assemble_contrib_subtree(*node, *csnode, child_contrib,contrib_idx);
   }

   // StarPU codelet
   extern struct starpu_codelet cl_subtree_assemble_contrib;

   template <typename T, typename PoolAlloc>
   void insert_subtree_assemble_contrib(
         NumericFront<T, PoolAlloc> *node,
         sylver::SymbolicFront *csnode,
         starpu_data_handle_t node_hdl,
         starpu_data_handle_t contrib_hdl,
         starpu_data_handle_t root_hdl,
         starpu_data_handle_t *dest_hdls, int ndest,
         void **child_contrib, int contrib_idx,
         int prio) {

      int ret;
      int nh = 0;
         
      struct starpu_data_descr *descrs = new starpu_data_descr[ndest+3];

      for (int i=0; i<ndest; i++) {
         descrs[nh].handle = dest_hdls[i]; descrs[nh].mode = (starpu_data_access_mode) (STARPU_RW | STARPU_COMMUTE);
         nh++;
      }

      // Handle on subtree
      descrs[nh].handle = root_hdl; descrs[nh].mode = STARPU_R;
      nh++;

      // Handle on node to be assembled
      // descrs[nh].handle = node_hdl; descrs[nh].mode = STARPU_R;
      // nh++;

      // Handle on contrib blocks
      descrs[nh].handle = contrib_hdl; descrs[nh].mode = STARPU_R;
      nh++;

      ret = starpu_task_insert(&cl_subtree_assemble_contrib,
                               STARPU_DATA_MODE_ARRAY, descrs, nh,
                               STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
                               STARPU_VALUE, &csnode, sizeof(sylver::SymbolicFront*),
                               STARPU_VALUE, &child_contrib, sizeof(void**),
                               STARPU_VALUE, &contrib_idx, sizeof(int),
                               STARPU_PRIORITY, prio,
                               0);

      delete[] descrs;
   }


   ////////////////////////////////////////////////////////////////////////////////
   // Assemble block task
      
   // CPU kernel
   template <typename T, typename PoolAlloc>
   void assemble_block_cpu_func(void *buffers[], void *cl_arg) {

      // typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;
      // std::vector<int, PoolAllocInt> *map;

      NumericFront<T, PoolAlloc> *node = nullptr, *cnode = nullptr;
      int ii, jj; // Block indexes
      int *map;

      starpu_codelet_unpack_args(cl_arg, 
                                 &node, &cnode,
                                 &ii, &jj, &map);

      assemble_block(*node, *cnode, ii, jj, map);
   }

   // assemble_block StarPU codelet
   extern struct starpu_codelet cl_assemble_block;

   template <typename T, typename PoolAlloc>
   void insert_assemble_block(
         NumericFront<T, PoolAlloc> *node,
         NumericFront<T, PoolAlloc> const* cnode,
         int ii, int jj,
         int *cmap,
         starpu_data_handle_t bc_hdl,
         starpu_data_handle_t *dest_hdls, int ndest,
         starpu_data_handle_t node_hdl, // Symbolic handle for destination node
         starpu_data_handle_t cnode_hdl,
         int prio) {

      assert(ndest > 0);

      int ret;
      int nh = 0;
         
      struct starpu_data_descr *descrs = new starpu_data_descr[ndest+2];

      descrs[nh].handle = bc_hdl; descrs[nh].mode = STARPU_R;
      nh++;
         
      for (int i=0; i<ndest; i++) {
         descrs[nh].handle = dest_hdls[i]; descrs[nh].mode = STARPU_RW;
         nh++;
      }

      // printf("[insert_assemble_block] node_hdl: %p\n", node_hdl);

      // Access symbolic handle of node in read mode to ensure that
      // it has been initialized
      // descrs[nh].handle = node_hdl; descrs[nh].mode = STARPU_R;
      // nh++;

      // Access symbolic handle of child node in read mode to
      // ensure that assemblies are done before cleaning it
      descrs[nh].handle = cnode_hdl; descrs[nh].mode = STARPU_R;
      nh++;

      ret = starpu_task_insert(&cl_assemble_block,
                               STARPU_DATA_MODE_ARRAY, descrs, nh,
                               STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
                               STARPU_VALUE, &cnode, sizeof(NumericFront<T, PoolAlloc>*),
                               STARPU_VALUE, &ii, sizeof(int),
                               STARPU_VALUE, &jj, sizeof(int),
                               STARPU_VALUE, &cmap, sizeof(int*),
                               STARPU_PRIORITY, prio,
                               0);
      delete[] descrs;

   }

   ////////////////////////////////////////////////////////////////////////////////
   // Assemble contrib block task

   // CPU kernel
   template <typename T, typename PoolAlloc>
   void assemble_contrib_block_cpu_func(void *buffers[], void *cl_arg) {

      // typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;
      // std::vector<int, PoolAllocInt> *map;

      NumericFront<T, PoolAlloc> *node = nullptr, *cnode = nullptr;
      int ii, jj; // Block row and col indexes
      int *map;
      std::vector<spral::ssids::cpu::Workspace> *workspaces;
         
      starpu_codelet_unpack_args(
            cl_arg, &node, &cnode,
            &ii, &jj, &map, &workspaces);

      // printf("[assemble_contrib_block_cpu_func]\n");

      // int workerid = starpu_worker_get_id();
      // spral::ssids::cpu::Workspace &work = (*workspaces)[workerid];

#if defined(MEMLAYOUT_1D)
      assemble_contrib_block_1d(*node, *cnode, ii, jj, map);
#else
      assemble_contrib_block(*node, *cnode, ii, jj, map);
#endif
   }

   // assemble_contrib_block StarPU codelet
   extern struct starpu_codelet cl_assemble_contrib_block;

   /// @brief Launch StarPU task for assembling block (ii,jj) in
   /// cnode into node
   ///
   /// @param node Destination node
   /// @param cnode Source node holding block (ii,jj)
   /// @param cmap Mapping vector: i-th column in cnode must be
   /// assembled in cmap(i) column of destination node
   template <typename T, typename PoolAlloc>
   void insert_assemble_contrib_block(
         NumericFront<T, PoolAlloc> *node, // Destinaton node
         NumericFront<T, PoolAlloc> *cnode,// Source node
         int ii, int jj,
         int *cmap,
         starpu_data_handle_t bc_hdl,
         starpu_data_handle_t *dest_hdls, int ndest,
         starpu_data_handle_t node_hdl, // Symbolic handle of destination node
         starpu_data_handle_t contrib_hdl,
         starpu_data_handle_t cnode_hdl, // Symbolic handle of source node
         std::vector<spral::ssids::cpu::Workspace> *workspaces,
         int prio) {

      assert(ndest > 0);

      int ret;
      int nh = 0;
         
      struct starpu_data_descr *descrs = new starpu_data_descr[ndest+3];

      descrs[nh].handle = bc_hdl; descrs[nh].mode = STARPU_R;
      nh++;
         
      for (int i=0; i<ndest; i++) {
         descrs[nh].handle = dest_hdls[i]; descrs[nh].mode = STARPU_RW;
         nh++;
      }

      // Access symbolic handle of node in read mode to ensure that
      // it has been initialized
      // descrs[nh].handle = node_hdl; descrs[nh].mode = STARPU_R;
      // nh++;

      // Access symbolic handle of child node in read mode to
      // ensure that assemblies are done before cleaning it
      descrs[nh].handle = cnode_hdl; descrs[nh].mode = STARPU_R;
      nh++;

      // Handle on contrib blocks
      descrs[nh].handle = contrib_hdl; descrs[nh].mode = STARPU_R;
      nh++;
         
#if defined(SPLDLT_USE_PROFILING)
      // Compute flops
      Tile<T, PoolAlloc>& cblock = cnode->get_contrib_block(ii, jj);
      int blk_m = cblock.m;
      int blk_n = cblock.n;
      double flops = static_cast<double>(blk_m)*static_cast<double>(blk_n);
      if (ii==jj) 
         flops -= (static_cast<double>(blk_n)*
                   (static_cast<double>(blk_n)-1.0))/2.0; // Remove flops for coef above diagonal

      // printf("[insert_assemble_contrib_block] flops = %.2f\n", flops);
#endif
            
      ret = starpu_task_insert(
            &cl_assemble_contrib_block,
            STARPU_DATA_MODE_ARRAY, descrs, nh,
            STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
            STARPU_VALUE, &cnode, sizeof(NumericFront<T, PoolAlloc>*),
            STARPU_VALUE, &ii, sizeof(int),
            STARPU_VALUE, &jj, sizeof(int),
            STARPU_VALUE, &cmap, sizeof(int*),
            STARPU_VALUE, &workspaces, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
            STARPU_PRIORITY, prio,
#if defined(SPLDLT_USE_PROFILING)
            STARPU_FLOPS, flops,
#endif
            0);

      delete[] descrs;

   }

   ////////////////////////////////////////////////////////////////////////////////
   // Activate node

   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void activate_node_cpu_func(void *buffers[], void *cl_arg) {
         
      bool posdef;
      sylver::SymbolicFront *snode;
      NumericFront<T, PoolAlloc> *node;
      void** child_contrib;
      int blksz;
      FactorAlloc *factor_alloc;

      // printf("[activate_node_cpu_func]\n");

      starpu_codelet_unpack_args(
            cl_arg, &posdef, &snode, &node, &child_contrib, &blksz,
            &factor_alloc);

      activate_front(posdef, *node, child_contrib, *factor_alloc);
         
   }

   // activate_node StarPU codelet
   extern struct starpu_codelet cl_activate_node;

   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void insert_activate_node(
         starpu_data_handle_t node_hdl, // Node's symbolic handle
         starpu_data_handle_t *cnode_hdls, int nhdl, // Children node's symbolic handles
         bool posdef,
         sylver::SymbolicFront *snode,
         NumericFront<T, PoolAlloc> *node,
         void** child_contrib,
         int blksz,
         FactorAlloc *factor_alloc
         ) {


      struct starpu_data_descr *descrs = new starpu_data_descr[nhdl+1];

      int nh = 0;
      descrs[nh].handle = node_hdl; descrs[nh].mode = STARPU_RW;
      nh++;

      for (int i=0; i<nhdl; i++) {
         descrs[nh].handle = cnode_hdls[i]; descrs[nh].mode = STARPU_R;
         nh++;
      }

      int ret;
      ret = starpu_task_insert(&cl_activate_node,
                               STARPU_DATA_MODE_ARRAY, descrs, nh,
                               STARPU_VALUE, &posdef, sizeof(bool),
                               STARPU_VALUE, &snode, sizeof(sylver::SymbolicFront*),
                               STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
                               STARPU_VALUE, &child_contrib, sizeof(void**),
                               STARPU_VALUE, &blksz, sizeof(int),
                               STARPU_VALUE, &factor_alloc, sizeof(FactorAlloc*),
                               0);

      delete[] descrs;
   }

   ////////////////////////////////////////////////////////////
   // Activate and init node

   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void activate_init_node_cpu_func(void *buffers[], void *cl_arg) {
         
      bool posdef;
      NumericFront<T, PoolAlloc> *node;
      void** child_contrib;
      FactorAlloc *factor_alloc;
      PoolAlloc *pool_alloc;
      T *aval = nullptr;
      T *scaling = nullptr;

      starpu_codelet_unpack_args(
            cl_arg, &posdef, &node, &child_contrib, 
            &factor_alloc, &pool_alloc, &aval, &scaling);
         
      // Allocate data structures
      activate_front(
            posdef, *node, child_contrib, *factor_alloc);
      
      // Add coefficients from original matrix
      init_node(*node, aval, scaling);
   }

   // activate_init_node StarPU codelet
   extern struct starpu_codelet cl_activate_init_node;

   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void insert_activate_init_node(
         starpu_data_handle_t node_hdl, // Node's symbolic handle
         starpu_data_handle_t *cnode_hdls, int nhdl, // Children node's symbolic handles
         bool posdef,
         NumericFront<T, PoolAlloc> *node,
         void** child_contrib,
         FactorAlloc *factor_alloc,
         PoolAlloc *pool_alloc,
         T *aval, T *scaling
         ) {

      struct starpu_data_descr *descrs = new starpu_data_descr[nhdl+1];

      int nh = 0;
      descrs[nh].handle = node_hdl; descrs[nh].mode = STARPU_RW;
      nh++;

      for (int i=0; i<nhdl; i++) {
         descrs[nh].handle = cnode_hdls[i]; descrs[nh].mode = STARPU_R;
         nh++;
      }
         
      int ret;
      ret = starpu_task_insert(&cl_activate_init_node,
                               STARPU_DATA_MODE_ARRAY, descrs, nh,
                               STARPU_VALUE, &posdef, sizeof(bool),
                               STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
                               STARPU_VALUE, &child_contrib, sizeof(void**),
                               STARPU_VALUE, &factor_alloc, sizeof(FactorAlloc*),
                               STARPU_VALUE, &pool_alloc, sizeof(PoolAlloc*),
                               STARPU_VALUE, &aval, sizeof(T*),
                               STARPU_VALUE, &scaling, sizeof(T*),
                               0);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

      delete[] descrs;
   }

}} // namespaces spldlt::starpu
