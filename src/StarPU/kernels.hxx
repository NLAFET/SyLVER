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

namespace sylver {
namespace spldlt {
namespace starpu {

   // unregister handles for a node in StarPU
   template <typename NumericFrontType>
   void unregister_node_submit(
         NumericFrontType& node) {

      // printf("[unregister_node_submit]\n");
         
      // Get node info
      sylver::SymbolicFront const& snode = node.symb();
      int const blksz = node.blksz();
      int const m = node.nrow();
      int const n = node.ncol();
      int const nr = node.nr(); // Number of block rows
      int const nc = node.nc(); // Number of block columns

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
   template <typename NumericFrontType>
   void init_node_cpu_func(void *buffers[], void *cl_arg) {

      using ValueType = typename NumericFrontType::ValueType;
      
      NumericFrontType* front = nullptr;
      ValueType* aval;
      ValueType* scaling;

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
   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void insert_init_node(
         NumericFront<T, FactorAlloc, PoolAlloc> *front,
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
            STARPU_VALUE, &front, sizeof(NumericFront<T, FactorAlloc, PoolAlloc>*),
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
   template <typename NumericFrontType>
   void fini_node_cpu_func(void *buffers[], void *cl_arg) {
         
      NumericFrontType* node = nullptr;
      bool posdef;
         
      starpu_codelet_unpack_args(cl_arg, &node, &posdef);

      // printf("[fini_node_cpu_func] node idx = %d\n", node->symb.idx+1);
      if (posdef) {
         unregister_node_posdef(*node);
      }
      else {
         unregister_node_indef(*node);
      }
         
      fini_node(*node);
   }

   // fini_node codelet
   extern struct starpu_codelet cl_fini_node;
      
   template <typename NumericFrontType>
   void insert_fini_node(
         starpu_data_handle_t node_hdl,
         starpu_data_handle_t *hdls, int nhdl, // Children node's symbolic handles
         NumericFrontType* node,
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
            STARPU_VALUE, &node, sizeof(NumericFrontType*),
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
   // Assemble block task
      
   // CPU kernel
   template <typename NumericFrontType>
   void assemble_block_cpu_func(void *buffers[], void *cl_arg) {

      // typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;
      // std::vector<int, PoolAllocInt> *map;

      NumericFrontType* node = nullptr;
      NumericFrontType* cnode = nullptr;
      int ii, jj; // Block indexes
      int *map;

      starpu_codelet_unpack_args(cl_arg, 
                                 &node, &cnode,
                                 &ii, &jj, &map);

      assemble_block(*node, *cnode, ii, jj, map);
   }

   // assemble_block StarPU codelet
   extern struct starpu_codelet cl_assemble_block;

   template <typename NumericFrontType>
   void insert_assemble_block(
         NumericFrontType* node,
         NumericFrontType const* cnode,
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
                               STARPU_VALUE, &node, sizeof(NumericFrontType*),
                               STARPU_VALUE, &cnode, sizeof(NumericFrontType*),
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
   template <typename NumericFrontType>
   void assemble_contrib_block_cpu_func(void *buffers[], void *cl_arg) {

      // typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;
      // std::vector<int, PoolAllocInt> *map;

      NumericFrontType* node = nullptr;
      NumericFrontType* cnode = nullptr;
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
   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void insert_assemble_contrib_block(
         NumericFront<T, FactorAlloc, PoolAlloc> *node, // Destinaton node
         NumericFront<T, FactorAlloc, PoolAlloc> *cnode,// Source node
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
            STARPU_VALUE, &node, sizeof(NumericFront<T, FactorAlloc, PoolAlloc>*),
            STARPU_VALUE, &cnode, sizeof(NumericFront<T, FactorAlloc, PoolAlloc>*),
            STARPU_VALUE, &ii, sizeof(int),
            STARPU_VALUE, &jj, sizeof(int),
            STARPU_VALUE, &cmap, sizeof(int*),
            STARPU_VALUE, &workspaces, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
            STARPU_PRIORITY, prio,
#if defined(SPLDLT_USE_PROFILING)
            STARPU_FLOPS, flops,
#endif
            0);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

      
      delete[] descrs;

   }

   ////////////////////////////////////////////////////////////////////////////////
   // Activate node

   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void activate_node_cpu_func(void *buffers[], void *cl_arg) {
         
      bool posdef;
      sylver::SymbolicFront* snode;
      NumericFront<T, FactorAlloc, PoolAlloc>* node;
      void** child_contrib;
      int blksz;
      FactorAlloc *factor_alloc;

      // printf("[activate_node_cpu_func]\n");

      starpu_codelet_unpack_args(
            cl_arg, &posdef, &snode, &node, &child_contrib, &blksz,
            &factor_alloc);

      if (posdef) {
         node->activate_posdef();
      }
      else {
         node->activate(child_contrib);
      }
         
   }

   // activate_node StarPU codelet
   extern struct starpu_codelet cl_activate_node;

   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void insert_activate_node(
         starpu_data_handle_t node_hdl, // Node's symbolic handle
         starpu_data_handle_t *cnode_hdls, int nhdl, // Children node's symbolic handles
         bool posdef,
         sylver::SymbolicFront *snode,
         NumericFront<T, FactorAlloc, PoolAlloc> *node,
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
                               STARPU_VALUE, &node, sizeof(NumericFront<T, FactorAlloc, PoolAlloc>*),
                               STARPU_VALUE, &child_contrib, sizeof(void**),
                               STARPU_VALUE, &blksz, sizeof(int),
                               STARPU_VALUE, &factor_alloc, sizeof(FactorAlloc*),
                               0);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

      delete[] descrs;
   }

   ////////////////////////////////////////////////////////////
   // Activate and init node

   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void activate_init_node_cpu_func(void *buffers[], void *cl_arg) {
         
      bool posdef;
      NumericFront<T, FactorAlloc, PoolAlloc> *node;
      void** child_contrib;
      FactorAlloc *factor_alloc;
      PoolAlloc *pool_alloc;
      T *aval = nullptr;
      T *scaling = nullptr;

      starpu_codelet_unpack_args(
            cl_arg, &posdef, &node, &child_contrib, 
            &factor_alloc, &pool_alloc, &aval, &scaling);
         
      // Allocate data structures
      if (posdef) {
         node->activate_posdef();
      }
      else {
         node->activate(child_contrib);
      }
      
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
         NumericFront<T, FactorAlloc, PoolAlloc> *node,
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
                               STARPU_VALUE, &node, sizeof(NumericFront<T, FactorAlloc, PoolAlloc>*),
                               STARPU_VALUE, &child_contrib, sizeof(void**),
                               STARPU_VALUE, &factor_alloc, sizeof(FactorAlloc*),
                               STARPU_VALUE, &pool_alloc, sizeof(PoolAlloc*),
                               STARPU_VALUE, &aval, sizeof(T*),
                               STARPU_VALUE, &scaling, sizeof(T*),
                               0);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

      delete[] descrs;
   }

}}} // namespaces sylver::spldlt::starpu
