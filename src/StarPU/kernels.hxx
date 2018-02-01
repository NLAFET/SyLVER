#pragma once

// #include <vector>
// SpLDLT
// #include "SymbolicSNode.hxx"
// #include "NumericNode.hxx"
// #include "kernels/factor.hxx"
// #include "kernels/assemble.hxx"

#include <starpu.h>

namespace spldlt { namespace starpu {

      // // Register handles for a node in StarPU
      // template <typename T, typename PoolAlloc>
      // void register_node(
      //       SymbolicSNode &snode,
      //       spldlt::NumericNode<T, PoolAlloc> &node,
      //       int blksz
      //       ) {

      //    int m = snode.nrow;
      //    int n = snode.ncol;
      //    T *a = node.lcol;
      //    int lda = align_lda<T>(m);
      //    int nr = (m-1) / blksz + 1; // number of block rows
      //    int nc = (n-1) / blksz + 1; // number of block columns
      //    // snode.handles.reserve(nr*nc);
      //    snode.handles.resize(nr*nc); // allocate handles
      //    // printf("[register_node] nr: %d\n", nr);
      //    for(int j = 0; j < nc; ++j) {
               
      //       int blkn = std::min(blksz, n - j*blksz);
               
      //       for(int i = j; i < nr; ++i) {

      //          int blkm = std::min(blksz, m - i*blksz);

      //          starpu_matrix_data_register(
      //                &(snode.handles[i + j*nr]), // StarPU handle ptr 
      //                STARPU_MAIN_RAM, // memory 
      //                reinterpret_cast<uintptr_t>(&a[(j*blksz)*lda+(i*blksz)]),
      //                lda, blkm, blkn,
      //                sizeof(T));
      //       }
      //    }

      //    int ldcontrib = m-n;
         
      //    // Allocate and init handles in contribution blocks         
      //    if (ldcontrib>0) {
      //       // Index of first block in contrib
      //       int rsa = n/blksz;
      //       // Number of block in contrib
      //       int ncontrib = nr-rsa;

      //       for(int j = rsa; j < nr; j++) {
      //          for(int i = j; i < nr; i++) {
      //             // Register block in StarPU
      //             node.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].register_handle();
      //          }
      //       }
      //    }

      //    // T *contrib = node.contrib;

      //    // // Allocate and init handles in contribution blocks         
      //    // if (contrib) {
      //    //    // Index of first block in contrib
      //    //    int rsa = n/blksz;
      //    //    // Number of block in contrib
      //    //    int ncontrib = nr-rsa;
      //    //    snode.contrib_handles.resize(ncontrib*ncontrib);

      //    //    for(int j = rsa; j < nr; j++) {
      //    //       // First col in contrib block
      //    //       int first_col = std::max(j*blksz, n);
      //    //       // Block width
      //    //       int blkn = std::min((j+1)*blksz, m) - first_col;

      //    //       for(int i = j; i < nr; i++) {
      //    //          // First col in contrib block
      //    //          int first_row = std::max(i*blksz, n);
      //    //          // Block height
      //    //          int blkm = std::min((i+1)*blksz, m) - first_row;

      //    //          // starpu_matrix_data_register(
      //    //          //       &(snode.contrib_handles[(i-rsa)+(j-rsa)*ncontrib]), // StarPU handle ptr
      //    //          //       STARPU_MAIN_RAM, // memory 
      //    //          //       reinterpret_cast<uintptr_t>(&contrib[(first_col-n)*ldcontrib+(first_row-n)]),
      //    //          //       ldcontrib, blkm, blkn, sizeof(T));
                  
      //    //          node.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].register_handle();
      //    //       }
      //    //    }

      //    // }
      // }

      // unregister handles for a node in StarPU
      // template <typename T, typename PoolAlloc>
      // void unregister_node_submit(
      //       SymbolicSNode &snode,
      //       spldlt::NumericNode<T, PoolAlloc> &node,
      //       int blksz
      //       ) {

      //    // Get node info
      //    int m = snode.nrow;
      //    int n = snode.ncol;
      //    int nr = (m-1) / blksz + 1; // number of block rows
      //    int nc = (n-1) / blksz + 1; // number of block columns

      //    // Unregister block handles in the factors
      //    for(int j = 0; j < nc; ++j) {
      //       for(int i = j; i < nr; ++i) {
      //          starpu_data_unregister_submit(snode.handles[i + j*nr]);
      //       }
      //    }

      //    // Unregister block handles in the contribution blocks
      //    int ldcontrib = m-n;

      //    if (ldcontrib>0) {
      //       // Index of first block in contrib
      //       int rsa = n/blksz;
      //       // Number of block in contrib
      //       int ncontrib = nr-rsa;

      //       for(int j = rsa; j < nr; j++) {
      //          for(int i = j; i < nr; i++) {
      //             // Register block in StarPU
      //             node.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].unregister_handle_submit();
      //          }
      //       }

      //    }
      // }

      /* Unregister handles in StarPU*/
      // void unregister
      
      /* init_node StarPU task*/
      /* CPU task */
      // template <typename T, typename PoolAlloc>
      // void init_node_cpu_func(void *buffers[], void *cl_arg) {

      //    SymbolicSNode *snode = nullptr;
      //    spldlt::NumericNode<T, PoolAlloc> *node = nullptr;
      //    T *aval;

      //    starpu_codelet_unpack_args(
      //          cl_arg,
      //          &snode,
      //          &node,
      //          &aval);

      //    init_node(*snode, *node, aval);
      // }

      // init_node codelet
      // struct starpu_codelet cl_init_node;      

      // template <typename T, typename PoolAlloc>
      // void insert_init_node(
      //       SymbolicSNode *snode,
      //       spldlt::NumericNode<T, PoolAlloc> *node,
      //       starpu_data_handle_t node_hdl,
      //       T *aval, int prio) {
                  
      //    int ret;

      //    ret = starpu_insert_task(
      //          &cl_init_node,
      //          STARPU_RW, node_hdl,
      //          STARPU_VALUE, &snode, sizeof(SymbolicSNode*),
      //          STARPU_VALUE, &node, sizeof(spldlt::NumericNode<T, PoolAlloc>*),
      //          STARPU_VALUE, &aval, sizeof(T*),
      //          STARPU_PRIORITY, prio,
      //          0);

      //    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

      // }

      // fini_node StarPU kernels
      // fini_node CPU kernel
      // template <typename T, typename PoolAlloc>
      // void fini_node_cpu_func(void *buffers[], void *cl_arg) {
         
      //    spldlt::NumericNode<T, PoolAlloc> *node = nullptr;
         
      //    starpu_codelet_unpack_args(cl_arg, &node);

      //    fini_node(*node);
      // }

      // fini_node codelet
      // struct starpu_codelet cl_fini_node;
      
      // template <typename T, typename PoolAlloc>
      // void insert_fini_node(spldlt::NumericNode<T, PoolAlloc> *node,
      //                       starpu_data_handle_t node_hdl, 
      //                       int prio) {

      //    int ret;
         
      //    ret = starpu_insert_task(
      //          &cl_fini_node,
      //          STARPU_RW, node_hdl,
      //          STARPU_VALUE, &node, sizeof(spldlt::NumericNode<T, PoolAlloc>*),
      //          STARPU_PRIORITY, prio,
      //          0);
      // }

      /* factorize_block StarPU task */

      /* CPU task */
      /* TODO generic prec */
      // void factorize_block_cpu_func(void *buffers[], void *cl_arg) {
         
      //    double *blk = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
      //    unsigned m = STARPU_MATRIX_GET_NX(buffers[0]);
      //    unsigned n = STARPU_MATRIX_GET_NY(buffers[0]);
      //    unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);

      //    factorize_diag_block(m, n, blk, ld);
      // }

      /* TODO generic prec */
      // void factorize_contrib_block_cpu_func(void *buffers[], void *cl_arg) {
         
      //    double *blk = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
      //    unsigned m = STARPU_MATRIX_GET_NX(buffers[0]);
      //    unsigned n = STARPU_MATRIX_GET_NY(buffers[0]);
      //    unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);

      //    double *contrib = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
      //    unsigned mcontrib = STARPU_MATRIX_GET_NX(buffers[1]);
      //    unsigned ncontrib = STARPU_MATRIX_GET_NY(buffers[1]);
      //    unsigned ldcontrib = STARPU_MATRIX_GET_LD(buffers[1]);

      //    int k;

      //    starpu_codelet_unpack_args(
      //          cl_arg,
      //          &k);

      //    // printf("[factorize_contrib_block_cpu_func] kk: %d\n", kk);
      //    // printf("[factorize_contrib_block_cpu_func] contrib: %p, ldcontrib: %d\n", 
      //    // contrib, ldcontrib);
      //    // printf("[factorize_contrib_block_cpu_func] mcontrib: %d, ncontrib: %d, ldcontrib: %d\n", mcontrib, ncontrib, ldcontrib);

      //    factorize_diag_block(m, n, blk, ld, contrib, ldcontrib,
      //                         k==0);
      // }
      
      /* FIXME: although it would be better to statically initialize
         the codelet, it is not well suported by g++ */

      // struct starpu_codelet cl_factorize_block = {
      //    // .where = w,
      //    // .cpu_funcs = {factorize_block_cpu_func, NULL},
      //    .nbuffers = STARPU_VARIABLE_NBUFFERS,
      //    // .name = "FACTO_BLK"
      // };

      /* factorize block codelet */
      // struct starpu_codelet cl_factorize_contrib_block;

      // void insert_factorize_block(
      //       int k,
      //       starpu_data_handle_t bc_hdl,
      //       starpu_data_handle_t contrib_hdl,
      //       starpu_data_handle_t node_hdl, // Symbolic node handle
      //       int prio) {

      //    int ret;

      //    if (node_hdl) {
      //       ret = starpu_insert_task(
      //             &cl_factorize_contrib_block,
      //             STARPU_VALUE, &k, sizeof(int),
      //             STARPU_RW, bc_hdl,
      //             STARPU_RW, contrib_hdl,
      //             STARPU_R, node_hdl,
      //             STARPU_PRIORITY, prio,
      //             0);
      //    }
      //    else {
      //       ret = starpu_insert_task(
      //             &cl_factorize_contrib_block,
      //             STARPU_VALUE, &k, sizeof(int),
      //             STARPU_RW, bc_hdl,
      //             STARPU_RW, contrib_hdl,
      //             STARPU_PRIORITY, prio,
      //             0);
      //    }

      //    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
      // }

      /* factorize block codelet */
      // struct starpu_codelet cl_factorize_block;      

      // void insert_factorize_block(
      //       starpu_data_handle_t bc_hdl,
      //       starpu_data_handle_t node_hdl, // Symbolic node handle
      //       int prio) {
                  
      //    int ret;

      //    if (node_hdl) {
      //       ret = starpu_insert_task(
      //             &cl_factorize_block,
      //             STARPU_RW, bc_hdl,
      //             STARPU_R, node_hdl,
      //             STARPU_PRIORITY, prio,
      //             0);
      //    }
      //    else {
      //       ret = starpu_insert_task(
      //             &cl_factorize_block,
      //             STARPU_RW, bc_hdl,
      //             STARPU_PRIORITY, prio,
      //             0);
      //    }

      //    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
      // }

      // solve_block StarPU task

      // CPU task
      // void solve_block_cpu_func(void *buffers[], void *cl_arg) {

      //    /* Get diag block pointer and info */
      //    double *blk = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
      //    unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);

      //    /* Get sub diag block pointer and info */
      //    double *blk_ik = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
      //    unsigned m = STARPU_MATRIX_GET_NX(buffers[1]);
      //    unsigned n = STARPU_MATRIX_GET_NY(buffers[1]);
      //    unsigned ld_ik = STARPU_MATRIX_GET_LD(buffers[1]);

      //    /* Call kernel function */
      //    solve_block(m, n, blk, ld, blk_ik, ld_ik);
      // }

      /* solve_block codelet */
      // struct starpu_codelet cl_solve_block;

      // /* Insert solve of subdiag block into StarPU */
      // void insert_solve_block(
      //       starpu_data_handle_t bc_kk_hdl, /* diag block handle */
      //       starpu_data_handle_t bc_ik_hdl, /* sub diag block handle */
      //       starpu_data_handle_t node_hdl,
      //       int prio
      //       ) {

      //    int ret;

      //    ret = starpu_insert_task(
      //          &cl_solve_block,
      //          STARPU_R, bc_kk_hdl,
      //          STARPU_RW, bc_ik_hdl,
      //          STARPU_R, node_hdl,
      //          STARPU_PRIORITY, prio,
      //          0);

      //    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
      // }

      // CPU task
      // void solve_contrib_block_cpu_func(void *buffers[], void *cl_arg) {

      //    // Get diag block pointer and info
      //    double *blk = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
      //    unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);

      //    // Get sub diag block pointer and info
      //    double *blk_ik = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
      //    unsigned m = STARPU_MATRIX_GET_NX(buffers[1]);
      //    unsigned n = STARPU_MATRIX_GET_NY(buffers[1]);
      //    unsigned ld_ik = STARPU_MATRIX_GET_LD(buffers[1]);

      //    double *contrib = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
      //    unsigned ldcontrib = STARPU_MATRIX_GET_LD(buffers[2]);

      //    int k, nb;

      //    starpu_codelet_unpack_args(
      //          cl_arg, &k, &nb);

      //    solve_block(m, n, blk, ld, blk_ik, ld_ik,
      //                contrib, ldcontrib, k==0, nb);
      // }

      // struct starpu_codelet cl_solve_contrib_block;

      /* Insert solve of subdiag block into StarPU */
      // void insert_solve_block(
      //       int k, int nb,
      //       starpu_data_handle_t bc_kk_hdl, // Diag block handle
      //       starpu_data_handle_t bc_ik_hdl, // Sub diag block handle
      //       starpu_data_handle_t contrib_hdl, // Contrib block handle
      //       starpu_data_handle_t node_hdl,
      //       int prio) {

      //    int ret;

      //    ret = starpu_insert_task(
      //          &cl_solve_contrib_block,
      //          STARPU_R, bc_kk_hdl,
      //          STARPU_RW, bc_ik_hdl,
      //          STARPU_RW, contrib_hdl,
      //          STARPU_R, node_hdl,
      //          STARPU_PRIORITY, prio,
      //          STARPU_VALUE, &k, sizeof(int),
      //          STARPU_VALUE, &nb, sizeof(int),
      //          0);

      //    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
      // }      

      /* update_block StarPU task */

      /* CPU task */
      // void update_block_cpu_func(void *buffers[], void *cl_arg) {

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
        
      //    update_block(m, n, blk_ij, ld_ij,
      //                 k,
      //                 blk_ik, ld_ik, 
      //                 blk_jk, ld_jk);
      // }

      // // update_block codelet
      // struct starpu_codelet cl_update_block;      

      // void insert_update_block(
      //       starpu_data_handle_t bc_ij_hdl, /* A_ij block handle */
      //       starpu_data_handle_t bc_ik_hdl, /* A_ik block handle */
      //       starpu_data_handle_t bc_jk_hdl, /* A_jk block handle */
      //       starpu_data_handle_t node_hdl,
      //       int prio) {

      //    int ret;

      //    ret = starpu_insert_task(
      //          &cl_update_block,
      //          STARPU_RW, bc_ij_hdl,
      //          STARPU_R, bc_ik_hdl,
      //          STARPU_R, bc_jk_hdl,
      //          STARPU_R, node_hdl,
      //          STARPU_PRIORITY, prio,
      //          0);

      //    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");         
      // }

      // CPU kernel
      // void update_contrib_block_cpu_func(void *buffers[], void *cl_arg) {

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

      //    double *contrib = (double *)STARPU_MATRIX_GET_PTR(buffers[3]);
      //    unsigned cbm = STARPU_MATRIX_GET_NX(buffers[3]);
      //    unsigned cbn = STARPU_MATRIX_GET_NY(buffers[3]);
      //    unsigned ldcontrib = STARPU_MATRIX_GET_LD(buffers[3]);

      //    int kk, nb;

      //    starpu_codelet_unpack_args(
      //          cl_arg, &kk, &nb);

      //    update_block(m, n, blk_ij, ld_ij,
      //                 k,
      //                 blk_ik, ld_ik, 
      //                 blk_jk, ld_jk,
      //                 contrib, ldcontrib,
      //                 cbm, cbn,
      //                 kk==0, nb);
      // }


      // struct starpu_codelet cl_update_contrib_block;      

      // void insert_update_block(
      //       int k, int nb,
      //       starpu_data_handle_t bc_ij_hdl, /* A_ij block handle */
      //       starpu_data_handle_t bc_ik_hdl, /* A_ik block handle */
      //       starpu_data_handle_t bc_jk_hdl, /* A_jk block handle */
      //       starpu_data_handle_t contrib_hdl, /* A_ij block handle */
      //       starpu_data_handle_t node_hdl,
      //       int prio) {

      //    int ret;

      //    ret = starpu_insert_task(
      //          &cl_update_contrib_block,
      //          STARPU_RW, bc_ij_hdl,
      //          STARPU_R, bc_ik_hdl,
      //          STARPU_R, bc_jk_hdl,
      //          STARPU_RW, contrib_hdl,
      //          STARPU_R, node_hdl,
      //          STARPU_PRIORITY, prio,
      //          STARPU_VALUE, &k, sizeof(int),
      //          STARPU_VALUE, &nb, sizeof(int),
      //          0);

      //    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");         
      // }

      /* update_diag_block StarPU task */

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

      // Update contrib StarPU task

      /* CPU task */
      // void update_contrib_cpu_func(void *buffers[], void *cl_arg) {

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
        
      //    int kk;

      //    starpu_codelet_unpack_args(cl_arg, &kk);

      //    update_block(m, n, blk_ij, ld_ij,
      //                 k,
      //                 blk_ik, ld_ik, 
      //                 blk_jk, ld_jk,
      //                 kk==0);
      // }

      // // update_contrib codelet
      // struct starpu_codelet cl_update_contrib;      

      // void insert_update_contrib(
      //       int k,
      //       starpu_data_handle_t bc_ij_hdl, /* A_ij block handle */
      //       starpu_data_handle_t bc_ik_hdl, /* A_ik block handle */
      //       starpu_data_handle_t bc_jk_hdl, /* A_jk block handle */
      //       starpu_data_handle_t node_hdl,
      //       int prio) {

      //    int ret;

      //    ret = starpu_insert_task(
      //          &cl_update_contrib,
      //          STARPU_RW, bc_ij_hdl,
      //          STARPU_R, bc_ik_hdl,
      //          STARPU_R, bc_jk_hdl,
      //          STARPU_R, node_hdl,
      //          STARPU_PRIORITY, prio,
      //          STARPU_VALUE, &k, sizeof(int),
      //          0);

      //    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");         
      // }

      /* update_between StarPU task */

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

      // Factor subtree task

      // extern "C" void spldlt_factor_subtree_c(bool posdef, double *aval, void *akeep, void *fkeep, int p, void **child_contrib, struct cpu_factor_options const* options);

      // Debug
      extern "C" void spldlt_factor_subtree_c(void *akeep, void *fkeep, int p);

      // CPU kernel
      // template <typename T>
      void factor_subtree_cpu_func(void *buffers[], void *cl_arg) {

         // bool posdef;
         // T *aval;
         void *akeep;
         void *fkeep;
         int p;
         // void **child_contrib;
         // struct cpu_factor_options *options;

         // starpu_codelet_unpack_args(
         //       cl_arg,
         //       &posdef, 
         //       &aval,
         //       &akeep, 
         //       &fkeep,
         //       &p,
         //       &child_contrib,
         //       &options);

         starpu_codelet_unpack_args(
               cl_arg,
               &akeep, 
               &fkeep,
               &p,
               0);

         // printf("[factor_subtree_cpu_func] akeep = %p, fkeep = %p\n", akeep, fkeep);
         // printf("[factor_subtree_cpu_func] part: %d, child_contrib: %p\n", p+1, child_contrib);
         
         // spldlt_factor_subtree_c(posdef, aval, akeep, fkeep, p, child_contrib, options);
         // Debug
         spldlt_factor_subtree_c(akeep, fkeep, p);

      }

      // StarPU codelet
      struct starpu_codelet cl_factor_subtree;

      // Debug
      void insert_factor_subtree(
            starpu_data_handle_t root_hdl, // Symbolic handle on root node
            void *akeep, 
            void *fkeep,
            int p) {

         int ret;

         // printf("[insert_factor_subtree] akeep = %p, akeep = %p\n", akeep, fkeep);
         printf("[insert_factor_subtree] root_hdl = %p, p = %d\n", root_hdl, p);

         ret = starpu_task_insert(&cl_factor_subtree,
                                  STARPU_RW, root_hdl,
                                  STARPU_VALUE, &akeep, sizeof(void*),
                                  STARPU_VALUE, &fkeep, sizeof(void*),
                                  STARPU_VALUE, &p, sizeof(int),
                                  0);

         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
      }

      // template <typename T>
      // void insert_factor_subtree(
      //       starpu_data_handle_t root_hdl, // Symbolic handle on root node
      //       bool posdef,
      //       T *aval,
      //       void *akeep, 
      //       void *fkeep,
      //       int p,
      //       void **child_contrib,
      //       struct cpu_factor_options const* options) {

      //    int ret;

      //    // printf("[insert_factor_subtree] akeep = %p, akeep = %p\n", akeep, fkeep);

      //    ret = starpu_task_insert(&cl_factor_subtree,
      //                             STARPU_RW, root_hdl,
      //                             STARPU_VALUE, &posdef, sizeof(bool),
      //                             STARPU_VALUE, &aval, sizeof(T*),
      //                             STARPU_VALUE, &akeep, sizeof(void*),
      //                             STARPU_VALUE, &fkeep, sizeof(void*),
      //                             STARPU_VALUE, &p, sizeof(int),
      //                             STARPU_VALUE, &child_contrib, sizeof(void**),
      //                             STARPU_VALUE, &options, sizeof(struct cpu_factor_options *),
      //                             0);

      //    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
      // }

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

      // Assemble subtree task

      // CPU kernel
      // template <typename T, typename PoolAlloc>
      // void subtree_assemble_cpu_func(void *buffers[], void *cl_arg) {

      //    spldlt::NumericNode<T, PoolAlloc> *node = nullptr;
      //    SymbolicSNode *csnode;
      //    void **child_contrib;
      //    int contrib_idx;

      //    // printf("[subtree_assemble_cpu_func]");

      //    starpu_codelet_unpack_args(cl_arg,
      //                               &node, &csnode,
      //                               &child_contrib, &contrib_idx);

      //    SymbolicNode const& snode = node->symb;

      //    // Retreive contribution block from subtrees
      //    int cn, ldcontrib, ndelay, lddelay;
      //    double const *cval, *delay_val;
      //    int const *crlist, *delay_perm;
      //    spral_ssids_contrib_get_data(
      //          child_contrib[contrib_idx], &cn, &cval, &ldcontrib, &crlist,
      //          &ndelay, &delay_perm, &delay_val, &lddelay
      //          );
      //    if(!cval) return; // child was all delays, nothing more to do

      //    for(int j = 0; j < cn; ++j) {
               
      //       int c = csnode->map[ j ]; // Destination column
                  
      //       T const* src = &cval[j*ldcontrib];

      //       if (c < snode.ncol) {

      //          int ldd = node->get_ldl();
      //          T *dest = &node->lcol[c*ldd];

      //          for (int i = j ; i < cn; ++i) {
      //             // Assemble destination block
      //             dest[ csnode->map[ i ]] += src[i];
      //          }
      //       }
      //    }

      // }

      // // StarPU codelet
      // struct starpu_codelet cl_subtree_assemble;

      // template <typename T, typename PoolAlloc>
      // void insert_subtree_assemble(
      //       spldlt::NumericNode<T, PoolAlloc> *node,
      //       SymbolicSNode *csnode,
      //       starpu_data_handle_t node_hdl,
      //       starpu_data_handle_t root_hdl,
      //       starpu_data_handle_t *dest_hdls, int ndest,
      //       void **child_contrib, int contrib_idx
      //       ) {

      //    int ret;
      //    int nh = 0;
         
      //    struct starpu_data_descr *descrs = new starpu_data_descr[ndest+2];

      //    for (int i=0; i<ndest; i++) {
      //       descrs[nh].handle = dest_hdls[i]; descrs[nh].mode = STARPU_RW;
      //       nh++;
      //    }

      //    // Handle on subtree
      //    descrs[nh].handle = root_hdl; descrs[nh].mode = STARPU_R;
      //    nh++;

      //    // Handle on node to be assembled
      //    descrs[nh].handle = node_hdl; descrs[nh].mode = STARPU_R;
      //    nh++;

      //    ret = starpu_task_insert(&cl_subtree_assemble,
      //                             STARPU_DATA_MODE_ARRAY, descrs, nh,
      //                             STARPU_VALUE, &node, sizeof(spldlt::NumericNode<T, PoolAlloc>*),
      //                             STARPU_VALUE, &csnode, sizeof(SymbolicSNode*),
      //                             STARPU_VALUE, &child_contrib, sizeof(void**),
      //                             STARPU_VALUE, &contrib_idx, sizeof(int),
      //                             0);
       
      //    delete[] descrs;
      // }

      // // Subtree assemble contrib task

      // // CPU kernel
      // template <typename T, typename PoolAlloc>
      // void subtree_assemble_contrib_cpu_func(void *buffers[], void *cl_arg) {

      //    spldlt::NumericNode<T, PoolAlloc> *node = nullptr;
      //    SymbolicSNode *csnode;
      //    void **child_contrib;
      //    int contrib_idx;
      //    int blksz;

      //    // printf("[subtree_assemble_contrib_cpu_func]");

      //    starpu_codelet_unpack_args(cl_arg,
      //                               &node, &csnode,
      //                               &child_contrib, &contrib_idx,
      //                               &blksz);

      //    SymbolicNode const& snode = node->symb;

      //    // Retreive contribution block from subtrees
      //    int cn, ldcontrib, ndelay, lddelay;
      //    double const *cval, *delay_val;
      //    int const *crlist, *delay_perm;
      //    spral_ssids_contrib_get_data(
      //          child_contrib[contrib_idx], &cn, &cval, &ldcontrib, &crlist,
      //          &ndelay, &delay_perm, &delay_val, &lddelay
      //          );
      //    if(!cval) return; // child was all delays, nothing more to do

      //    int sa = snode.ncol / blksz; // Index of first block in contrib
      //    int nr = (snode.nrow-1) / blksz + 1;
      //    int ncontrib = nr-sa;
      //    for(int j = 0; j < cn; ++j) {

      //       int c = csnode->map[ j ]; // Destination column

      //       T const* src = &cval[j*ldcontrib];

      //       if (c >= snode.ncol) {


      //          int cc = c / blksz; // Destination block column
      //          int dest_col_sa = (snode.ncol > cc*blksz) ? 0 : (cc*blksz-snode.ncol); // First col in block

      //          for (int i = j; i < cn; ++i) {
      //             int r = csnode->map[ i ]; // Destination row in parent front
      //             int rr = r / blksz; // Destination block row
      //             // First row index in CB of destination block
      //             int dest_row_sa = (snode.ncol > rr*blksz) ? 0 : (rr*blksz-snode.ncol);
      //             Block<T, PoolAlloc> &dest_blk = node->contrib_blocks[(rr-sa)+(cc-sa)*ncontrib];
      //             int dest_blk_lda = dest_blk.lda;
      //             T *dest = &dest_blk.a[ (c - snode.ncol - dest_col_sa)*dest_blk_lda ];
      //             // Assemble destination block
      //             dest[ r - snode.ncol - dest_row_sa ] += src[i];
      //          }
      //       }
      //    }

      // }

      // // StarPU codelet
      // struct starpu_codelet cl_subtree_assemble_contrib;

      // template <typename T, typename PoolAlloc>
      // void insert_subtree_assemble_contrib(
      //       spldlt::NumericNode<T, PoolAlloc> *node,
      //       SymbolicSNode *csnode,
      //       starpu_data_handle_t node_hdl,
      //       starpu_data_handle_t root_hdl,
      //       starpu_data_handle_t *dest_hdls, int ndest,
      //       void **child_contrib, int contrib_idx,
      //       int blksz, int prio) {

      //    int ret;
      //    int nh = 0;
         
      //    struct starpu_data_descr *descrs = new starpu_data_descr[ndest+2];

      //    for (int i=0; i<ndest; i++) {
      //       descrs[nh].handle = dest_hdls[i]; descrs[nh].mode = STARPU_RW;
      //       nh++;
      //    }

      //    // Handle on subtree
      //    descrs[nh].handle = root_hdl; descrs[nh].mode = STARPU_R;
      //    nh++;

      //    // Handle on node to be assembled
      //    descrs[nh].handle = node_hdl; descrs[nh].mode = STARPU_R;
      //    nh++;

      //    ret = starpu_task_insert(&cl_subtree_assemble_contrib,
      //                             STARPU_DATA_MODE_ARRAY, descrs, nh,
      //                             STARPU_VALUE, &node, sizeof(spldlt::NumericNode<T, PoolAlloc>*),
      //                             STARPU_VALUE, &csnode, sizeof(SymbolicSNode*),
      //                             STARPU_VALUE, &child_contrib, sizeof(void**),
      //                             STARPU_VALUE, &contrib_idx, sizeof(int),
      //                             STARPU_VALUE, &blksz, sizeof(int),
      //                             STARPU_PRIORITY, prio,
      //                             0);

      //    delete[] descrs;
      // }

      // // Assemble block task
      
      // // CPU kernel
      // template <typename T, typename PoolAlloc>
      // void assemble_block_cpu_func(void *buffers[], void *cl_arg) {

      //    // typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;
      //    // std::vector<int, PoolAllocInt> *map;

      //    spldlt::NumericNode<T, PoolAlloc> *node = nullptr, *cnode = nullptr;
      //    int ii, jj; // Block indexes
      //    int *map;
      //    int blksz; // Block size

      //    starpu_codelet_unpack_args(cl_arg, 
      //                               &node, &cnode,
      //                               &ii, &jj, &map, &blksz);

      //    assemble_block(*node, *cnode, ii, jj, map, blksz);
      // }

      // // StarPU codelet
      // struct starpu_codelet cl_assemble_block;

      // template <typename T, typename PoolAlloc>
      // void insert_assemble_block(
      //       spldlt::NumericNode<T, PoolAlloc> *node,
      //       spldlt::NumericNode<T, PoolAlloc> *cnode,
      //       int ii, int jj,
      //       int *cmap, int nb,
      //       starpu_data_handle_t bc_hdl,
      //       starpu_data_handle_t *dest_hdls, int ndest,
      //       starpu_data_handle_t node_hdl, // Symbolic handle for destination node
      //       starpu_data_handle_t cnode_hdl,
      //       int prio) {

      //    int ret;
      //    int nh = 0;
         
      //    struct starpu_data_descr *descrs = new starpu_data_descr[ndest+3];

      //    descrs[nh].handle = bc_hdl; descrs[nh].mode = STARPU_R;
      //    nh++;
         
      //    for (int i=0; i<ndest; i++) {
      //       descrs[nh].handle = dest_hdls[i]; descrs[nh].mode = STARPU_RW;
      //       nh++;
      //    }

      //    // printf("[insert_assemble_block] node_hdl: %p\n", node_hdl);

      //    // Access symbolic handle of node in read mode to ensure that
      //    // it has been initialized
      //    descrs[nh].handle = node_hdl; descrs[nh].mode = STARPU_R;
      //    nh++;

      //    // Access symbolic handle of child node in read mode to
      //    // ensure that assemblies are done before cleaning it
      //    descrs[nh].handle = cnode_hdl; descrs[nh].mode = STARPU_R;
      //    nh++;

      //    ret = starpu_task_insert(&cl_assemble_block,
      //                             STARPU_DATA_MODE_ARRAY, descrs, nh,
      //                             STARPU_VALUE, &node, sizeof(spldlt::NumericNode<T, PoolAlloc>*),
      //                             STARPU_VALUE, &cnode, sizeof(spldlt::NumericNode<T, PoolAlloc>*),
      //                             STARPU_VALUE, &ii, sizeof(int),
      //                             STARPU_VALUE, &jj, sizeof(int),
      //                             STARPU_VALUE, &cmap, sizeof(int*),
      //                             STARPU_VALUE, &nb, sizeof(int),
      //                             STARPU_PRIORITY, prio,
      //                             0);
      //    delete[] descrs;

      // }

      // // Assemble contrib block task

      // // CPU kernel
      // template <typename T, typename PoolAlloc>
      // void assemble_contrib_block_cpu_func(void *buffers[], void *cl_arg) {

      //    // typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;
      //    // std::vector<int, PoolAllocInt> *map;

      //    spldlt::NumericNode<T, PoolAlloc> *node = nullptr, *cnode = nullptr;
      //    int ii, jj; // Block row and col indexes
      //    int *map;
      //    int blksz; // Block size

      //    starpu_codelet_unpack_args(
      //          cl_arg, &node, &cnode,
      //          &ii, &jj, &map, &blksz);

      //    // printf("[assemble_contrib_block_cpu_func]\n");

      //    assemble_contrib_block(*node, *cnode, ii, jj, map, blksz);         
      // }

      // // StarPU codelet
      // struct starpu_codelet cl_assemble_contrib_block;

      // template <typename T, typename PoolAlloc>
      // void insert_assemble_contrib_block(
      //       spldlt::NumericNode<T, PoolAlloc> *node, // Destinaton node
      //       spldlt::NumericNode<T, PoolAlloc> *cnode,// Source node
      //       int ii, int jj,
      //       int *cmap, // Mapping vector i.e. i-th column must be
      //                  // assembled in cmap(i) column of destination
      //                  // node
      //       int blksz,
      //       starpu_data_handle_t bc_hdl,
      //       starpu_data_handle_t *dest_hdls, int ndest,
      //       starpu_data_handle_t node_hdl, // Symbolic handle of destination node
      //       starpu_data_handle_t cnode_hdl, // Symbolic handle of source node
      //       int prio) {

      //    int ret;
      //    int nh = 0;
         
      //    struct starpu_data_descr *descrs = new starpu_data_descr[ndest+3];

      //    descrs[nh].handle = bc_hdl; descrs[nh].mode = STARPU_R;
      //    nh++;
         
      //    for (int i=0; i<ndest; i++) {
      //       descrs[nh].handle = dest_hdls[i]; descrs[nh].mode = STARPU_RW;
      //       nh++;
      //    }

      //    // Access symbolic handle of node in read mode to ensure that
      //    // it has been initialized
      //    descrs[nh].handle = node_hdl; descrs[nh].mode = STARPU_R;
      //    nh++;

      //    // Access symbolic handle of child node in read mode to
      //    // ensure that assemblies are done before cleaning it
      //    descrs[nh].handle = cnode_hdl; descrs[nh].mode = STARPU_R;
      //    nh++;

      //    ret = starpu_task_insert(&cl_assemble_contrib_block,
      //                             STARPU_DATA_MODE_ARRAY, descrs, nh,
      //                             STARPU_VALUE, &node, sizeof(spldlt::NumericNode<T, PoolAlloc>*),
      //                             STARPU_VALUE, &cnode, sizeof(spldlt::NumericNode<T, PoolAlloc>*),
      //                             STARPU_VALUE, &ii, sizeof(int),
      //                             STARPU_VALUE, &jj, sizeof(int),
      //                             STARPU_VALUE, &cmap, sizeof(int*),
      //                             STARPU_VALUE, &blksz, sizeof(int),
      //                             STARPU_PRIORITY, prio,
      //                             0);
      //    delete[] descrs;

      // }

      /* As it is not possible to statically intialize codelet in C++,
         we do it via this function */
      // template <typename T, typename PoolAlloc>
      // template <typename T>
      void codelet_init() {

         // // init_node StarPU codelet
         // starpu_codelet_init(&cl_init_node);
         // cl_init_node.where = STARPU_CPU;
         // cl_init_node.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_init_node.name = "INIT_NODE";
         // cl_init_node.cpu_funcs[0] = init_node_cpu_func<T, PoolAlloc>;

         // // fini_node StarPU codelet
         // starpu_codelet_init(&cl_fini_node);
         // cl_fini_node.where = STARPU_CPU;
         // cl_fini_node.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_fini_node.name = "FINI_NODE";
         // cl_fini_node.cpu_funcs[0] = fini_node_cpu_func<T, PoolAlloc>;

         // // factorize_block StarPU codelet
         // starpu_codelet_init(&cl_factorize_block);
         // cl_factorize_block.where = STARPU_CPU;
         // cl_factorize_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_factorize_block.name = "FACTO_BLK";
         // cl_factorize_block.cpu_funcs[0] = factorize_block_cpu_func;

         // // factorize_contrib_block StarPU codelet
         // starpu_codelet_init(&cl_factorize_contrib_block);
         // cl_factorize_contrib_block.where = STARPU_CPU;
         // cl_factorize_contrib_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_factorize_contrib_block.name = "FACTO_CONTRIB_BLK";
         // cl_factorize_contrib_block.cpu_funcs[0] = factorize_contrib_block_cpu_func;

         // // solve_block StarPU codelet
         // starpu_codelet_init(&cl_solve_block);
         // cl_solve_block.where = STARPU_CPU;
         // cl_solve_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_solve_block.name = "SOLVE_BLK";
         // cl_solve_block.cpu_funcs[0] = solve_block_cpu_func;

         // // solve_contrib_block StarPU codelet
         // starpu_codelet_init(&cl_solve_contrib_block);
         // cl_solve_contrib_block.where = STARPU_CPU;
         // cl_solve_contrib_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_solve_contrib_block.name = "SOLVE_CONTRIB_BLK";
         // cl_solve_contrib_block.cpu_funcs[0] = solve_contrib_block_cpu_func;

         // // update_block StarPU codelet
         // starpu_codelet_init(&cl_update_block);
         // cl_update_block.where = STARPU_CPU;
         // cl_update_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_update_block.name = "UPDATE_BLK";
         // cl_update_block.cpu_funcs[0] = update_block_cpu_func;

         // // update_contrib_block StarPU codelet
         // starpu_codelet_init(&cl_update_contrib_block);
         // cl_update_contrib_block.where = STARPU_CPU;
         // cl_update_contrib_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_update_contrib_block.name = "UPDATE_CONTRIB_BLK";
         // cl_update_contrib_block.cpu_funcs[0] = update_contrib_block_cpu_func;

         // // update_diag_block StarPU codelet
         // starpu_codelet_init(&cl_update_diag_block);
         // cl_update_diag_block.where = STARPU_CPU;
         // cl_update_diag_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_update_diag_block.name = "UPDATE_BLK";
         // cl_update_diag_block.cpu_funcs[0] = update_diag_block_cpu_func;

         // // update_contrib StarPU codelet
         // starpu_codelet_init(&cl_update_contrib);
         // cl_update_contrib.where = STARPU_CPU;
         // cl_update_contrib.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_update_contrib.name = "UPDATE_CONTRIB";
         // cl_update_contrib.cpu_funcs[0] = update_contrib_cpu_func;

         // // update_between StarPU codelet
         // starpu_codelet_init(&cl_update_between);
         // cl_update_between.where = STARPU_CPU;
         // cl_update_between.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_update_between.name = "UPDATE_BETWEEN_BLK";
         // cl_update_between.cpu_funcs[0] = update_between_cpu_func<T, PoolAlloc>;

         // // assemble_block StarPU codelet
         // starpu_codelet_init(&cl_assemble_block);
         // cl_assemble_block.where = STARPU_CPU;
         // cl_assemble_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_assemble_block.name = "ASSEMBLE_BLK";
         // cl_assemble_block.cpu_funcs[0] = assemble_block_cpu_func<T, PoolAlloc>;

         // // assemble_contrib_block StarPU codelet
         // starpu_codelet_init(&cl_assemble_contrib_block);
         // cl_assemble_contrib_block.where = STARPU_CPU;
         // cl_assemble_contrib_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_assemble_contrib_block.name = "ASSEMBLE_CONTRIB_BLK";
         // cl_assemble_contrib_block.cpu_funcs[0] = assemble_contrib_block_cpu_func<T, PoolAlloc>;

         // // subtree_assemble StarPU codelet
         // starpu_codelet_init(&cl_subtree_assemble);
         // cl_subtree_assemble.where = STARPU_CPU;
         // cl_subtree_assemble.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_subtree_assemble.name = "SUBTREE_ASSEMBLE";
         // cl_subtree_assemble.cpu_funcs[0] = subtree_assemble_cpu_func<T, PoolAlloc>;

         // // subtree_assemble_contrib StarPU codelet
         // starpu_codelet_init(&cl_subtree_assemble_contrib);
         // cl_subtree_assemble_contrib.where = STARPU_CPU;
         // cl_subtree_assemble_contrib.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_subtree_assemble_contrib.name = "SUBTREE_ASSEMBLE_CONTRIB";
         // cl_subtree_assemble_contrib.cpu_funcs[0] = subtree_assemble_contrib_cpu_func<T, PoolAlloc>;

         // facto_subtree StarPU codelet
         starpu_codelet_init(&cl_factor_subtree);
         cl_factor_subtree.where = STARPU_CPU;
         cl_factor_subtree.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_factor_subtree.name = "FACTOR_SUBTREE";
         // cl_factor_subtree.cpu_funcs[0] = factor_subtree_cpu_func<T>;
         cl_factor_subtree.cpu_funcs[0] = factor_subtree_cpu_func;

         // // get_contrib StarPU codelet
         // starpu_codelet_init(&cl_get_contrib);
         // cl_get_contrib.where = STARPU_CPU;
         // cl_get_contrib.nbuffers = STARPU_VARIABLE_NBUFFERS;
         // cl_get_contrib.name = "GET_CONTRIB";
         // cl_get_contrib.cpu_funcs[0] = get_contrib_cpu_func;

      }
}} /* namespaces spldlt::starpu  */
