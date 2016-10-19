#pragma once

#include <vector>

#include "SymbolicSNode.hxx"
#include "kernels/factor.hxx"

#include <starpu.h>

namespace spldlt { namespace starpu {

      template <typename T, typename PoolAlloc>
      void register_node(
            SymbolicSNode &snode,
            NumericNode<T, PoolAlloc> &node,
            int nb
            ) {

         int m = snode.nrow;
         int n = snode.ncol;
         T *a = node.lcol;
         int lda = align_lda<T>(m);
         int nr = (m-1) / nb + 1; // number of block rows
         int nc = (n-1) / nb + 1; // number of block columns
         // snode.handles.reserve(nr*nc);
         snode.handles.resize(nr*nc); // allocate handles

         for(int j = 0; j < nc; ++j) {
               
            int blkn = std::min(nb, n - j*nb);
               
            for(int i = j; i < nr; ++i) {
               
               int blkm = std::min(nb, m - i*nb);
                                 
               starpu_matrix_data_register(
                     &(snode.handles[i + j*nr]), // StarPU handle ptr 
                     STARPU_MAIN_RAM, // memory 
                     reinterpret_cast<uintptr_t>(&a[(j*nb)*lda+(i*nb)]),
                     lda, blkm, blkn,
                     sizeof(T));
            }
         }         
      }
      
      /* factorize_block StarPU task */

      /* CPU task */
      /* TODO generic prec */
      void factorize_block_cpu_func(void *buffers[], void *cl_arg) {
         
         double *blk = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
         unsigned m = STARPU_MATRIX_GET_NX(buffers[0]);
         unsigned n = STARPU_MATRIX_GET_NY(buffers[0]);
         unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);


         factorize_diag_block(m, n, blk, ld);
      }
      
      /* FIXME: although it would be better to statically initialize
         the codelet, it is not well suported by g++ */

      // struct starpu_codelet cl_factorize_block = {
      //    // .where = w,
      //    // .cpu_funcs = {factorize_block_cpu_func, NULL},
      //    .nbuffers = STARPU_VARIABLE_NBUFFERS,
      //    // .name = "FACTO_BLK"
      // };

      /* factorize block codelet */
      struct starpu_codelet cl_factorize_block;      

      /* Insert factorization of diag block into StarPU*/
      void insert_factorize_block(starpu_data_handle_t bc_hdl) {
                  
         int ret;
     
         ret = starpu_insert_task(
               &cl_factorize_block,
               STARPU_RW, bc_hdl,
               0);

         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
      }

      /* solve_block StarPU task */

      /* CPU task */
      void solve_block_cpu_func(void *buffers[], void *cl_arg) {

         /* Get diag block pointer and info */
         double *blk = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
         unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);

         /* Get sub diag block pointer and info */
         double *blk_ik = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
         unsigned m = STARPU_MATRIX_GET_NX(buffers[1]);
         unsigned n = STARPU_MATRIX_GET_NY(buffers[1]);
         unsigned ld_ik = STARPU_MATRIX_GET_LD(buffers[1]);

         /* Call kernel function */
         solve_block(m, n, blk, ld, blk_ik, ld_ik);
      }

      /* solve_block codelet */
      struct starpu_codelet cl_solve_block;

      /* Insert solve of sub diag block into StarPU */
      void insert_solve_block(
            starpu_data_handle_t bc_kk_hdl, /* diag block handle */
            starpu_data_handle_t bc_ik_hdl /* sub diag block handle */
            ) {

         int ret;

         ret = starpu_insert_task(
               &cl_solve_block,
               STARPU_R, bc_kk_hdl,
               STARPU_RW, bc_ik_hdl,
               0);

         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
      }

      /* update_block StarPU task */

      /* CPU task */
      void update_block_cpu_func(void *buffers[], void *cl_arg) {

         /* Get A_ij pointer */
         double *blk_ij = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
         unsigned m = STARPU_MATRIX_GET_NX(buffers[0]);
         unsigned n = STARPU_MATRIX_GET_NY(buffers[0]);
         unsigned ld_ij = STARPU_MATRIX_GET_LD(buffers[0]);
                  
         /* Get A_ik pointer */
         double *blk_ik = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
         unsigned k = STARPU_MATRIX_GET_NY(buffers[1]);
         unsigned ld_ik = STARPU_MATRIX_GET_LD(buffers[1]); 

         double *blk_jk = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
         unsigned ld_jk = STARPU_MATRIX_GET_LD(buffers[2]); 
        
         update_block(m, n, blk_ij, ld_ij,
                      k,
                      blk_ik, ld_ik, 
                      blk_jk, ld_jk);

      }

      /* update block codelet */
      struct starpu_codelet cl_update_block;      

      void insert_update_block(
            starpu_data_handle_t bc_ij_hdl, /* A_ij block handle */
            starpu_data_handle_t bc_ik_hdl, /* A_ik block handle */
            starpu_data_handle_t bc_jk_hdl /* A_jk block handle */
            ) {

         int ret;

         ret = starpu_insert_task(
               &cl_update_block,
               STARPU_RW, bc_ij_hdl,
               STARPU_R, bc_ik_hdl,
               STARPU_R, bc_jk_hdl,
               0);

         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");         
      }

      /* update_between StarPU task */

      /* CPU task */
      void update_between_cpu_func(void *buffers[], void *cl_arg) {

         /* Get workspace */
         double *work = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
         /* Get rowmap workspace */
         int *rowmap = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
         /* Get colmap workspace */
         int *colmap = (int *)STARPU_VECTOR_GET_PTR(buffers[2]);
         
      }

      /* update beween codelet */
      struct starpu_codelet cl_update_between;      
      
      /* Note that bc_ik_hdls and bc_jk_hdls hadles are used for coherency */
      void insert_update_between(
            starpu_data_handle_t *bc_ik_hdls, int nblk_ik, /* A_ij block handle */
            starpu_data_handle_t *bc_jk_hdls, int nblk_jk, /* A_ik block handle */
            starpu_data_handle_t bc_ij_hdl,
            starpu_data_handle_t work_hdl,
            starpu_data_handle_t row_list_hdl,
            starpu_data_handle_t col_list_hdl
            ) {
         
         struct starpu_data_descr *descrs = 
            new starpu_data_descr[nblk_ik+nblk_jk+4];

         int nh = 0;
         
         // worksapce 
         descrs[nh].handle =  work_hdl; descrs[nh].mode = STARPU_SCRATCH;
         nh = nh + 1;
         // row_list
         descrs[nh].handle =  row_list_hdl; descrs[nh].mode = STARPU_SCRATCH;
         nh = nh + 1;
         // col_list
         descrs[nh].handle =  col_list_hdl; descrs[nh].mode = STARPU_SCRATCH;
         nh = nh + 1;
         // A_ij 
         descrs[nh].handle =  bc_ij_hdl; descrs[nh].mode = (starpu_data_access_mode) (STARPU_RW | STARPU_COMMUTE);
         nh = nh + 1;
         // A_ik handles
         for(int i = 0; i < nblk_ik; i++) {
            descrs[nh].handle = bc_ik_hdls[i];  descrs[nh].mode = STARPU_R;
            nh = nh + 1;
         }
         // A_jk handles
         for(int i = 0; i < nblk_jk; i++){
            descrs[nh].handle = bc_jk_hdls[i];  descrs[nh].mode = STARPU_R;
            nh = nh + 1;
         }
         
         int ret;

         ret = starpu_task_insert(&cl_update_between,
                                  STARPU_DATA_MODE_ARRAY, descrs,   nh,
                                  0);

         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

      }

      /* As it is not possible to statically intialize codelet in C++,
         we do it via this function */
      void codelet_init() {
         
         // Initialize factorize_block StarPU codelet
         starpu_codelet_init(&cl_factorize_block);
         cl_factorize_block.where = STARPU_CPU;
         cl_factorize_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_factorize_block.name = "FACTO_BLK";
         cl_factorize_block.cpu_funcs[0] = factorize_block_cpu_func;

         // Initialize solve_block StarPU codelet
         starpu_codelet_init(&cl_solve_block);
         cl_solve_block.where = STARPU_CPU;
         cl_solve_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_solve_block.name = "SOLVE_BLK";
         cl_solve_block.cpu_funcs[0] = solve_block_cpu_func;

         // Initialize update_block StarPU codelet
         starpu_codelet_init(&cl_update_block);
         cl_update_block.where = STARPU_CPU;
         cl_update_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_update_block.name = "UPDATE_BLK";
         cl_update_block.cpu_funcs[0] = update_block_cpu_func;

         // Initialize update_between StarPU codelet
         starpu_codelet_init(&cl_update_between);
         cl_update_between.where = STARPU_CPU;
         cl_update_between.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_update_between.name = "UPDATE_BETWEEN_BLK";
         cl_update_between.cpu_funcs[0] = update_between_cpu_func;

      }   
}} /* namespaces spldlt::starpu  */
