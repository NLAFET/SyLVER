/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

namespace sylver {
namespace spldlt {
namespace starpu {

   ////////////////////////////////////////////////////////////
   // update_between StarPU task

   //
   // CPU kernel
   //
   
   template <typename T, typename PoolAlloc>
   void update_between_cpu_func(void *buffers[], void *cl_arg) {

      /* Get workspace */
      T *buffer = (T *)STARPU_MATRIX_GET_PTR(buffers[0]);
      // unsigned buf_m = STARPU_MATRIX_GET_NX(buffers[0]);
      // unsigned buf_n = STARPU_MATRIX_GET_NY(buffers[0]);
      // unsigned buf_ld = STARPU_MATRIX_GET_LD(buffers[0]);
      // printf("[update_between_cpu_func] buf_m: %d, buf_n: %d, buf_ld: %d\n", buf_m, buf_n, buf_ld);
      /* Get rowmap workspace */
      int *rlst = (int *)STARPU_VECTOR_GET_PTR(buffers[1]);
      /* Get colmap workspace */
      int *clst = (int *)STARPU_VECTOR_GET_PTR(buffers[2]);
      /* Get A_ij block */
      T *a_ij = (T *)STARPU_MATRIX_GET_PTR(buffers[3]);
      unsigned lda = STARPU_MATRIX_GET_LD(buffers[3]);
      /* Get any A_ik block width */ 
      // unsigned n = STARPU_MATRIX_GET_NY(buffers[4]);
        
      int kk, ii, jj;
      int blksz;
      int cptr, cptr2;
      int rptr, rptr2;

      SymbolicSNode *snode = nullptr, *asnode = nullptr;
      spldlt::NumericNode<T, PoolAlloc> *node = nullptr;

      starpu_codelet_unpack_args(
            cl_arg,
            &snode,
            &node,
            &kk,
            &asnode,
            &ii, &jj,
            &blksz,
            &cptr, &cptr2,
            &rptr, &rptr2
            );

      // printf("[update_between_cpu_func] n: %d\n", n);
      // printf("[update_between_cpu_func] kk: %d, ii: %d, jj: %d\n", kk, ii, jj);
      // printf("[update_between_block_task] cptr: %d, cptr2: %d, rptr: %d, rptr2: %d\n", cptr, cptr2, rptr, rptr2);

      // printf("[update_between_cpu_func] snode: %p\n", snode);
      // printf("[update_between_cpu_func]  snode nrow: %d\n", snode->nrow);
      // printf("[update_between_cpu_func] asnode nrow: %d\n", asnode->nrow);

      // int mc = cptr2-cptr+1; // number of rows in Ajk
      // int mr = rptr2-rptr+1; // number of rows in Aik
      // printf("[update_between_cpu_func] mr: %d, mc: %d, n: %d, blksz: %d\n", mr, mc, n, blksz);
      // rlst = new int[mr]; // Get ptr on rowmap array
      // clst = new int[mc];
      // buffer = new T[mr*mc];
      // printf("[update_between_cpu_func] buffer: %p\n", buffer);
      // for (int j = 0; j < blksz; j++)
      //    for (int i = 0; i < blksz; i++)
      //       buffer[j*blksz+i] = 0.0;

      // Determine A_ik and A_kj column width 
      int blkn = std::min(snode->ncol - kk*blksz, blksz);
      // printf("[update_between_cpu_func] blkn: %d\n", blkn);

      update_between_block(
            blkn, // blcok column width
            kk, ii, jj, // block row and block column index of A_ij block in destination node
            blksz, // block size
            cptr, cptr2, // local row indexes of a_kj elements 
            // in source node
            rptr, rptr2, // local row indexes of a_ik elements 
            // in source node
            *snode, // symbolic source node  
            *node, // numeric source node
            *asnode,  // symbolic destination node
            a_ij, lda, // block to be updated in destination node  
            buffer, // workspace
            rlst, clst // workpaces for col and row mapping
            );
   }

   // update beween codelet
   struct starpu_codelet cl_update_between;      
      
   // Note that bc_ik_hdls and bc_jk_hdls hadles are used for coherency
   template <typename T, typename PoolAlloc>
   void insert_update_between(
         SymbolicSNode *snode,
         spldlt::NumericNode<T, PoolAlloc> *node,
         starpu_data_handle_t *bc_ik_hdls, int nblk_ik, /* A_ij block handle */
         starpu_data_handle_t *bc_jk_hdls, int nblk_jk, /* A_ik block handle */
         int kk,
         SymbolicSNode *asnode,
         starpu_data_handle_t anode_hdl,
         starpu_data_handle_t bc_ij_hdl,
         int ii, int jj,
         starpu_data_handle_t work_hdl,
         starpu_data_handle_t row_list_hdl,
         starpu_data_handle_t col_list_hdl,
         int blksz,
         int cptr, int cptr2,
         int rptr, int rptr2,
         int prio) {
         
      struct starpu_data_descr *descrs = 
         new starpu_data_descr[nblk_ik+nblk_jk+5];

      int ret;
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
      // printf("[insert_update_between] a_ij handle ptr: %p\n", bc_ij_hdl);
      descrs[nh].handle =  bc_ij_hdl; descrs[nh].mode = /* STARPU_RW;*/  (starpu_data_access_mode) (STARPU_RW | STARPU_COMMUTE);
      nh = nh + 1;
      // printf("[insert_update_between] nblk_ik: %d, nblk_jk: %d\n", nblk_ik, nblk_jk);
      // A_ik handles
      // DEBUG
      // printf("[insert_update_between] A_ik handle %d ptr: %p\n", 0, bc_ik_hdls[0]);
      // descrs[nh].handle = bc_ik_hdls[0];  descrs[nh].mode = STARPU_R;
      // nh = nh + 1;
      for(int i = 0; i < nblk_ik; i++) {
         // printf("[insert_update_between] a_ik handle %d, ptr: %p\n", i, bc_ik_hdls[i]);
         descrs[nh].handle = bc_ik_hdls[i];  descrs[nh].mode = STARPU_R;
         nh = nh + 1;
      }
      // A_jk handles
      for(int i = 0; i < nblk_jk; i++){
         // printf("[insert_update_between] a_jk handle %d, ptr: %p\n", i, bc_jk_hdls[i]);
         descrs[nh].handle = bc_jk_hdls[i];  descrs[nh].mode = STARPU_R;
         nh = nh + 1;
      }
         
      // Make sure that destination node as been initialized
      descrs[nh].handle =  anode_hdl; descrs[nh].mode = STARPU_R;
      nh = nh + 1;         

      ret = starpu_task_insert(&cl_update_between,
                               STARPU_DATA_MODE_ARRAY, descrs,   nh,
                               STARPU_VALUE, &snode, sizeof(SymbolicSNode*),
                               STARPU_VALUE, &node, sizeof(spldlt::NumericNode<T, PoolAlloc>*),
                               STARPU_VALUE, &kk, sizeof(int),
                               STARPU_VALUE, &asnode, sizeof(SymbolicSNode*),
                               STARPU_VALUE, &ii, sizeof(int),
                               STARPU_VALUE, &jj, sizeof(int),
                               STARPU_VALUE, &blksz, sizeof(int),
                               STARPU_VALUE, &cptr, sizeof(int),
                               STARPU_VALUE, &cptr2, sizeof(int),
                               STARPU_VALUE, &rptr, sizeof(int),
                               STARPU_VALUE, &rptr2, sizeof(int),
                               STARPU_PRIORITY, prio,
                               0);

      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

      delete[] descrs;
   }

   
}}}  // End of namespaces sylver::spldlt::starpu
