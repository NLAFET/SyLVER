/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

namespace sylver {
namespace spldlt {
namespace starpu {

   ////////////////////////////////////////////////////////////
   // Factor subtree task

   //
   // CPU kernel
   //
   
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

      // Options passed to SSIDS for subtree factorization 
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
            factor_subtree(
                  akeep, fkeep, p, aval, scaling, child_contrib,
                  &subtree_opts,
                  &stats);
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

   //
   // StarPU codelets
   //

   // factor_subtree StarPU codelet
   extern struct starpu_codelet cl_factor_subtree;
   // factor_subtree StarPU codelet
   extern struct starpu_codelet cl_factor_subtree_gpu;

   //
   // StarPU task insert routines
   //

   /// @brief Launch StarPU task for factorizing a subtree on a
   /// specific worker
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
   /// @param root_hdl symbolic StarPU handle representing the root
   /// node of the subtree
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
   
   ////////////////////////////////////////////////////////////
   // Assemble subtree block task

   //
   // CPU kernel
   //
   
   template <typename NumericFrontType>
   void subtree_assemble_block_cpu_func(void *buffers[], void *cl_arg) {

      NumericFrontType *node = nullptr;
      sylver::SymbolicFront *csnode;
      void **child_contrib;
      int contrib_idx;
      int i = -1; // Block-row index 
      int j = -1; // Block-column index

      starpu_codelet_unpack_args(
            cl_arg,
            &node, &csnode,
            &child_contrib, &contrib_idx,
            &i, &j);

      assert((i >= 0) && (j >= 0));
      
      assemble_subtree_block(
            *node, *csnode, child_contrib, contrib_idx, i, j);
   }   

   // subtree_assemble StarPU codelet
   extern struct starpu_codelet cl_subtree_assemble_block;

   template <typename NumericFrontType>
   void insert_subtree_assemble_block(
         NumericFrontType *node,
         sylver::SymbolicFront const* csnode,
         starpu_data_handle_t node_hdl,
         starpu_data_handle_t root_hdl,
         starpu_data_handle_t *dest_hdls, int ndest,
         void **child_contrib, int contrib_idx,
         int i, int j
         ) {

      int ret;
      int nh = 0;

      // Number of StarPU data descriptors
      int ndescr = ndest+2;
      
      struct starpu_data_descr *descrs = new starpu_data_descr[ndescr];

      for (int i=0; i<ndest; i++) {
         assert(nh < ndescr);
         descrs[nh].handle = dest_hdls[i]; descrs[nh].mode = STARPU_RW;
         nh++;
      }

      // Handle on subtree
      assert(nh < ndescr);
      descrs[nh].handle = root_hdl; descrs[nh].mode = STARPU_R;
      nh++;

      ret = starpu_task_insert(
            &cl_subtree_assemble_block,
            STARPU_DATA_MODE_ARRAY, descrs, nh,
            STARPU_VALUE, &node, sizeof(NumericFrontType*),
            STARPU_VALUE, &csnode, sizeof(sylver::SymbolicFront*),
            STARPU_VALUE, &child_contrib, sizeof(void**),
            STARPU_VALUE, &contrib_idx, sizeof(int),
            STARPU_VALUE, &i, sizeof(int),
            STARPU_VALUE, &j, sizeof(int),
            0);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
      
      delete[] descrs;

   }

   ////////////////////////////////////////////////////////////
   // Assemble subtree task

   //
   // CPU kernel
   //
   
   template <typename NumericFrontType>
   void subtree_assemble_cpu_func(void *buffers[], void *cl_arg) {

      NumericFrontType *node = nullptr;
      sylver::SymbolicFront *csnode;
      void **child_contrib;
      int contrib_idx;

      // printf("[subtree_assemble_cpu_func]\n");

      starpu_codelet_unpack_args(
            cl_arg,
            &node, &csnode,
            &child_contrib, &contrib_idx);

      assemble_subtree(*node, *csnode, child_contrib, contrib_idx);

   }

   // subtree_assemble StarPU codelet
   extern struct starpu_codelet cl_subtree_assemble;

   template <typename NumericFrontType>
   void insert_subtree_assemble(
         NumericFrontType *node,
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

      ret = starpu_task_insert(
            &cl_subtree_assemble,
            STARPU_DATA_MODE_ARRAY, descrs, nh,
            STARPU_VALUE, &node, sizeof(NumericFrontType*),
            STARPU_VALUE, &csnode, sizeof(sylver::SymbolicFront*),
            STARPU_VALUE, &child_contrib, sizeof(void**),
            STARPU_VALUE, &contrib_idx, sizeof(int),
            0);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
       
      delete[] descrs;
   }

   ////////////////////////////////////////////////////////////
   // Subtree assemble contrib block task

   //
   // CPU kernel
   //
   
   template <typename NumericFrontType>
   void subtree_assemble_contrib_block_cpu_func(void *buffers[], void *cl_arg) {

      NumericFrontType *node = nullptr;
      sylver::SymbolicFront *csnode = nullptr;
      void **child_contrib;
      int contrib_idx;
      int i = -1; // Block-row index 
      int j = -1; // Block-column index

      starpu_codelet_unpack_args(
            cl_arg,
            &node, &csnode,
            &child_contrib, &contrib_idx,
            &i, &j);

      assert((i >= 0) && (j >= 0));
      assert(nullptr != node);
      assert(nullptr != csnode);
      
      assemble_contrib_subtree_block(
            *node, *csnode, child_contrib, contrib_idx, i, j);
                                     
   }

   // StarPU codelet
   extern struct starpu_codelet cl_subtree_assemble_contrib_block;


   template <typename NumericFrontType>
   void insert_subtree_assemble_contrib_block(
         NumericFrontType *node,
         sylver::SymbolicFront const* csnode,
         int i, int j,
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

      ret = starpu_task_insert(&cl_subtree_assemble_contrib_block,
                               STARPU_DATA_MODE_ARRAY, descrs, nh,
                               STARPU_VALUE, &node, sizeof(NumericFrontType*),
                               STARPU_VALUE, &csnode, sizeof(sylver::SymbolicFront*),
                               STARPU_VALUE, &child_contrib, sizeof(void**),
                               STARPU_VALUE, &contrib_idx, sizeof(int),
                               STARPU_VALUE, &i, sizeof(int),
                               STARPU_VALUE, &j, sizeof(int),
                               STARPU_PRIORITY, prio,
                               0);

      delete[] descrs;

   }
   
   ////////////////////////////////////////////////////////////
   // Subtree assemble contrib task

   // CPU kernel
   template <typename NumericFrontType>
   void subtree_assemble_contrib_cpu_func(void *buffers[], void *cl_arg) {

      NumericFrontType *node = nullptr;
      sylver::SymbolicFront *csnode = nullptr;
      void **child_contrib;
      int contrib_idx;

      // printf("[subtree_assemble_contrib_cpu_func]\n");

      starpu_codelet_unpack_args(cl_arg,
                                 &node, &csnode,
                                 &child_contrib, &contrib_idx);

      assemble_contrib_subtree(*node, *csnode, child_contrib, contrib_idx);
   }

   // StarPU codelet
   extern struct starpu_codelet cl_subtree_assemble_contrib;

   template <typename NumericFrontType>
   void insert_subtree_assemble_contrib(
         NumericFrontType*node,
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
                               STARPU_VALUE, &node, sizeof(NumericFrontType*),
                               STARPU_VALUE, &csnode, sizeof(sylver::SymbolicFront*),
                               STARPU_VALUE, &child_contrib, sizeof(void**),
                               STARPU_VALUE, &contrib_idx, sizeof(int),
                               STARPU_PRIORITY, prio,
                               0);

      delete[] descrs;
   }
   
}}}  // End of namespaces sylver::spldlt::starpu
