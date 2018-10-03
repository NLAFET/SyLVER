#include "kernels_indef.hxx"

#include <starpu.h>

namespace spldlt { namespace starpu {

      starpu_data_handle_t workspace_hdl;
      
      /* factor_block_app StarPU codelet */
      struct starpu_codelet cl_factor_block_app;

      /* applyN_block_app StarPU codelet */
      struct starpu_codelet cl_applyN_block_app;      

      /* applyT_block_app StarPU codelet */
      struct starpu_codelet cl_applyT_block_app;      
      
      /* updateT_block_app StarPU codelet */
      struct starpu_codelet cl_updateT_block_app;      

      /* updateN_block_app StarPU codelet */
      struct starpu_codelet cl_updateN_block_app;      

      /* adjust StarPU codelet */
      struct starpu_codelet cl_adjust;

      /* restore StarPU codelet */
      struct starpu_codelet cl_restore_failed_block_app;

#if defined(SPLDLT_USE_PROFILING)
      // udpate_contrib_block_indef StarPU perfmodel
      struct starpu_perfmodel update_contrib_block_app_perfmodel;
#endif

      // udpate_contrib_block_indef StarPU codelet
      struct starpu_codelet cl_update_contrib_block_app;

      // permute_failed StarPU codelet
      struct starpu_codelet cl_permute_failed;

      // cl_form_contrib StarPU codelet
      struct starpu_codelet cl_form_contrib;

      // cl_zero_contrib_blocks StarPU codelet
      struct starpu_codelet cl_zero_contrib_blocks;

      ////////////////////////////////////////
      // assemble_contrib_sync

      // assemble_contrib_sync StarPU codelet
      struct starpu_codelet cl_assemble_contrib_sync;

      void assemble_contrib_sync_cpu_func(void *buffers[], void *cl_arg) {
         // printf("[assemble_contrib_sync_cpu_func]\n");
      }

      // template <typename T, typename PoolAlloc>
      void insert_assemble_contrib_sync(
            starpu_data_handle_t contrib_hdl,
            int nodeidx) {

         int ret;

         starpu_tag_t tag_assemble_contrib = (starpu_tag_t) (3*nodeidx+1);
         // starpu_tag_t tag2 = (starpu_tag_t) (2*nodeidx+1);
         // starpu_tag_declare_deps(tag2, 1, tag1);

         // printf("[insert_assemble_contrib_sync] nodeidx = %d, tag1 = %d, , tag2 = %d\n", 
         //        nodeidx, tag1, tag2);

         // ret = starpu_task_insert(
         //       &cl_assemble_contrib_sync,
         //       STARPU_TAG, tag1,
         //       STARPU_RW, contrib_hdl,
         //       0);
         // STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

         struct starpu_task *task = starpu_task_create();
         task->cl = &cl_assemble_contrib_sync; 
         task->use_tag = 1;
         task->tag_id = tag_assemble_contrib;
         task->handles[0] = contrib_hdl;
         ret = starpu_task_submit(task);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

      }

      ////////////////////////////////////////
      // nelim_sync

      // nelim_sync StarPU codelet
      struct starpu_codelet cl_nelim_sync;

      void nelim_sync_cpu_func(void *buffers[], void *cl_arg) {
         // printf("[nelim_sync_cpu_func]\n");
      }

      void insert_nelim_sync(
            starpu_data_handle_t node_hdl,
            int nodeidx) {

         int ret;
         
         starpu_tag_t tag_assemble_contrib = (starpu_tag_t) (3*nodeidx+1);
         starpu_tag_t tag_factor_failed = (starpu_tag_t) (3*nodeidx+2);
         // starpu_tag_t tag1 = (starpu_tag_t) (3*nodeidx);
         // starpu_tag_t tag2 = (starpu_tag_t) (2*nodeidx+1);
         starpu_tag_t tag_nelim = (starpu_tag_t) (3*nodeidx);

         // starpu_tag_declare_deps(
         //       tag_nelim, 1, 
         //       tag_assemble_contrib);

         starpu_tag_declare_deps(
               tag_nelim, 1, 
               tag_factor_failed);
         
         // starpu_tag_declare_deps(
         //       tag_nelim, 2,
         //       tag_assemble_contrib,
         //       tag_factor_failed
         //       );

         // printf("[insert_nelim_sync] nodeidx = %d, tag1 = %d, , tag2 = %d\n", 
         //        nodeidx, tag1, tag2);

         // ret = starpu_task_insert(
         //       &cl_nelim_sync,
         //       STARPU_TAG, tag2,
         //       STARPU_RW, node_hdl,
         //       0);
         // STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

         struct starpu_task *task = starpu_task_create();
         task->cl = &cl_nelim_sync; 
         task->use_tag = 1;
         task->tag_id = tag_nelim;
         task->handles[0] = node_hdl;
         ret = starpu_task_submit(task);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

      }

      ////////////////////////////////////////
      // factor_failed_sync

      void factor_failed_sync_cpu_func(void *buffers[], void *cl_arg) {}

      // factor_failed_sync StarPU codelet
      struct starpu_codelet cl_factor_failed_sync;
      
      void insert_factor_failed_sync(int nodeidx) {

         int ret;

         starpu_tag_t tag_factor_failed = (starpu_tag_t) (3*nodeidx+2);

         struct starpu_task *task = starpu_task_create();
         task->cl = &cl_factor_failed_sync; 
         task->use_tag = 1;
         task->tag_id = tag_factor_failed;
         ret = starpu_task_submit(task);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

      }
      
      ////////////////////////////////////////
      // assemble_delays
      
      struct starpu_codelet cl_assemble_delays;

      ////////////////////////////////////////
      // assemble_delays_subtree

      struct starpu_codelet cl_assemble_delays_subtree;

}} /* namespaces spldlt::starpu  */
