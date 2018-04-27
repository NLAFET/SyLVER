#include <starpu.h>

namespace spldlt { namespace starpu {
      
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
      struct starpu_codelet cl_restore_block_app;

      // udpate_contrib_block_indef StarPU codelet
      struct starpu_codelet cl_update_contrib_block_app;

      // permute_failed StarPU codelet
      struct starpu_codelet cl_permute_failed;

      // factor_front_indef_secondpass_nocontrib StarPU codelet
      struct starpu_codelet cl_factor_front_indef_failed;

      // // factor_sync StarPU codelet
      // struct starpu_codelet cl_factor_sync;

      // assemble_contrib_sync StarPU codelet
      struct starpu_codelet cl_assemble_contrib_sync;

      // nelim_sync StarPU codelet
      struct starpu_codelet cl_nelim_sync;
      
}} /* namespaces spldlt::starpu  */
