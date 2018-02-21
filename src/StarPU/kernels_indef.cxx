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
      struct starpu_codelet cl_udpate_contrib_block_indef;
      
}} /* namespaces spldlt::starpu  */
