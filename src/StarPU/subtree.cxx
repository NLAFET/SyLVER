/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

// StarPU
#include <starpu.h>

namespace sylver {
namespace spldlt {
namespace starpu {

   // factor_subtree StarPU codelet
   struct starpu_codelet cl_factor_subtree;

   // factor_subtree_gpu StarPU codelet
   struct starpu_codelet cl_factor_subtree_gpu;

   // subtree_assemble StarPU codelet
   struct starpu_codelet cl_subtree_assemble;
   
   // subtree_assemble_block StarPU codelet
   struct starpu_codelet cl_subtree_assemble_block;

   // subtree_assemble_contrib StarPU codelet
   struct starpu_codelet cl_subtree_assemble_contrib;
   
   // subtree_assemble_contrib_block StarPU codelet
   struct starpu_codelet cl_subtree_assemble_contrib_block;
   
}}}  // End of namespaces sylver::spldlt::starpu
