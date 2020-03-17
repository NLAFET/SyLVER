/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

// StarPU
#include <starpu.h>

namespace sylver {
namespace spldlt {
namespace starpu {
   
   // fini_cnodes StarPU codelet
   struct starpu_codelet cl_fini_cnodes;
   
   // assemble StarPU codelet
   struct starpu_codelet cl_assemble;

   // assemble_contrib StarPU codelet
   struct starpu_codelet cl_assemble_contrib;

}}} // End of namespaces sylver::spldlt::starpu
