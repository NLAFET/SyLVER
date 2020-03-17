/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#if defined(SPLDLT_USE_STARPU)
#include <starpu.h>
#endif

namespace sylver {
namespace splu {
namespace starpu {
      
   extern starpu_data_handle_t workspace_hdl;

}}} // End of namespace sylver::splu::starpu
