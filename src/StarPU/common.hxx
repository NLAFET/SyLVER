/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

namespace sylver {
namespace starpu {

#if defined(SPLDLT_USE_GPU)
   // @brief Disable tensor cores for all CUDA workers
   void disable_tc();
   // @brief Enable tensor cores for all CUDA workers
   void enable_tc();
#endif

}} // End of namespace sylver::starpu
