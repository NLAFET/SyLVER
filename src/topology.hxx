/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

namespace sylver {
namespace topology {

   struct NumaRegion {
      int nproc;
      int ngpu;
      int *gpus;
   };

}} // End of namespace sylver::topology
