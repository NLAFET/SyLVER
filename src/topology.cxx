/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

#include "topology.hxx"

#include <iostream>

#include <hwloc.h>

using namespace sylver::topology;

extern "C"
void sylver_topology_create_c(
      int* nregions, NumaRegion** regions) {

   *nregions = 0;
      
   hwloc_topology_t topology;

   hwloc_topology_init(&topology);
   hwloc_topology_load(topology);

   // Count the number of NUMA nodes
   int n_numa_nodes = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);

   std::cout << "[sylver_topology_create_c] Number of NUMA nodes = " << n_numa_nodes << std::endl;
   
   if (n_numa_nodes > 0) {
      *nregions = n_numa_nodes;
      *regions = new NumaRegion[*nregions];
      for(int i=0; i<*nregions; ++i) {
         NumaRegion& region = (*regions)[i];
         hwloc_obj_t hw_numa_obj = hwloc_get_obj_by_type(
               topology, HWLOC_OBJ_NUMANODE, i);
         // Count the number of physical cores for this NUMA node
         int ncores = hwloc_get_nbobjs_inside_cpuset_by_type(
               topology, hw_numa_obj->cpuset,
               HWLOC_OBJ_CORE
               // HWLOC_OBJ_PU
               );
         std::cout << "[sylver_topology_create_c] NUMA node " << i << ", Number of cores = " << ncores << std::endl;
         if (ncores > 0) {
            region.nproc = ncores;
         }
      }
   }
      
   hwloc_topology_destroy(topology);
   
}
