#include "SymbolicTree.hxx"

// #include "ssids/cpu/SymbolicSubtree.hxx"

using namespace spldlt;
// using namespace spral::ssids::cpu;

extern "C"
void *spldlt_create_symbolic_tree(
      void* akeep, int n, int nnodes, int const* sptr, int const* sparent, 
      long const* rptr, int const* rlist, long const* nptr, long const* nlist, 
      int nparts, int const* part, int const* contrib_idx, int const* exec_loc)
   //    void* akeep,
   //    int n, int nnodes, int const* sptr, int const* sparent,
   //    long const* rptr, int const* rlist, long const* nptr,
   //    long const* nlist, int nparts, int const* part, 
   //    int const* contrib_idx, int const* exec_loc, 
   //    int const* contrib_dest,
   //    struct cpu_factor_options const* options) 
{
   // return (void *) new SymbolicTree(akeep, n, nnodes, sptr, sparent, rptr, rlist, nptr, nlist, 
   //                                  nparts, part, contrib_idx, exec_loc, contrib_dest);

   // return (void*) new SymbolicSubtree(
   //       n, 1, nnodes, sptr, sparent, rptr, rlist, nptr, nlist, 0,
   //       NULL, *options
   //       );

   return (void *) new SymbolicTree(
         akeep, n, nnodes, sptr, sparent, rptr, rlist, nptr, nlist, nparts, part, 
         contrib_idx, exec_loc);
}
