#include "SymbolicTree.hxx"

// #include "ssids/cpu/SymbolicSubtree.hxx"

using namespace spldlt;
// using namespace spral::ssids::cpu;

extern "C"
void *spldlt_create_symbolic_tree(int n, int nnodes, int const* sptr, int const* sparent, 
                                  long const* rptr, int const* rlist, int const* nptr, 
                                  int const* nlist, struct cpu_factor_options const* options) {
   return (void *) new SymbolicTree(n, nnodes, sptr, sparent, rptr, rlist, nptr, nlist);

   // return (void*) new SymbolicSubtree(
   //       n, 1, nnodes, sptr, sparent, rptr, rlist, nptr, nlist, 0,
   //       NULL, *options
   //       );   
}
