#include "SymbolicAtree.hxx"

using namespace spldlt;

extern "C"
void *spldlt_create_symbolic_atree(int nnodes, int const* sptr, int const* sparent, 
                                   long const* rptr, int const* rlist, int const* nptr, 
                                   int const* nlist) {
   return (void *) new SymbolicAtree(nnodes, sptr, sparent, rptr, rlist, nptr, nlist);
}
