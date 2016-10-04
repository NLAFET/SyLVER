#include "NumericTree.hxx"

#include "ssids/cpu/AppendAlloc.hxx"

using namespace spldlt;

namespace {
   typedef double T;
   const int PAGE_SIZE = 8*1024*1024; // 8MB
   typedef NumericTree<T, PAGE_SIZE, AppendAlloc<T>> NumericTreeDbl;
}

extern "C"
void *spldlt_create_numeric_tree_dbl(void const* symbolic_tree_ptr, const double *const aval) {

   auto const& symbolic_tree = *static_cast<SymbolicTree const*>(symbolic_tree_ptr);

   auto* tree = new NumericTreeDbl(symbolic_tree, aval);

   return (void *) tree;
}
