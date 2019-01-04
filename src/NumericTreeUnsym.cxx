#include "NumericTreeUnsym.hxx"

using namespace spldlt;

namespace {
   typedef double T;
   const int PAGE_SIZE = 8*1024*1024; // 8MB
   typedef NumericTreeUnsym<T> NumericTreeUnsymDbl; 
}

extern "C"
void* splu_create_numeric_tree_dbl(
      void *symbolic_tree_ptr,
      double *lval // Values of A in lower triangle
      ) {

   // Retreive SymbolicTree object from pointer
   auto& symbolic_tree = *static_cast<SymbolicTree*>(symbolic_tree_ptr);

   auto* tree = new NumericTreeUnsymDbl(symbolic_tree, lval);
   return (void *) tree;
}
