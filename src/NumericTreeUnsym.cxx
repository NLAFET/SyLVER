/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#include "NumericTreeUnsym.hxx"
#include "sylver_ciface.hxx"
#include "AppendAlloc.hxx"

using namespace spldlt; // TODO remove or change to sylver namespace if necessary

namespace {
   typedef double T; // Working precision
   typedef spldlt::AppendAlloc<T> FactorAlloc; // Factor allocator
   const int PAGE_SIZE = 8*1024*1024; // 8MB
   typedef sylver::splu::NumericTreeUnsym<T, FactorAlloc, true, PAGE_SIZE> NumericTreeUnsymDiagdomDbl; 
}

extern "C"
void* splu_create_numeric_tree_dbl(
      void *symbolic_tree_ptr, // C pointer to symbolic tree stucture
      double *val, // Values of A
      double *scaling, // Scaling vector (NULL if none)
      struct sylver::options_c *options // Input options
      ) {

   // Retreive SymbolicTree object from pointer
   auto& symbolic_tree = *static_cast<SymbolicTree*>(symbolic_tree_ptr);

   auto* tree = new NumericTreeUnsymDiagdomDbl(symbolic_tree, val, scaling, *options);
   return (void *) tree;
}
