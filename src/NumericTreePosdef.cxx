/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

#include "AppendAlloc.hxx"
#include "NumericTreePosdef.hxx"

#include <memory>

using namespace sylver;
using namespace sylver::spldlt; // TODO change to sylver::spldlt namespace

namespace {
   int const PAGE_SIZE = 8*1024*1024; // 8MB
   typedef sylver::spldlt::NumericTreePosdef<double, PAGE_SIZE, sylver::AppendAlloc<double>> NumericTreePosdefDbl;
}
   
extern "C"
void *spldlt_create_numeric_tree_posdef_dbl(
      void *fkeep, // Factor info structure (Fortran)
      void *symbolic_tree_ptr, // Symbolic tree info structure (C++)
      double *aval, // Values of A
      const double *const scaling, // Scaling vector (NULL if none)
      void** child_contrib, // Contributions from child subtrees
      sylver::options_t *options, // Options in
      sylver::inform_t* stats // Info out
      ) {

   // Retreive SymbolicTree object from pointer
   auto &symbolic_tree = *static_cast<sylver::SymbolicTree*>(symbolic_tree_ptr);

   auto* numeric_tree = new NumericTreePosdefDbl
      (fkeep, symbolic_tree, aval, const_cast<double*>(scaling), child_contrib, *options, *stats);
   return (void *) numeric_tree;
}

// delete tree structure in memory
extern "C"
void spldlt_destroy_numeric_tree_posdef_dbl(void* target) {
   auto *tree = static_cast<NumericTreePosdefDbl*>(target);
   delete tree;
}
