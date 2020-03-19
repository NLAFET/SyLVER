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

extern "C"
sylver::Flag 
spldlt_tree_solve_fwd_posdef_dbl(
      void const* tree_ptr, // pointer to relevant type of NumericTree
      int nrhs,         // number of right-hand sides
      double* x,        // ldx x nrhs array of right-hand sides
      int ldx           // leading dimension of x
      ) {
   // Call method
   try {

      auto &tree = *static_cast<NumericTreePosdefDbl const*>(tree_ptr); // positive definite!
      tree.solve_fwd(nrhs, x, ldx);

   } catch(std::bad_alloc const&) {
      return sylver::Flag::ERROR_ALLOCATION;
   }
   return sylver::Flag::SUCCESS;
}

extern "C"
sylver::Flag
spldlt_tree_solve_bwd_posdef_dbl(
      void const* tree_ptr, // pointer to relevant type of NumericTree
      int nrhs,         // number of right-hand sides
      double* x,        // ldx x nrhs array of right-hand sides
      int ldx           // leading dimension of x
      ) {

   // Call method
   try {

      auto &tree = *static_cast<NumericTreePosdefDbl const*>(tree_ptr); // positive definite!
      tree.solve_bwd(nrhs, x, ldx);
      
   } catch(std::bad_alloc const&) {
      return sylver::Flag::ERROR_ALLOCATION;
   }
   return sylver::Flag::SUCCESS;
}
