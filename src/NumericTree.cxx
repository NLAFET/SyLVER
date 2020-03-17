/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

#include "sylver_ciface.hxx"
#include "NumericTree.hxx"
#include "AppendAlloc.hxx"

#include <memory>

using namespace sylver;
using namespace ::spldlt; // TODO change to sylver::spldlt namespace

namespace {
   typedef double T;
   const int PAGE_SIZE = 8*1024*1024; // 8MB
   // Stack allocator
   typedef ::spldlt::NumericTree<T, PAGE_SIZE, sylver::AppendAlloc<T>, true> NumericTreePosdefDbl; // posdef = true
   typedef ::spldlt::NumericTree<T, PAGE_SIZE, sylver::AppendAlloc<T>, false> NumericTreeIndefDbl; // posdef = false
   // Default allocator 
   // typedef NumericTree<T, PAGE_SIZE, std::allocator<T>, true> NumericTreePosdefDbl; // posdef = true
   // typedef NumericTree<T, PAGE_SIZE, std::allocator<T>, false> NumericTreeIndefDbl; // posdef = false
}

// Double precision wrapper around templated routines

extern "C"
void *spldlt_create_numeric_tree_dbl(
      bool posdef,
      void *fkeep,
      void *symbolic_tree_ptr,
      double *aval, // Values of A
      const double *const scaling, // Scaling vector (NULL if none)
      void** child_contrib, // Contributions from child subtrees
      sylver::options_t *options, // Options in
      sylver::inform_t* stats // Info out
      ) {

   // Retreive SymbolicTree object from pointer
   auto &symbolic_tree = *static_cast<sylver::SymbolicTree*>(symbolic_tree_ptr);

   if (posdef) {
      auto* tree = new NumericTreePosdefDbl
         (fkeep, symbolic_tree, aval, const_cast<double*>(scaling), child_contrib, *options, *stats);
      return (void *) tree;
   }
   else {
      auto* tree = new NumericTreeIndefDbl
         (fkeep, symbolic_tree, aval, const_cast<double*>(scaling), child_contrib, *options, *stats);
      return (void *) tree;
   }
}

// delete tree structure in memory
extern "C"
void spldlt_destroy_numeric_tree_dbl(bool posdef, void* target) {

   if (posdef) {
      auto *tree = static_cast<NumericTreePosdefDbl*>(target);
      delete tree;
   }
   else {
      auto *tree = static_cast<NumericTreeIndefDbl*>(target);
      delete tree;
   }
}

extern "C"
sylver::Flag 
spldlt_tree_solve_fwd_dbl(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void const* tree_ptr, // pointer to relevant type of NumericTree
      int nrhs,         // number of right-hand sides
      double* x,        // ldx x nrhs array of right-hand sides
      int ldx           // leading dimension of x
      ) {
   // Call method
   try {

      if (posdef) {
         auto &tree = *static_cast<NumericTreePosdefDbl const*>(tree_ptr); // positive definite!
         tree.solve_fwd(nrhs, x, ldx);
      } else {
         auto &tree = *static_cast<NumericTreeIndefDbl const*>(tree_ptr); // positive definite!
         tree.solve_fwd(nrhs, x, ldx);
      }

   } catch(std::bad_alloc const&) {
      return sylver::Flag::ERROR_ALLOCATION;
   }
   return sylver::Flag::SUCCESS;
}

extern "C"
sylver::Flag
spldlt_tree_solve_bwd_dbl(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void const* tree_ptr, // pointer to relevant type of NumericTree
      int nrhs,         // number of right-hand sides
      double* x,        // ldx x nrhs array of right-hand sides
      int ldx           // leading dimension of x
      ) {

   // Call method
   try {

      if (posdef) {
         auto &tree = *static_cast<NumericTreePosdefDbl const*>(tree_ptr); // positive definite!
         tree.solve_bwd(nrhs, x, ldx);
      } else {
         auto &tree = *static_cast<NumericTreeIndefDbl const*>(tree_ptr); // positive definite!
         tree.solve_bwd(nrhs, x, ldx);
      }
      
   } catch(std::bad_alloc const&) {
      return sylver::Flag::ERROR_ALLOCATION;
   }
   return sylver::Flag::SUCCESS;
}

extern "C"
sylver::Flag
spldlt_tree_solve_diag_bwd_dbl(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void const* tree_ptr, // pointer to relevant type of NumericTree
      int nrhs,         // number of right-hand sides
      double* x,        // ldx x nrhs array of right-hand sides
      int ldx           // leading dimension of x
      ) {

   // Call method
   try {
      if (posdef) {
         auto &tree = *static_cast<NumericTreePosdefDbl const*>(tree_ptr);
         tree.solve_diag_bwd(nrhs, x, ldx);
      } else {
         auto &tree = *static_cast<NumericTreeIndefDbl const*>(tree_ptr);
         tree.solve_diag_bwd(nrhs, x, ldx);
      }
   } catch(std::bad_alloc const&) {
      return sylver::Flag::ERROR_ALLOCATION;
   }
   return sylver::Flag::SUCCESS;
}


extern "C"
sylver::Flag
spldlt_tree_solve_diag_dbl(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void const* tree_ptr, // pointer to relevant type of NumericTree
      int nrhs,         // number of right-hand sides
      double* x,        // ldx x nrhs array of right-hand sides
      int ldx           // leading dimension of x
      ) {

   // Call method
   try {
      if (posdef) {
         auto &tree = *static_cast<NumericTreePosdefDbl const*>(tree_ptr);
         tree.solve_diag(nrhs, x, ldx);
      } else {
         auto &tree = *static_cast<NumericTreeIndefDbl const*>(tree_ptr);
         tree.solve_diag(nrhs, x, ldx);
      }
   } catch(std::bad_alloc const&) {
      return sylver::Flag::ERROR_ALLOCATION;
   }
   return sylver::Flag::SUCCESS;
}
