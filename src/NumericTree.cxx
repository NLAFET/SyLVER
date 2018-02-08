#include "ssids/cpu/ThreadStats.hxx"

#include "NumericTree.hxx"

// #include "ssids/cpu/AppendAlloc.hxx"

using namespace spldlt;

// namespace {
//    typedef double T;
//    // const int PAGE_SIZE = 8*1024*1024; // 8MB
//    // typedef NumericTree<T, PAGE_SIZE, spral::ssids::cpu::AppendAlloc<T>, true> NumericTreeDbl; // posdef = true
//    typedef NumericTree<T> NumericTreeDbl;
// }

// Double precision wrapper around templated routines

extern "C"
void *spldlt_create_numeric_tree_dbl(
      void *fkeep,
      void *symbolic_tree_ptr,
      double *aval, // Values of A
      void** child_contrib // Contributions from child subtrees
      ) {

   // Retreive SymbolicTree object from pointer
   auto &symbolic_tree = *static_cast<SymbolicTree*>(symbolic_tree_ptr);

   auto* tree = new NumericTree<double>(fkeep, symbolic_tree, aval, child_contrib);

   return (void *) tree;
}

// extern "C"
// void *spldlt_create_numeric_tree_dbl(
//       void *fkeep,
//       void *symbolic_tree_ptr, 
//       double *aval, // Values of A
//       void** child_contrib, // Contributions from child subtrees
//       int const* exec_loc_aux,
//       struct cpu_factor_options const* options // Options in
//       ) {

//    auto &symbolic_tree = *static_cast<SymbolicTree*>(symbolic_tree_ptr);

//    auto* tree = new NumericTreeDbl(fkeep, symbolic_tree, aval, child_contrib, exec_loc_aux, *options);

//    return (void *) tree;
// }

// // delete tree structure in memory
// extern "C"
// void spldlt_destroy_numeric_tree_dbl(void* target) {

//    auto *tree = static_cast<NumericTreeDbl*>(target);
//    delete tree;
// }

extern "C"
spral::ssids::cpu::Flag 
spldlt_tree_solve_fwd_dbl(void const* tree_ptr, // pointer to relevant type of NumericTree
                          int nrhs,         // number of right-hand sides
                          double* x,        // ldx x nrhs array of right-hand sides
                          int ldx           // leading dimension of x
      ) {
//    // Call method
//    try {
      
//       auto &tree = *static_cast<NumericTreeDbl const*>(tree_ptr); // positive definite!
//       tree.solve_fwd(nrhs, x, ldx);

//    } catch(std::bad_alloc const&) {
//       return Flag::ERROR_ALLOCATION;
//    }
   return spral::ssids::cpu::Flag::SUCCESS;
}

extern "C"
spral::ssids::cpu::Flag
spldlt_tree_solve_bwd_dbl(void const* tree_ptr, // pointer to relevant type of NumericTree
                          int nrhs,         // number of right-hand sides
                          double* x,        // ldx x nrhs array of right-hand sides
                          int ldx           // leading dimension of x
      ) {

//    // Call method
//    try {
//       auto &tree = *static_cast<NumericTreeDbl const*>(tree_ptr); // positive definite!
//       tree.solve_bwd(nrhs, x, ldx);
      
//    } catch(std::bad_alloc const&) {
//       return Flag::ERROR_ALLOCATION;
//    }
   return spral::ssids::cpu::Flag::SUCCESS;
}

extern "C"
spral::ssids::cpu::Flag
spldlt_tree_solve_diag_bwd_dbl(
      void const* tree_ptr, // pointer to relevant type of NumericTree
      int nrhs,         // number of right-hand sides
      double* x,        // ldx x nrhs array of right-hand sides
      int ldx           // leading dimension of x
      ) {

//    // Call method
//    try {
//       auto &tree = *static_cast<NumericTreeDbl const*>(tree_ptr);
//       tree.solve_diag_bwd(nrhs, x, ldx);
//    } catch(std::bad_alloc const&) {
//       return Flag::ERROR_ALLOCATION;
//    }

   return spral::ssids::cpu::Flag::SUCCESS;
}
