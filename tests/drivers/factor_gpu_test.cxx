/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

// Sylver tests
#include "testing.hxx"
#include "testing_chol_gpu.hxx"

using namespace sylver::tests;

int main(int argc, char** argv) {

   int ret = 0;
   
   printf("[factor_gpu_test]\n");

   spldlt::SpldltOpts opts;
   opts.parse_opts(argc, argv);

   printf("[factor_node_test] Matrix m = %d, n = %d\n", opts.m, opts.m);

   // int m = 4;
   // chol_test<double>(m);
   chol_test<double>(opts.m);
   // chol_test<float>(opts.m);
   
   return ret;
}
