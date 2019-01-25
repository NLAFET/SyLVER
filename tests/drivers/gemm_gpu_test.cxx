/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

// Sylver tests
#include "testing.hxx"
#include "testing_gemm_gpu.hxx"

// STD
#include <iostream>

using namespace sylver::tests;

int main(int argc, char** argv) {

   int ret = 0;
   
   printf("[factor_gpu_test]\n");

   spldlt::SpldltOpts opts;
   opts.parse_opts(argc, argv);

   std::cout << "[factor_node_test] Matrix m = " << opts.m << ", n = " << opts.m
             << std::endl;
   
   // ret = gemm_test<double>(opts.m, opts.n, opts.k, opts.algo);
   ret = gemm_test<float>(opts.m, opts.n, opts.k, opts.algo, opts.usetc);
   
   return ret;
}
