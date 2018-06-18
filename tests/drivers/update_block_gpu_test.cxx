// SpLDLT
#include "testing.hxx"
#include "testing_update_block_gpu.hxx"

// STD
#include <cstdio>

using namespace spldlt;
using namespace spldlt::tests;

int main(int argc, char** argv) {

   printf("[update_block_gpu_test]\n");

   spldlt::SpldltOpts opts;
   opts.parse_opts(argc, argv);
   
   printf("[update_block_gpu] Matrix %d x %d x %d\n", opts.m, opts.n, opts.k);
   
   update_block_test<double, false>(opts.m, opts.n, opts.k);
}
