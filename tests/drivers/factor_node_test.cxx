// SpLDLT
#include "testing.hxx"
#include "testing_factor_node_indef.hxx"

using namespace spldlt;
using namespace spldlt::tests;

int main(int argc, char** argv) {

   printf("[factor_node_test]\n");

   spldlt::SpldltOpts opts;
   opts.parse_opts(argc, argv);

   printf("[factor_node_test] Matrix %dx%d\n", opts.m, opts.n);
   printf("[factor_node_test] blksz = %d, ib:%d\n", opts.nb, opts.ib);
   printf("[factor_node_test] ncpu = %d\n", opts.ncpu);
   printf("[factor_node_test] posdef = %d\n", opts.posdef);

   // factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, opts.m, opts.n, opts.nb, opts.ncpu);
   // No delays
   factor_node_indef_test<double, 32, false>(0.01, 1e-20, opts.posdef, false, false, opts.m, opts.n, opts.nb, opts.ncpu);   
}
