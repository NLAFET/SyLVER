#include "testing.hxx"

#include <cstdio>
// #include "ssids/cpu/kernels/ldlt_app.cxx" // .cxx as we need internal namespace

#include "tests/ssids/kernels/framework.hxx"
#include "ssids/cpu/kernels/wrappers.hxx"
// #include "ldlt_tpp.hxx"
// #include "ldlt_app.hxx"
#include "testing_factor_indef.hxx"
#include "testing_factor_node_indef.hxx"

// using namespace spral::ssids::cpu;
using namespace spldlt;

int main(int argc, char** argv) {
   
   printf("[Tests] Test kernels\n");

   // int m = 128;
   // int lda = m;
   // double *a = new double[m*lda];

   // gen_posdef(m, a, lda);

   // printf("[Tests] Test LDLT kernels\n");
   // spldlt::run_ldlt_tpp_tests();

   // Factorize node in indefinite case using APTP (sequtential)
   // spldlt::run_factor_indef_tests();

   // Factorize node in indefinite case using APTP (parallel)
   // spldlt::run_factor_node_indef_tests();

   spldlt::SpldltOpts opts;
   opts.parse_opts(argc, argv);

   printf("Matrix %dx%d\n", opts.m, opts.n);
   printf("Blocking, nb: %d, ib:%d\n", opts.nb, opts.ib);
   printf("No cpu: %d\n", opts.ncpu);

   factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, opts.m, opts.n, opts.nb, opts.ncpu);
}
