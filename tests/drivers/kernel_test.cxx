// SpLDLT
#include "testing.hxx"
#include "testing_factor_node_indef.hxx"
#include "testing_form_contrib.hxx"

#include <cstdio>

// SSIDS
#include "tests/ssids/kernels/framework.hxx"
#include "ssids/cpu/kernels/wrappers.hxx"

using namespace spldlt;
using namespace spldlt::tests;

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

   // printf("Matrix %dx%d\n", opts.m, opts.n);
   // printf("Blocking, nb: %d, ib:%d\n", opts.nb, opts.ib);
   // printf("No CPU: %d\n", opts.ncpu);

   // factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, opts.m, opts.n, opts.nb, opts.ncpu);
   // factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, opts.m, opts.n, opts.nb, opts.ncpu);

   // factor_node_indef_test<double, 32, true>(0.01, 1e-20, false, false, 8, 8, 8, 1);
   // factor_node_indef_test<double, 32, true>(0.01, 1e-20, false, false, 8, 8, 8, 1);
   
   ////////////////////////////////////////////////////////////////////////////////
   // Square matrices 
   // Sequential (1 worker)
   // No delays
   // factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, 32, 32, 32, 1);
   // factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, 64, 64, 32, 1); // Outer blocking
   // factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, 64, 64, 64, 1); // Inner blocking
   // factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, 1024, 1024, 128, 1); // Inner and outer blocking

   // Cause delays
   // factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, 32, 32, 32, 1);
   // factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, 64, 64, 32, 1);
   int nerr = 0;
   
   // nerr += run_factor_node_indef_tests();
   nerr += run_form_contrib_tests();

   if(nerr==0) {
      printf(ANSI_COLOR_BLUE "\n====================================\n"
             ANSI_COLOR_GREEN  "   All tests passed sucessfully\n"
             ANSI_COLOR_BLUE   "====================================\n"
             ANSI_COLOR_RESET);
   } else {
      printf(ANSI_COLOR_BLUE "\n====================================\n"
             ANSI_COLOR_RED    "   %d tests FAILED!\n"
             ANSI_COLOR_BLUE  "====================================\n"
             ANSI_COLOR_RESET, nerr);
   }

}
