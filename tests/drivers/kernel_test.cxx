// SpLDLT
#include "testing.hxx"
#include "testing_factor_node_indef.hxx"
#include "testing_form_contrib.hxx"
#include "testing_lu_nopiv.hxx"
#include "testing_lu_pp.hxx"
// #include "testing_lu_tpp.hxx"
#include "testing_factor_node_unsym.hxx"

#include <cstdio>

// SSIDS
#include "ssids/cpu/kernels/wrappers.hxx"
// SSIDS test
#include "tests/ssids/kernels/framework.hxx"
#include "tests/ssids/kernels/AlignedAllocator.hxx"

using namespace spldlt;
using namespace spldlt::tests;

int main(int argc, char** argv) {
   
   printf("[Tests] Test kernels\n");

   int nerr = 0;
      
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
   
   ////////////////////////////////////////////////////////////
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

   ////////////////////////////////////////////////////////////
   // Unsym matrices (no piv)
   // square matrices (m=k)
   // lu_nopiv_test<double>(32, 32, true, true); // Diagonally dominant
   // lu_nopiv_test<double>(128, 128, true, true);
   // lu_nopiv_test<double>(32, 32, false, true); // General matrix
   // lu_nopiv_test<double>(128, 128, false, true);
   // lu_nopiv_test<double>(256, 256, false, true);

   // Unsym matrices (partial pivoting)
   // square matrices (m=k)
   // lu_pp_test<double>(8, 8, true, true); // Diagonally dominant
   // lu_pp_test<double>(16, 16, true, true); // Diagonally dominant
   // lu_pp_test<double>(32, 32, true, true); // Diagonally dominant
   // lu_pp_test<double>(128, 128, true, true); // Diagonally dominant
   // lu_pp_test<double>(32, 32, false, true); // General matrix
   // lu_pp_test<double>(128, 128, false, true); // General matrix
   // lu_pp_test<double>(256, 256, false, true); // General matrix

   // Unsym matrices (threshold partial pivoting)
   // lu_tpp_test<double>(0.01, 8, 8, true, true); // Diagonally dominant

   ////////////////////////////////////////////////////////////
   // Unsym front (restricted pivoting)
   struct spral::ssids::cpu::cpu_factor_options options;

   // Restricted pivoting
   options.pivot_method = PivotMethod::app_aggressive;

   // options.cpu_block_size = 4;
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 4, 4, 1, 0, true, true); // Diagonally dominant
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 4, 4, 1, 0, false, true); // Diagonally dominant
   // options.cpu_block_size = 8;
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 8, 8, 1, 0, true, true); // Diagonally dominant
   // options.cpu_block_size = 4;
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 8, 8, 1, 0, true, true); // Diagonally dominant
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 16, 16, 1, 0, true, true); // Diagonally dominant

   // options.cpu_block_size = 8;
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 20, 20, 1, 0, true, true); // Diagonally dominant

   // Threshold partial pivoting
   options.pivot_method = PivotMethod::app_block;

   // options.u = 0.01; // Theshold parameter   
   // options.cpu_block_size = 4;
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 4, 4, 1, 0, true, true); // Diagonally dominant
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 8, 8, 1, 0, true, true); // Diagonally dominant
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 16, 16, 1, 0, true, true); // Diagonally dominant

   // options.cpu_block_size = 2;

   // options.u = 1.0; // Theshold parameter   
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 4, 4, 1, 0, true, true); // Diagonally dominant
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 4, 4, 1, 0, false, true);

   // options.u = 0.5; // Theshold parameter   
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 6, 6, 1, 0, false, true);
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 5, 5, 1, 0, false, true);

   // options.u = 1.0; // Theshold parameter   
   // options.cpu_block_size = 4;
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 8, 8, 1, 0, false, true);

   options.u = 0.5; // Theshold parameter
   options.cpu_block_size = 4;
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 12, 12, 1, 0, false, true);
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 16, 16, 1, 0, false, true);
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 6, 6, 1, 0, false, true);
   factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 10, 10, 1, 0, false, true);

   // options.u = 0.1; // Theshold parameter
   // options.cpu_block_size = 4;
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 12, 12, 1, 0, false, true);

   // options.u = 0.5; // Theshold parameter
   // options.cpu_block_size = 8;
   // factor_node_unsym_test<double, spral::test::AlignedAllocator<double>>(options, 24, 24, 1, 0, false, true);

   // nerr += run_factor_node_indef_tests();
   // nerr += run_form_contrib_tests();

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
