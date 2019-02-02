// SyLVER
#include "testing.hxx"
#include "testing_factor_node_indef.hxx"
#include "testing_factor_node_posdef.hxx"
// STD
#include <string>

using namespace spldlt;
using namespace spldlt::tests;

int main(int argc, char** argv) {

   std::string context = "factor_node_test";
   
   spldlt::SpldltOpts opts;
   opts.parse_opts(argc, argv);

   std::cout << "[" <<  context << "]" << " Matrix m = " << opts.m << ", n = " << opts.n << std::endl;
   std::cout << "[" <<  context << "]" << " Number of CPUs = " << opts.ncpu << std::endl;
   printf("[factor_node_test] blksz = %d\n", opts.nb);
   printf("[factor_node_test] ncpu = %d\n", opts.ncpu);
#if defined(SPLDLT_USE_GPU)
   printf("[factor_node_test] ngpu = %d\n", opts.ngpu);
#endif
   printf("[factor_node_test] posdef = %d\n", opts.posdef);
   if (opts.check)
      printf("[factor_node_test] check enabled\n");

   if (opts.chol) {

      switch (opts.prec) {
      case sylver::tests::prec::FP32:
         std::cout << "[" <<  context << "]" << " FP32" << std::endl;
         spldlt::tests::factor_node_posdef_test<float>(opts.m, opts.n, opts.nb, opts.ncpu, opts.ngpu, opts.check, opts.usetc);
         break;
      case sylver::tests::prec::FP64:
         std::cout << "[" <<  context << "]" << " FP64" << std::endl;
         spldlt::tests::factor_node_posdef_test<double>(opts.m, opts.n, opts.nb, opts.ncpu, opts.ngpu, opts.check, opts.usetc);
         break;
      default: std::cout << "[" <<  context << "]" <<  " Requested working precision NOT available" << std::endl;
      }

   }
   else {
      // factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, opts.m, opts.n, opts.nb, opts.ncpu);
      // factor_node_indef_test<double, 32, false>(0.01, 1e-20, opts.posdef, true, false, opts.m, opts.n, opts.nb, opts.ncpu);
      // No delays
      spldlt::tests::factor_node_indef_test<double, 32, false>(0.01, 1e-20, opts.posdef, false, false, opts.m, opts.n, opts.nb, opts.ncpu, opts.ngpu);   
   }
}

