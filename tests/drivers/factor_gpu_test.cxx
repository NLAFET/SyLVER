/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

// Sylver tests
#include "testing.hxx"
#include "testing_chol_gpu.hxx"

// STD
#include <iostream>

using namespace sylver::tests;

int main(int argc, char** argv) {

   int ret = 0;
   std::string context = "factor_node_test";
   
   spldlt::SpldltOpts opts;
   opts.parse_opts(argc, argv);

   std::cout << "[" << context << "]" << " Matrix m = " << opts.m << ", n = " << opts.m
             << std::endl;
   
   switch (opts.prec) {
   case sylver::tests::prec::FP32:
      ret = chol_test<float>(opts.m, opts.algo, opts.usetc, (float)opts.cond, opts.itref, (float)opts.tol);
      break;
   case sylver::tests::prec::FP64:
      ret = chol_test<double>(opts.m, opts.algo, opts.usetc, (double)opts.cond, opts.itref, (double)opts.tol);
      break;
   default: std::cout << "[" <<  context << "]" <<  " Requested working precision NOT available" << std::endl;
   }

   // ret = chol_test<double>(opts.m, opts.algo, opts.usetc);
   
   return ret;
}
