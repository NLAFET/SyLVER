// SyLVER
#include "testing.hxx"
#include "testing_factor_node_posdef.hxx"
// STD
#include <cstdio>

using namespace spldlt;
using namespace spldlt::tests;

int main(int argc, char** argv) {

   std::cout << "[Tests][factor_node_posdef] Starting tests.." << std::endl;

   int nerr = 0;

   ////////////////////////////////////////////////////////////
   // Square matrices

   // Sequential, CPU-only

   // No blocking
   TEST(( factor_node_posdef_test<double>(32, 32, 32, 1, 0, true, false) ));
   
   
   if(nerr==0) {
      printf(ANSI_COLOR_BLUE "\n====================================\n"
             ANSI_COLOR_GREEN  "   All tests passed sucessfully\n"
             ANSI_COLOR_BLUE   "====================================\n"
             ANSI_COLOR_RESET);
      return 0;
   } else {
      printf(ANSI_COLOR_BLUE "\n====================================\n"
             ANSI_COLOR_RED    "   %d tests FAILED!\n"
             ANSI_COLOR_BLUE  "====================================\n"
             ANSI_COLOR_RESET, nerr);
      return 1;
   }   
}
