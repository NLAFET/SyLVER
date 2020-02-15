// SyLVER
#include "testing.hxx"
#include "testing_form_contrib.hxx"
// STD
#include <cstdio>

using namespace spldlt;
using namespace spldlt::tests;

int main(int argc, char** argv) {

   std::cout << "[Tests][form_contrib] Starting tests.." << std::endl;

   int nerr = 0;

   nerr += run_form_contrib_tests();

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
