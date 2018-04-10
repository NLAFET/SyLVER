#include "testing_form_contrib.hxx"

namespace spldlt { namespace tests {

   int run_form_contrib_tests() {
      
      int nerr = 0;
      
      // TEST(( form_contrib_test<double>(0.01, 1e-20, true, 64, 32, 0, 31, 32) ));
      // TEST(( form_contrib_test<double, true>(0.01, 1e-20, true, 8, 4, 2, 3, 32) ));
      TEST(( form_contrib_test<double>(0.01, 1e-20, true, 64, 32, 10, 19, 32) ));
      TEST(( form_contrib_test<double>(0.01, 1e-20, true, 1000, 500, 101, 300, 64) ));
      TEST(( form_contrib_test<double>(0.01, 1e-20, true, 1000, 500, 101, 300, 128) ));
      TEST(( form_contrib_test<double>(0.01, 1e-20, true, 1000, 500, 101, 300, 256) ));
      TEST(( form_contrib_test<double>(0.01, 1e-20, true, 615, 273, 67, 259, 128) ));
      TEST(( form_contrib_test<double>(0.01, 1e-20, true, 615, 273, 67, 259, 256) ));
      TEST(( form_contrib_test<double>(0.01, 1e-20, true, 615, 273, 67, 259, 512) ));

      return nerr;
   }

}} // end of namespace spldlt::tests
