#include "testing_factor_node_indef.hxx"

namespace spldlt {

   int run_factor_node_indef_tests() {

      int nerr = 0;

      printf("[run_factor_node_indef_tests tests]\n");

      ////////////////////////////////////////////////////////////////////////////////
      // Square matrices 
      // Sequential (1 worker)
      // No delays
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, 32, 32, 32, 1) )); // No blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, 128, 128, 32, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, 128, 128, 128, 1) )); // Inner blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, 1024, 1024, 128, 1) )); // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, 2048, 2048, 256, 1) )); // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, 3000, 3000, 256, 1) )); // Inner and outer blocking

      // Cause delays
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, 32, 32, 32, 1) )); // No blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, 64, 64, 32, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, 64, 64, 64, 1) )); // Inner blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, 1024, 1024, 128, 1) )); // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, 2048, 2048, 256, 1) )); // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, 3000, 3000, 256, 1) )); // Inner and outer blocking

      ////////////////////////////////////////////////////////////////////////////////
      // Parallel (8 worker)
      // No delays
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, 1024, 1024, 128, 8) )); // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, 2048, 2048, 256, 8) )); // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, 3000, 3000, 256, 8) )); // Inner and outer blocking

      // Cause delays
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, 1024, 1024, 128, 8) )); // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, 2048, 2048, 256, 8) )); // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, 3000, 3000, 256, 8) )); // Inner and outer blocking

      return nerr;
   }
}
