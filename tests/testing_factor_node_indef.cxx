#include "testing_factor_node_indef.hxx"

namespace spldlt {
namespace tests {

   /// @brief Performs tests for factor_node_indef kernel in
   /// sequential
   int run_factor_node_indef_tests_seq() {

      int nerr = 0;
         
      // Sequential (1 worker)

      ////////////////////////////////////////////////////////////
      // Square matrices

      // No delays

      // No blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 32, 32, 32, 1) ));
      // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 128, 128, 32, 1) ));
      // Inner blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 128, 128, 128, 1) ));
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1024, 1024, 128, 1) ));
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 2048, 2048, 256, 1) ));
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 3000, 3000, 256, 1) ));
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 5179, 5179, 512, 1) ));
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 4085, 4085, 315, 1) ));

      // Cause delays

      // No blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 32, 32, 32, 1) ));
      // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 64, 64, 32, 1) ));
      // Inner blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 64, 64, 64, 1) ));
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 1024, 1024, 128, 1) ));
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 2048, 2048, 256, 1) ));
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 3000, 3000, 256, 1) ));

      ////////////////////////////////////////////////////////////////////////////////
      // Rectangular matrices
      // No delays
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 2, 1, 1, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 8, 4, 4, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 128, 64, 32, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1000, 200, 32, 1) )); // Outer blocking      
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 512, 256, 64, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 64, 32, 16, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 128, 64, 32, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1000, 500, 250, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1092, 451, 123, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1061, 419, 100, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1061, 419, 400, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 2000, 500, 128, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 3000, 500, 256, 1) )); // Outer blocking

      // Cause delays
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 2, 1, 1, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 4, 2, 2, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 8, 4, 4, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 16, 8, 4, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 64, 32, 16, 1) )); // Outer blocking with blksz < iblksz
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 128, 64, 32, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 2048, 1024, 128, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 2000, 1000, 128, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 1000, 500, 250, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 1092, 451, 123, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 2000, 500, 128, 1) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 3000, 500, 256, 1) )); // Outer blocking

      return nerr;
   }

   int run_factor_node_indef_tests_par() {

      int nerr = 0;

      // Parallel (8 worker)

      ////////////////////////////////////////////////////////////////////////////////
      // Square matrices
      
      // No delays
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1024, 1024, 128, 8) ));
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 2048, 2048, 256, 8) ));
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 3000, 3000, 256, 8) ));
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 5179, 5179, 512, 8) ));
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 4085, 4085, 315, 8) ));

      // Cause delays
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 1024, 1024, 128, 8) )); 
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 2048, 2048, 256, 8) )); // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 3000, 3000, 256, 8) )); // Inner and outer blocking

      ////////////////////////////////////////////////////////////////////////////////
      // Rectangular matrices

      // No delays
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 2, 1, 1, 8) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 8, 4, 4, 8) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 128, 64, 32, 8) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1000, 200, 32, 8) )); // Outer blocking      
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 512, 256, 64, 8) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 64, 32, 16, 8) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 128, 64, 32, 8) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1000, 500, 250, 8) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1092, 451, 123, 8) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 2000, 500, 128, 8) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 3000, 500, 256, 8) )); // Outer blocking

      // Cause delays
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 2, 1, 1, 8) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 8, 4, 4, 8) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 128, 64, 32, 8) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 1000, 200, 32, 8) )); // Outer blocking      
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 512, 256, 64, 8) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 64, 32, 16, 8) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 128, 64, 32, 8) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 1000, 500, 250, 8) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 1092, 451, 123, 8) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 2000, 500, 128, 8) )); // Outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 3000, 500, 256, 8) )); // Outer blocking

      return nerr;

   }

#if defined(SPLDLT_USE_GPU)
   int run_factor_node_indef_tests_par_cuda() {

      int nerr = 0;

      // Parallel (8 worker + 1 CUDA device)

      ////////////////////////////////////////////////////////////
      // Square matrices

      // No delays
      // Inner and outer blocking
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 500, 500, 32, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 500, 500, 64, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1000, 1000, 128, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1000, 1000, 256, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 2000, 2000, 32, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 2000, 2000, 64, 8, 1) ));

      // Cause delays
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 500, 500, 32, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 500, 500, 64, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 1000, 1000, 128, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 1000, 1000, 256, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 2000, 2000, 32, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 2000, 2000, 64, 8, 1) ));

      ////////////////////////////////////////////////////////////
      // Trapezoidal matrices

      // No delays
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 2000, 100, 32, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 2000, 100, 64, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1578, 195, 64, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1963, 582, 128, 8, 1) ));

      // Cause delays
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 2000, 100, 32, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, true, false, 2000, 100, 64, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1578, 195, 64, 8, 1) ));
      TEST(( factor_node_indef_test<double, 32, false>(0.01, 1e-20, false, false, false, 1963, 582, 128, 8, 1) ));

      return nerr;

   }
#endif
   
   /// @brief Performs all the tests for factor_node_indef kernel
   int run_factor_node_indef_tests() {

      int nerr = 0;

      printf("[run_factor_node_indef_tests tests]\n");

      nerr += run_factor_node_indef_tests_seq(); // Sequential tests
      nerr += run_factor_node_indef_tests_par(); // Parallel tests
#if defined(SPLDLT_USE_GPU)
      nerr += run_factor_node_indef_tests_par_cuda(); // Parallel tests with CUDA enabled
#endif
      
      return nerr;
   }

}} // namespace spldlt::tests
