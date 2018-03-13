#include "testing_factor_node_indef_app.hxx"

namespace spldlt { namespace tests {

      int run_factor_node_indef_app_tests() {

         int nerr = 0;
         
         printf("[run_factor_node_indef_app_tests tests]\n");
         
         TEST(( factor_node_indef_app_test<double, 32, true>(0.01, 1e-20, false, false, 2, 1, 1, 1) ));

         return nerr;      
      }

   }} // namespace spldlt::tests
