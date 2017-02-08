#include "testing_factor_node_indef.hxx"

namespace spldlt {

   int run_factor_node_indef_tests() {

      int err = 0;

      printf("[FactorNodeIndef tests]\n");

      /* 10x3 matrix
         blksz: 10
         inner blksz: 5
         debug: enabled
      */
      // factor_node_indef_test<double, 5, true>(0.01, 1e-20, true, false, 10, 3, 5);

      /* 12x12 matrix
         blksz: 12
         inner blksz: 4
         debug: enabled
      */
      // factor_node_indef_test<double, 4, true>(0.01, 1e-20, true, false, 12, 12, 4);

      // factor_node_indef_test<double, 4, true>(0.01, 1e-20, true, false, 8, 8, 4);

      factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, 512, 512, 128, 1);

      // factor_node_indef_test<double, 32, false>(0.01, 1e-20, true, false, 2048, 2048, 512);

      return err;
   }
}
