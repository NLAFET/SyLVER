#pragma once

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"

namespace spldlt { 
   namespace tests {

      template<typename T>
      int factor_node_unsym_test(
            struct spral::ssids::cpu::cpu_factor_options& options, int m, int n, 
            int blksz, int ncpu, int ngpu=0) {

         bool failed = false;

         // We don't allow these cases
         ASSERT_TRUE(n > 0);
         ASSERT_TRUE(m > 0);
         ASSERT_TRUE(ncpu > 0);
         ASSERT_TRUE(ngpu >= 0);

         ASSERT_TRUE(m >= n);

         // Generate test matrix
         int lda = spral::ssids::cpu::align_lda<T>(m);
         T* a = new double[m*lda];

         
         return failed ? -1 : 0;
      }

   } // namespace tests
} // namespace spldlt
