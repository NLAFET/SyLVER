// SSIDS
#include "ssids/cpu/cpu_iface.hxx"

namespace spldlt { namespace tests {

      template<typename T>
      int lu_nopiv_test(int m, int n, bool diagdom, bool check) {

         // Generate test matrix
         int lda = spral::ssids::cpu::align_lda<T>(m);
         T* a = nullptr;
         T* b = nullptr;

         if (check) {
            a = new double[m*lda];
            gen_mat(m, n, a, lda);
         }
      }

   } // end of namespace tests
} // end of namespace spldlt
