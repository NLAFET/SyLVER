// SSIDS
#include "ssids/cpu/cpu_iface.hxx"

namespace spldlt {
   namespace tests {

      template<typename T>
      int lu_nopiv_test(int m, int n, bool diagdom, bool check) {

         // Generate test matrix
         int lda = spral::ssids::cpu::align_lda<T>(m);
         T* a = nullptr;
         T* b = nullptr;

         if (check) {
            a = new T[n*lda];

            if (diagdom) gen_unsym_diagdom(m, n, a, lda);
            else         gen_mat(m, n, a, lda);

            b = new T[m];
            gen_unsym_rhs(m, n, a, lda, b);
         }

         T* lu = nullptr;

         if (check) {
            memcpy(lu, a, lda*n*sizeof(T));
         }
         else {
            lu = new T[n*lda];

            if (diagdom) gen_unsym_diagdom(m, n, lu, lda);
            else         gen_mat(m, n, lu, lda);
         }



         if (check) {

            int nrhs = 1;
            int ldsoln = m;
            T *soln = new T[nrhs*ldsoln];

            double bwderr = unsym_backward_error(
                  m, n, a, lda, b, nrhs, soln, ldsoln);
         }

         if (check) {
            delete[] a;
            delete[] b;
         }
         else {
            delete[] lu;
         }

      }

   } // end of namespace tests
} // end of namespace spldlt
