// SpLDLT
#include "kernels/lu_pp.hxx"

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"

namespace spldlt {
   namespace tests {

      /// @param m matrix order
      /// @param k number of rows/columns to be eliminated
      template<typename T>
      int lu_pp_test(int m, int k, bool diagdom, bool check) {

         bool failed = false;

         printf("[lu_pp_test] m = %d\n", m);
         printf("[lu_pp_test] k = %d\n", k);
         printf("[lu_pp_test] diagdom = %d, check = %d\n", diagdom, check);
         ASSERT_TRUE(m >= k);

         // Generate test matrix
         int lda = spral::ssids::cpu::align_lda<T>(m);
         T* a = nullptr;
         T* b = nullptr;

         // Factors
         T* lu = new T[m*lda]; 

         if (check) {
            
            a = new T[m*lda];

            if (diagdom) gen_unsym_diagdom(m, a, lda);
            else         gen_mat(m, m, a, lda);

            b = new T[m];
            gen_unsym_rhs(m, a, lda, b);

            memcpy(lu, a, lda*m*sizeof(T));
         }
         else {

            if (diagdom) gen_unsym_diagdom(m, lu, lda);
            else         gen_mat(m, m, lu, lda);
         }
         
         int *perm = new int[m];
         for (int i=0; i<m; ++i) perm[i] = i;

         // Perform (partial) LU factor with partial pivoting
         lu_pp_factor(m, k, lu, lda);

      }
      
   } // end of namespace tests
} // end of namespace spldlt
