#include "ldlt_app.hxx" 

#include "common.hxx"

#include "ssids/cpu/kernels/ldlt_app.hxx"
// #include "ssids/cpu/kernels/ldlt_app.cxx" // .cxx as we need internal namespace
#include "ssids/cpu/cpu_iface.hxx"

using namespace spral::ssids::cpu;

namespace spldlt {

   static const int INNER_BLOCK_SIZE = 32; // same as in ssids/cpu/kernels/ldlt_app.cxx 
   /*
     bool dblk_singular singular diagonal blocks
    */
   template<typename T, 
            bool debug=true>
   int ldlt_app_test(T u, T small, 
                     bool delays, bool singular, bool dblk_singular, 
                     int m, int n, int outer_block_size=INNER_BLOCK_SIZE,
                     int test=0, int seed=0) {

      bool failed = false;

      // Note: We generate an m x m test matrix, then factor it as an
      // m x n matrix followed by an (m-n) x (m-n) matrix [give or take delays]

      // Generate test matrix
      int lda = align_lda<T>(m);
      double* a = new T[m*lda];
      gen_sym_indef(m, a, lda);
      modify_test_matrix<T, INNER_BLOCK_SIZE>(
            singular, delays, dblk_singular, m, n, a, lda
            );

      return failed ? -1 : 0;
   }

   int run_ldlt_app_tests() {

      int err = 0;

      printf("[LDLT APP tests]\n");
      
      // 10x10 matrix
      // delays=true, sigular=false, dblk_singular=fasle
      ldlt_app_test<double, true>(0.01, 1e-20, true, false, false, 10, 10);
      
      return err;
   }

}
