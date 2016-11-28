#include "ldlt_tpp.hxx"

#include <cstring>

#include "tests/ssids/kernels/framework.hxx"
#include "ssids/cpu/kernels/wrappers.hxx"
#include "ssids/cpu/kernels/ldlt_tpp.hxx"

using namespace spral::ssids::cpu;

namespace spldlt {

   double find_l_abs_max(int n, double *a, int lda) {
      double best = 0.0;
      for(int c=0; c<n; ++c)
         for(int r=c; r<n; ++r)
            best = std::max(best, std::abs(a[c*lda+r]));
      return best;
   }

   int run_ldlt_tpp_test(double u, double small, int m, int n, int test=0, int seed=0) {
      bool failed = false;
      bool action = true; // Don't abort on singular matrices
      // Note: We generate an m x m test matrix, then factor it as an
      // m x n matrix followed by an (m-n) x (m-n) matrix [give or take delays]

      printf("[ldlt_tpp_test] %d x %d\n", m, n);
   
      // Generate test matrix
      int lda = m;
      double* a = new double[m*lda];
      gen_sym_indef(m, a, lda);
      // gen_posdef(m, a, lda);

      // Generate a RHS based on x=1, b=Ax
      double *b = new double[m];
      gen_rhs(m, a, lda, b);

      // Factorize using main routine
      // Allocate factors
      double *l = new double[m*lda];
      memcpy(l, a, m*lda*sizeof(double)); // Copy a to l
      // Set up permutation vector
      int *perm = new int[m];
      for(int i=0; i<m; i++) perm[i] = i;
      // Allocate diagonal and workspace
      double *d = new double[2*m];
      double *work = new double[2*m];
      
      // First factorize m x n matrix
      int q1 = ldlt_tpp_factor(m, n, perm, l, lda, d, work, m, action, u, small);
      printf("[ldlt_tpp_test] q1: %d\n", q1);
      // Update and factorize trailing matrix
      int q2 = 0;
      if(m > n) {
         // Apply outer product update
         do_update(m-n, q1, &l[n*(lda+1)], &l[n], lda, d);
         // Second (m-n) x (m-n) matrix [but add delays if any]
         q2 = ldlt_tpp_factor(m-q1, m-q1, &perm[q1], &l[q1*(lda+1)], lda, &d[2*q1], work, m, action, u, small, q1, &l[q1], lda);
      }
      // Check we eliminated all the columns
      EXPECT_EQ(m, q1+q2) << "(test " << test << " seed " << seed << ")" << std::endl;
      printf("[ldlt_tpp_test] q1+q2: %d\n", q1+q2);
      // Make sure all entries in L are bounded by 1/u
      double l_abs_max = find_l_abs_max(m, l, lda);
      EXPECT_LE(l_abs_max, 1.0/u) << "(test " << test << " seed " << seed << ")" << std::endl;
      printf("nrom(L, max): %e\n", l_abs_max);

      std::cout << "q1=" << q1 << " q2=" << q2 << std::endl;
      std::cout << "L:" << std::endl;
      // print_mat("%10.2e", m, l, lda, perm);
      print_mat(m, perm, l, lda);
      std::cout << "D:" << std::endl;
      print_d(m, d);
   }

   int run_ldlt_tpp_tests() {

      int err = 0;

      // arguments: u, small, m, n
      run_ldlt_tpp_test(0.01, 1e-20, 10, 10);

      return err;
   }
}
