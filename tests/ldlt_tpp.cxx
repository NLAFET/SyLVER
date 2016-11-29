#include "common.hxx"
#include "ldlt_tpp.hxx"

#include <cstring>

#include "tests/ssids/kernels/framework.hxx"
#include "ssids/cpu/kernels/wrappers.hxx"

using namespace spral::ssids::cpu;

namespace spldlt {

   template<typename T,
      bool debug=true>
   int ldlt_tpp_test(T u, T small, bool delays, bool singular, int m, int n, int test=0, int seed=0) {
      bool failed = false;
      bool action = true; // Don't abort on singular matrices
      // Note: We generate an m x m test matrix, then factor it as an
      // m x n matrix followed by an (m-n) x (m-n) matrix [give or take delays]

      if (debug) printf("[ldlt_tpp_test] %d x %d\n", m, n);
   
      // Generate test matrix
      int lda = m;
      T* a = new double[m*lda];
      gen_sym_indef(m, a, lda);
      // gen_posdef(m, a, lda);
      modify_test_matrix(singular, delays, m, n, a, lda);

      // Generate a RHS based on x=1, b=Ax
      T *b = new double[m];
      gen_rhs(m, a, lda, b);

      // Factorize using main routine
      // Allocate factors
      T *l = new double[m*lda];
      memcpy(l, a, m*lda*sizeof(double)); // Copy a to l
      // Set up permutation vector
      int *perm = new int[m];
      for(int i=0; i<m; i++) perm[i] = i;
      // Allocate diagonal and workspace
      T *d = new double[2*m];
      T *work = new double[2*m];
      
      // First factorize m x n matrix
      int q1 = ldlt_tpp_factor(m, n, perm, l, lda, d, work, m, action, u, small);
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
      // Make sure all entries in L are bounded by 1/u
      T l_abs_max = find_l_abs_max(m, l, lda);
      EXPECT_LE(l_abs_max, 1.0/u) << "(test " << test << " seed " << seed << ")" << std::endl;

      if (debug) {
         std::cout << "q1=" << q1 << " q2=" << q2 << std::endl;
         std::cout << "L:" << std::endl;
         // Print info
         // print_mat("%10.2e", m, l, lda, perm);
         print_mat(m, perm, l, lda);
         std::cout << "D:" << std::endl;
         print_d(m, d);
      }

      // Perform solve
      T *soln = new double[m];
      solve(m, q1, perm, l, lda, d, b, soln);

      // Print info
      printf("soln = ");
      for(int i=0; i<m; i++) printf(" %le", soln[i]);
      printf("\n");

      T bwderr = backward_error(m, a, lda, b, 1, soln, m);

      // Print info
      if (debug) printf("bwderr = %le\n", bwderr);
      // Make sure that the bwd error is small
      EXPECT_LE(bwderr, 2e-13) << "(test " << test << " seed " << seed << ")" << std::endl;

      // Cleanup memory
      delete[] a; delete[] l;
      delete[] b;
      delete[] perm;
      delete[] work;
      delete[] d; delete[] soln;

      return failed ? -1 : 0;
   }

   int run_ldlt_tpp_tests() {

      int err = 0;

      printf("[LDLT TPP tests]\n");
      // arguments: u, small, m, n
      // run_ldlt_tpp_test(0.01, 1e-20, true, false, 10, 10);
      // run_ldlt_tpp_test(0.01, 1e-20, true, false, 10, 5);
      err = ldlt_tpp_test<double, true>(0.01, 1e-20, true, false, 10, 3);

      return err;
   }
}
