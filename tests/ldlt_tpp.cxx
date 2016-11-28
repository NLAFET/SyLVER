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

   // Makes a (symmetric, half storage) matrix singular by making col2 an
   // appropriate multiple of col1
   template <typename T>
   void make_singular(int n, int col1, int col2, T *a, int lda) {
      T *col = new T[n];
   
      T a11 = a[col1*(lda+1)];
      T a21 = (col1 < col2) ? a[col1*lda + col2]
         : a[col2*lda + col1];
      T scal = a21 / a11;

      // Read col1 and double it
      for(int i=0; i<col1; i++)
         col[i] = scal*a[i*lda+col1];
      for(int i=col1; i<n; i++)
         col[i] = scal*a[col1*lda+i];

      // Store col to col2
      for(int i=0; i<col2; i++)
         a[i*lda+col2] = col[i];
      for(int i=col2; i<n; i++)
         a[col2*lda+i] = col[i];

      // Free mem
      delete[] col;
   }

   // Pick n/8 random rows and multiply by 1000. Then do the same for n/8 random entries.
   void cause_delays(int n, double *a, int lda) {
      int nsing = n/8;
      if(nsing==0) nsing=1;
      for(int i=0; i<nsing; i++) {
         // Add a row of oversized values
         int idx = n*((float) rand())/RAND_MAX;
         for(int c=0; c<idx; c++)
            a[c*lda+idx] *= 1000;
         for(int r=idx; r<n; r++)
            a[idx*lda+r] *= 1000;
         int row = n*((float) rand())/RAND_MAX;
         int col = n*((float) rand())/RAND_MAX;
         if(row > col) a[col*lda+row] *= 1000;
         else          a[row*lda+col] *= 1000;
      }
   }

   void modify_test_matrix(bool singular, bool delays, int m, int n, double *a, int lda) {
      if(delays)
         cause_delays(m, a, lda);
      if(singular && n!=1) {
         int col1 = n * ((float) rand())/RAND_MAX;
         int col2 = col1;
         while(col1 == col2)
            col2 = n * ((float) rand())/RAND_MAX;
         make_singular(m, col1, col2, a, lda);
      }
   }

   int run_ldlt_tpp_test(double u, double small, bool delays, bool singular, int m, int n, int test=0, int seed=0) {
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
      modify_test_matrix(singular, delays, m, n, a, lda);

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
      // run_ldlt_tpp_test(0.01, 1e-20, true, false, 10, 10);
      // run_ldlt_tpp_test(0.01, 1e-20, true, false, 10, 5);
      run_ldlt_tpp_test(0.01, 1e-20, true, false, 10, 3);

      return err;
   }
}
