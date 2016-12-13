#include "testing_factor_indef.hxx"

#include "common.hxx"
#include "kernels/ldlt_app.hxx"
#include "factor_indef.hxx"

#include <cstring>
#include <vector>

#include "tests/ssids/kernels/framework.hxx"
#include "tests/ssids/kernels/AlignedAllocator.hxx"
#include "ssids/cpu/BuddyAllocator.hxx"
// #include "ssids/cpu/kernels/ldlt_app.hxx"
#include "ssids/cpu/kernels/wrappers.hxx"
// #include "ssids/cpu/kernels/ldlt_app.cxx" // .cxx as we need internal namespace
#include "ssids/cpu/kernels/ldlt_tpp.hxx"
#include "ssids/cpu/cpu_iface.hxx"

namespace spldlt {

   using namespace spldlt::ldlt_app_internal;

   template<typename T>
   void solve(int m, int n, const int *perm, const T *l, int ldl, const T *d, const T *b, T *x) {
      for(int i=0; i<m; i++) x[i] = b[perm[i]];
      // Fwd slv
      ldlt_app_solve_fwd(m, n, l, ldl, 1, x, m);
      ldlt_app_solve_fwd(m-n, m-n, &l[n*(ldl+1)], ldl, 1, &x[n], m);
      // Diag slv
      ldlt_app_solve_diag(n, d, 1, x, m);
      ldlt_app_solve_diag(m-n, &d[2*n], 1, &x[n], m);
      // Bwd slv
      ldlt_app_solve_bwd(m-n, m-n, &l[n*(ldl+1)], ldl, 1, &x[n], m);
      ldlt_app_solve_bwd(m, n, l, ldl, 1, x, m);
      // Undo permutation
      T *temp = new T[m];
      for(int i=0; i<m; i++) temp[i] = x[i];
      for(int i=0; i<m; i++) x[perm[i]] = temp[i];
      // Free mem
      delete[] temp;
   }

   /*
     bool dblk_singular singular diagonal blocks
   */
   template<typename T,
            bool aggressive, // Use Cholesky-like app pattern
            bool debug=true,
            int iblksz=INNER_BLOCK_SIZE>
   int factor_indef_test(T u, T small, 
                         bool delays, bool singular, bool dblk_singular, 
                         int m, int n, 
                         int blksz=INNER_BLOCK_SIZE, 
                         int test=0, int seed=0) {

      bool failed = false;

      // Note: We generate an m x m test matrix, then factor it as an
      // m x n matrix followed by an (m-n) x (m-n) matrix [give or take delays]

      // Generate test matrix
      int lda = align_lda<T>(m);
      double* a = new T[m*lda];
      gen_sym_indef(m, a, lda);
      modify_test_matrix<T, iblksz>(
            singular, delays, dblk_singular, m, n, a, lda
            );
      
      // Generate a RHS based on x=1, b=Ax
      T *b = new T[m];
      gen_rhs(m, a, lda, b);

      // Print out matrices if requested
      if(debug) {
         std::cout << "A:" << std::endl;
         print_mat("%10.2e", m, a, lda);
      }

      // Setup options
      struct cpu_factor_options options;
      options.action = true;
      options.multiplier = 2.0;
      options.small = small;
      options.u = u;
      options.print_level = 0;
      options.small_subtree_threshold = 100*100*100;
      options.cpu_block_size = 256;
      options.pivot_method = (aggressive) ? PivotMethod::app_aggressive
                                          : PivotMethod::app_block;

      // Init factorization
      typedef BuddyAllocator<T,std::allocator<T>> PoolAllocator;
      PoolAllocator pool_alloc(m*n);

      // factor_indef_init<T, PoolAllocator>();

      // Factorize using main routine
      spral::test::AlignedAllocator<T> allocT;
      T *l = allocT.allocate(m*lda);
      memcpy(l, a, m*lda*sizeof(T)); // Copy a to l
      int *perm = new int[m];
      for(int i=0; i<m; i++) perm[i] = i;
      T *d = new T[2*m];
      // Setup workspaces
      std::vector<spral::ssids::cpu::Workspace> work;
      const int PAGE_SIZE = 8*1024*1024; // 8 MB
      // for(int i=0; i<omp_get_num_threads(); ++i)
      work.emplace_back(PAGE_SIZE);
      T* upd = nullptr;
      // int q1 = ldlt_app_factor(
      //       m, n, perm, l, lda, d, 0.0, upd, 0, 
      //       options, work, 
      //       // allocT
      //       pool_alloc
      //       );
      int const use_tasks = false;
      CopyBackup<T> backup(m, n, blksz);
      // int q1 = LDLT
      //    <T, iblksz, CopyBackup<T>, use_tasks, debug>
      //    ::factor(
      //          m, n, perm, l, lda, d, backup, options, options.pivot_method,
      //          blksz, 0.0, nullptr, 0, work
      //          );
      int q1 = FactorSymIndef
         <T, iblksz, CopyBackup<T>, debug, PoolAllocator>
         ::ldlt_app (
               m, n, perm, l, lda, d, backup, options, 
               // options.pivot_method,
               blksz, 0.0, upd, 0, work, pool_alloc);
      
      if(debug) {
         std::cout << "FIRST FACTOR CALL ELIMINATED " << q1 << " of " << n << " pivots" << std::endl;
         std::cout << "L after first elim:" << std::endl;
         print_mat("%10.2e", m, l, lda, perm);
         std::cout << "D:" << std::endl;
         print_d<T>(q1, d);
      }
      int q2 = 0;
      if(q1 < n) {
         // Finish off with simplistic kernel
         T *ld = new T[2*m];
         q1 += ldlt_tpp_factor(m-q1, n-q1, &perm[q1], &l[(q1)*(lda+1)], lda,
                               &d[2*(q1)], ld, m, options.action, u, small, q1, &l[q1], lda);
         delete[] ld;
      }
      if(m > n) {
         // Apply outer product update
         do_update<T>(m-n, q1, &l[n*(lda+1)], &l[n], lda, d);
         // Second (m-n) x (m-n) matrix [but add delays if any]
         int *perm2 = new int[m-q1];
         for(int i=0; i<m-q1; i++)
            perm2[i] = i;
         
         // q2 = ldlt_app_factor(
         //       m-q1, m-q1, perm2, &l[q1*(lda+1)], lda, &d[2*q1], 0.0, upd, 0,
         //       options, work,
         //       pool_alloc
         //       );
         CopyBackup<T> backup(m-q1, m-q1, blksz);
         q2 = LDLT
            <T, iblksz, CopyBackup<T>, use_tasks, debug>
            ::factor(
                  m-q1, m-q1, perm2, &l[q1*(lda+1)], lda, &d[2*q1], backup, options,
                  options.pivot_method, blksz, 0.0, nullptr, 0, work
                  );
         permute_rows(m-q1, q1, perm2, &perm[q1], &l[q1], lda);
         delete[] perm2;
         if(q1+q2 < m) {
            // Finish off with simplistic kernel
            T *ld = new T[2*m];
            q2 += ldlt_tpp_factor(m-q1-q2, m-q1-q2, &perm[q1+q2],
                                  &l[(q1+q2)*(lda+1)], lda, &d[2*(q1+q2)], ld, m, options.action,
                                  u, small, q1+q2, &l[q1+q2], lda);
            delete[] ld;
         }
      }
      EXPECT_EQ(m, q1+q2) << "(test " << test << " seed " << seed << ")" << std::endl;

      // Print out matrices if requested
      if(debug) {
         std::cout << "q1=" << q1 << " q2=" << q2 << std::endl;
         std::cout << "L:" << std::endl;
         print_mat("%10.2e", m, l, lda, perm);
         std::cout << "D:" << std::endl;
         print_d<T>(m, d);
      }

      // Perform solve
      T *soln = new T[m];
      solve(m, q1, perm, l, lda, d, b, soln);
      if(debug) {
         printf("soln = ");
         for(int i=0; i<m; i++) printf(" %le", soln[i]);
         printf("\n");
      }

      // Check residual
      T bwderr = backward_error(m, a, lda, b, 1, soln, m);
      if(debug) printf("bwderr = %le\n", bwderr);
      EXPECT_LE(bwderr, 5e-14) << "(test " << test << " seed " << seed << ")" << std::endl;

      // Cleanup memory
      delete[] a; allocT.deallocate(l, m*lda);
      delete[] b;
      delete[] perm;
      delete[] d; delete[] soln;

      return failed ? -1 : 0;
   }

   int run_factor_indef_tests() {

      int err = 0;

      printf("[LDLT APP tests]\n");
      
      /* 10x10 matrix
         Type=double, agressive=false, debug=true
         delays=true, sigular=false, dblk_singular=fasle
         ldlt_app_test<double, false, true>(0.01, 1e-20, true, false, false, 10, 10);
         default blocksize
      */
      // factorize_indef_test<double, false, true>(0.01, 1e-20, true, false, false, 10, 3);
      
      /*
         10x3 matrix
         Inner block size = 5
         Outer block size = Inner block size 
       */
      // factorize_indef_test<double, false, true, 5>(0.01, 1e-20, true, false, false, 10, 3, 10);

      /* 
         10x3 matrix
         Inner block size = 5
         Outer block size = 5
       */
      // factorize_indef_test<double, false, true, 5>(0.01, 1e-20, true, false, false, 10, 3, 10);

      /* 
         10x3 matrix
         Inner block size = 5
         Outer block size = 5
       */
      // factorize_indef_test<double, false, true, 5>(0.01, 1e-20, true, false, false, 10, 3, 5);

      /* 
         10x10 matrix
         Inner block size = 5
         Outer block size = 5
       */
      factor_indef_test<double, false, true, 4>(0.01, 1e-20, true, false, false, 8, 8, 4);
      
      return err;
   }

} // namespace spldlt
