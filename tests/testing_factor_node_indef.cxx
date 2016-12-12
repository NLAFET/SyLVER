#include "testing_factor_node_indef.hxx"
// STD
#include <vector>
#include <cstdio>
// SpLDLT
#include "SymbolicSNode.hxx"
#include "factor_indef.hxx"
#include "common.hxx"

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "tests/ssids/kernels/AlignedAllocator.hxx"
#include "ssids/cpu/BuddyAllocator.hxx"
#include "ssids/cpu/NumericNode.hxx"
#include "ssids/cpu/kernels/ldlt_tpp.hxx"
// SSIDS tests
#include "tests/ssids/kernels/framework.hxx"

#if defined(SPLDLT_USE_STARPU)
#include <starpu.h>
#include "StarPU/kernels_indef.hxx"
using namespace spldlt::starpu;
#endif

using namespace spral::ssids::cpu;

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

   template<typename T,
            int iblksz=INNER_BLOCK_SIZE,
            bool debug = false>
   int factor_node_indef_test(T u, T small, bool delays, bool singular, int m, int n,
                              int blksz=INNER_BLOCK_SIZE,
                              int test=0, int seed=0) {
   
      bool failed = false;

      if (debug) printf("[factor_node_indef_test] %d x %d\n", m, n);
   
      // Generate test matrix
      int lda = m;
      T* a = new double[m*lda];
      gen_sym_indef(m, a, lda);
      // gen_posdef(m, a, lda);
      modify_test_matrix(singular, delays, m, n, a, lda);

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
      options.pivot_method = PivotMethod::app_block;
      // options.pivot_method = (aggressive) ? PivotMethod::app_aggressive
      //                                     : PivotMethod::app_block;

      // Setup pool allocator
      typedef BuddyAllocator<T, std::allocator<T>> PoolAllocator;
      PoolAllocator pool_alloc(m*n);

      SymbolicSNode snode;
      snode.nrow = m;
      snode.ncol = n;
      NumericNode<T, PoolAllocator> node(snode, pool_alloc);
      
      // Init node
      // Setup allocator for factors
      spral::test::AlignedAllocator<T> allocT;
      
      node.lcol = allocT.allocate(m*lda);;
      memcpy(node.lcol, a, m*lda*sizeof(T)); // Copy a to l
      // Setup permutation vector
      node.perm = new int[m];
      for(int i=0; i<m; i++) node.perm[i] = i;
      T *d = new T[2*m];      
      // Setup workspaces
      std::vector<spral::ssids::cpu::Workspace> work;
      const int PAGE_SIZE = 8*1024*1024; // 8 MB
      // for(int i=0; i<omp_get_num_threads(); ++i)
      work.emplace_back(PAGE_SIZE);
      T* upd = nullptr;
      // Setup backup
      CopyBackup<T> backup(m, n, blksz);
      
      // Initialize solver (tasking system in particular)
#if defined(SPLDLT_USE_STARPU)
      int ret = starpu_init(NULL);
#endif

      // Init factoriization 
      factor_indef_init<T, PoolAllocator>();

      // int q1 = LDLT
      //    <T, iblksz, CopyBackup<T>, false, debug>
      //    ::factor(
      //          m, n, node.perm, node.lcol, lda, d, backup, options, options.pivot_method,
      //          blksz, 0.0, nullptr, 0, work
      //          );
     
      // Factor node
      int q1 = FactorSymIndef
         <T, iblksz, CopyBackup<T>, debug, PoolAllocator>
         ::ldlt_app (
               m, n, node.perm, node.lcol, lda, d, backup, options, 
               // options.pivot_method,
               blksz, 0.0, upd, 0, work, pool_alloc);

      // By default function calls are asynchronous, so we put a
      // barrier and wait for the DAG to be executed
#if defined(SPLDLT_USE_STARPU)
      starpu_task_wait_for_all();
#endif
      
      if(debug) {
         std::cout << "FIRST FACTOR CALL ELIMINATED " << q1 << " of " << n << " pivots" << std::endl;
         std::cout << "L after first elim:" << std::endl;
         print_mat("%10.2e", m, node.lcol, lda, node.perm);
         std::cout << "D:" << std::endl;
         print_d<T>(q1, d);
      }
      int q2 = 0;
      if(q1 < n) {
         // Finish off with simplistic kernel
         T *ld = new T[2*m];
         q1 += ldlt_tpp_factor(m-q1, n-q1, &node.perm[q1], &node.lcol[(q1)*(lda+1)], lda,
                               &d[2*(q1)], ld, m, options.action, u, small, q1, &node.lcol[q1], lda);
         delete[] ld;
      }
      EXPECT_EQ(m, q1+q2) << "(test " << test << " seed " << seed << ")" << std::endl;

      // Deinitialize solver (shutdown tasking system in particular)
#if defined(SPLDLT_USE_STARPU)
      starpu_shutdown();
#endif

      // Print out matrices if requested
      if(debug) {
         std::cout << "q1=" << q1 << " q2=" << q2 << std::endl;
         std::cout << "L:" << std::endl;
         print_mat("%10.2e", m, node.lcol, lda, node.perm);
         std::cout << "D:" << std::endl;
         print_d<T>(m, d);
      }

      // Perform solve
      T *soln = new T[m];
      solve(m, q1, node.perm, node.lcol, lda, d, b, soln);
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
      delete[] a; allocT.deallocate(node.lcol, m*lda);
      delete[] b;
      delete[] node.perm;
      delete[] d; delete[] soln;

      return failed ? -1 : 0;
   }

   int run_factor_node_indef_tests() {

      int err = 0;

      printf("[FactorNodeIndef tests]\n");

      /* 10x3 matrix
         blksz: 10
         inner blksz: 5
         debug: enabled
       */
      // factor_node_indef_test<double, 5, true>(0.01, 1e-20, true, false, 10, 3, 5);

      /* 10x10 matrix
         blksz: 10
         inner blksz: 5
         debug: enabled
       */
      factor_node_indef_test<double, 4, true>(0.01, 1e-20, true, false, 12, 12, 4);

      return err;
   }
}
