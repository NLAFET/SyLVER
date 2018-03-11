#pragma once

// STD
#include <vector>
#include <cstdio>
// SpLDLT
#include "SymbolicFront.hxx"
#include "factor_indef.hxx"
#include "common.hxx"

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "tests/ssids/kernels/AlignedAllocator.hxx"
#include "ssids/cpu/BuddyAllocator.hxx"
// #include "ssids/cpu/NumericNode.hxx"
#include "ssids/cpu/kernels/ldlt_tpp.hxx"
#include "ssids/cpu/kernels/ldlt_app.hxx"
// SSIDS tests
#include "tests/ssids/kernels/framework.hxx"

#if defined(SPLDLT_USE_STARPU)
#include <starpu.h>
#include "StarPU/kernels.hxx"
#include "StarPU/kernels_indef.hxx"
#endif

// using namespace spral::ssids::cpu;

namespace spldlt { namespace tests {

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
                              int blksz=INNER_BLOCK_SIZE, int ncpu=1,
                              int test=0, int seed=0) {
   
      bool failed = false;

      if (debug) printf("[factor_node_indef_test] %d x %d\n", m, n);
   
      // Generate test matrix
      int lda = spral::ssids::cpu::align_lda<T>(m);
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
      options.cpu_block_size = blksz;
      options.pivot_method = PivotMethod::app_block  /*PivotMethod::app_aggressive*/;
      // options.pivot_method = (aggressive) ? PivotMethod::app_aggressive
      //                                     : PivotMethod::app_block;
      options.failed_pivot_method = FailedPivotMethod::tpp;
         
      // Setup pool allocator
      typedef BuddyAllocator<T, std::allocator<T>> PoolAllocator;
      PoolAllocator pool_alloc(lda*n);

      SymbolicFront sfront;
      sfront.nrow = m;
      sfront.ncol = n;
      NumericFront<T, PoolAllocator> front(sfront, pool_alloc, blksz);
      front.ndelay_in = 0; // No incoming delayed columns      
      front.ndelay_out = 0;
      // Init node
      // Setup allocator for factors
      typedef spral::test::AlignedAllocator<T> FactorAllocator;
      FactorAllocator allocT;

      size_t len = (lda+2)*n; // Includes D
      if (debug) printf("m = %d, n = %d, lda = %d, len = %zu\n", m, n, lda, len);
      front.lcol = allocT.allocate(len);
      memcpy(front.lcol, a, lda*n*sizeof(T)); // Copy a to l
      
      if (debug) {
         std::cout << "LCOL:" << std::endl;
         print_mat("%10.2e", m, front.lcol, lda);
      }
      
      // Allocate block structure
      front.alloc_blocks();

      // Setup permutation vector
      front.perm = new int[m];
      for(int i=0; i<m; i++) front.perm[i] = i;
      T *d = &front.lcol[lda*n];
      // T *d = new T[2*m];      
//       T* upd = nullptr;
      // Setup backup
      typedef spldlt::ldlt_app_internal::CopyBackup<T, PoolAllocator> Backup;
      // CopyBackup<T, PoolAllocator> backup(m, n, blksz);
      front.alloc_backup();
      // Setup cdata
      front.alloc_cdata();

      // Allocate contribution blocks
      front.alloc_contrib_blocks();

      // Copy A (n+1 to m columns) into contrib blocks
      size_t contrib_dimn = m-n; // Dimension of contribution block
      if (contrib_dimn>0) {
         int nr = front.get_nr();
         int rsa = n/blksz;
         int ncontrib = nr-rsa;

         for(int j = rsa; j < nr; j++) {
            // First col in contrib block
            int first_col = std::max(j*blksz, n);
            // Tile width
            int blkn = std::min((j+1)*blksz, m) - first_col;
            for(int i = rsa; i < nr; i++) {
               // First col in contrib block
               int first_row = std::max(i*blksz, n);
               // Tile height
               int blkm = std::min((i+1)*blksz, m) - first_row;
               memcpy(
                     front.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].a,
                     &a[first_col*lda + first_row],
                     blkm*blkn*sizeof(T));
            }
         }
      }
     
      // Initialize solver (tasking system in particular)
#if defined(SPLDLT_USE_STARPU)
      struct starpu_conf *conf = new starpu_conf;
      starpu_conf_init(conf);
      conf->ncpus = ncpu;
      int ret = starpu_init(conf);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
#endif

      // Setup workspaces
      std::vector<spral::ssids::cpu::Workspace> workspaces;
      const int PAGE_SIZE = 8*1024*1024; // 8 MB
      int nworkers;
#if defined(SPLDLT_USE_STARPU)
      nworkers = starpu_cpu_worker_get_count();
#else
      nworkers = omp_get_num_threads();
#endif
      workspaces.reserve(nworkers);
      for(int i = 0; i < nworkers; ++i)
         workspaces.emplace_back(PAGE_SIZE);
      if(debug) printf("[factor_node_indef_test] nworkers =  %d\n", nworkers);

      // Init factoriization 
#if defined(SPLDLT_USE_STARPU)
      spldlt::starpu::codelet_init<T, FactorAllocator, PoolAllocator>();
      spldlt::starpu::codelet_init_indef<T, iblksz, Backup, PoolAllocator>();
      spldlt::starpu::codelet_init_factor_indef<T, PoolAllocator>();
#endif


#if defined(SPLDLT_USE_STARPU)
      // Register symbolic handles
      starpu_void_data_register(&(sfront.hdl)); // Node's symbolic handle
      starpu_void_data_register(&(front.contrib_hdl));
      // Register StarPU data handles
      spldlt::starpu::register_node_indef(front);
#endif
      
      if(debug) printf("[factor_node_indef_test] factor front..\n");

      auto start = std::chrono::high_resolution_clock::now();

      int q1 = 0; // Number of eliminated colmuns (first pass)
      int q2 = 0; // Number of eliminated colmuns (second pass)
      
      // Factor front (first and second pass) and from contrib blocks
      factor_front_indef(
            front, workspaces, pool_alloc, options);

      // By default function calls are asynchronous, so we put a
      // barrier and wait for the DAG to be executed
#if defined(SPLDLT_USE_STARPU)
      starpu_task_wait_for_all();      
#endif
      if(debug) printf("[factor_node_indef_test] factorization done\n");
      if(debug) printf("[factor_node_indef_test] nelim1 = %d\n", front.nelim1);
         
      q1 = front.nelim1;
      q2 = front.nelim - front.nelim1;
      
         //       // q1 = LDLT
//       //    <T, iblksz, CopyBackup<T>, false, debug>
//       //    ::factor(
//       //          m, n, node.perm, node.lcol, lda, d, backup, options, options.pivot_method,
//       //          blksz, 0.0, nullptr, 0, work
//       //          );
     
//       // Factor node in sequential
//       // q1 = FactorSymIndef
//       //    <T, iblksz, CopyBackup<T>, debug, PoolAllocator>
//       //    ::ldlt_app_notask(
//       //          m, n, node.perm, node.lcol, lda, d, backup, options, 
//       //          // options.pivot_method,
//       //          blksz, 0.0, upd, 0, work[0], pool_alloc);

//       // Factor node
//       q1 = FactorSymIndef
//          <T, iblksz, CopyBackup<T>, debug, PoolAllocator>
//          ::ldlt_app(
//                m, n, node.perm, node.lcol, lda, d, backup, options, 
//                // options.pivot_method,
//                blksz, 0.0, upd, 0, work, pool_alloc);
      
//       // q1 = spral::ssids::cpu::ldlt_app_factor(
//       //       m, n, node.perm, node.lcol, lda, d, 0.0, upd, 0,
//       //       options, work, pool_alloc);
      
// #if defined(SPLDLT_USE_STARPU)
//       starpu_task_wait_for_all();
// #endif
      
      std::cout << "FIRST FACTOR CALL ELIMINATED " << q1 << " of " << n << " pivots" << std::endl;
      std::cout << "SECOND FACTOR CALL ELIMINATED " << q2 << " of " << n << " pivots" << std::endl;
      
//       if(debug) {
//          std::cout << "L after first elim:" << std::endl;
//          print_mat("%10.2e", m, node.lcol, lda, node.perm);
//          std::cout << "D:" << std::endl;
//          print_d<T>(q1, d);
//       }
//       int q2 = 0;
//       if(q1 < n) {
//          // Finish off with simplistic kernel
// #if defined(SPLDLT_USE_STARPU)
//          starpu_fxt_trace_user_event(0);
// #endif         
//          T *ld = new T[2*m];
//          q1 += ldlt_tpp_factor(m-q1, n-q1, &node.perm[q1], &node.lcol[(q1)*(lda+1)], lda,
//                                &d[2*(q1)], ld, m, options.action, u, small, q1, &node.lcol[q1], lda);
//          delete[] ld;
// #if defined(SPLDLT_USE_STARPU)
//          starpu_fxt_trace_user_event(0);
// #endif         
//       }
//       EXPECT_EQ(m, q1+q2) << "(test " << test << " seed " << seed << ")" << std::endl;

      auto end = std::chrono::high_resolution_clock::now();
      long ttotal = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
      printf("[testing_factor_node_indef] factor time: %e\n", 1e-9*ttotal);

#if defined(SPLDLT_USE_STARPU)
      unregister_node_submit(front);
      starpu_data_unregister_submit(sfront.hdl); // Node's symbolic handle
      starpu_data_unregister_submit(front.contrib_hdl);
      starpu_task_wait_for_all(); // Wait for unregistration of handles      
#endif
      
      // Deinitialize solver (shutdown tasking system in particular)
#if defined(SPLDLT_USE_STARPU)
      starpu_shutdown();
#endif

//       std::cout << "q1=" << q1 << " q2=" << q2 << std::endl;
      
//       // Print out matrices if requested
//       if(debug) {
//          std::cout << "L:" << std::endl;
//          print_mat("%10.2e", m, node.lcol, lda, node.perm);
//          std::cout << "D:" << std::endl;
//          print_d<T>(m, d);
//       }

      // Copy factors from lcol into l array
      T *l = allocT.allocate(lda*m);
      memcpy(l, front.lcol, lda*n*sizeof(T)); // Copy a to l
      // Copy L (n+1 to m columns) from contrib blocks into l
      if (contrib_dimn>0) {
         int nr = front.get_nr();
         int rsa = n/blksz;
         int ncontrib = nr-rsa;

         for(int j = rsa; j < nr; j++) {
            // First col in contrib block
            int first_col = std::max(j*blksz, n);
            // Tile width
            int blkn = std::min((j+1)*blksz, m) - first_col;
            for(int i = rsa; i < nr; i++) {
               // First col in contrib block
               int first_row = std::max(i*blksz, n);
               // Tile height
               int blkm = std::min((i+1)*blksz, m) - first_row;
               memcpy(&l[first_col*lda + first_row],
                      front.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].a,
                      blkm*blkn*sizeof(T));
            }
         }
      }

      // if (debug) print_mat("%10.2e", m, front.lcol, lda, front.perm);
      if (debug) {
         std::cout << "L:" << std::endl;
         print_mat("%10.2e", m, l, lda, front.perm);
      }

      // Finish off with TPP
      if(q2 < n) {
         T *ld = new T[2*m];
         q2 += ldlt_tpp_factor(m-q2, n-q2, &front.perm[q2], &front.lcol[(q2)*(lda+1)], lda,
                               &d[2*(q2)], ld, m, options.action, u, small, q2, &front.lcol[q2], lda);
         delete[] ld;

      }

      if (debug) {
         std::cout << "q2 = " << q2 << std::endl;
      }

      EXPECT_EQ(m, q1+q2) << "(test " << test << " seed " << seed << ")" << std::endl;
      
      // Perform solve
      T *soln = new T[m];
      solve(m, q2, front.perm, l, lda, d, b, soln);
      if(debug) {
         printf("soln = ");
         for(int i=0; i<m; i++) printf(" %le", soln[i]);
         printf("\n");
      }

      // Check residual
      T bwderr = backward_error(m, a, lda, b, 1, soln, m);
      /*if(debug)*/ printf("bwderr = %le\n", bwderr);
      EXPECT_LE(bwderr, 5e-14) << "(test " << test << " seed " << seed << ")" << std::endl;

      // Cleanup memory
      allocT.deallocate(l, m*lda);
      delete[] a; allocT.deallocate(front.lcol, m*lda);
      delete[] b;
      delete[] front.perm;
      delete[] soln;

      return failed ? -1 : 0;
   }

   // Run tests for the APTP node factorization
   int run_factor_node_indef_tests();

   }} // namespace spldlt::tests
