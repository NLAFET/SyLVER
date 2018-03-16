#pragma once

// STD
#include <vector>
#include <cstdio>

// SpLDLT
#include "SymbolicFront.hxx"
#include "factor_indef.hxx"
// SpLDLT tests
#include "common.hxx"

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "tests/ssids/kernels/AlignedAllocator.hxx"
#include "ssids/cpu/BuddyAllocator.hxx"
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

   
   template<typename T,
            int iblksz=INNER_BLOCK_SIZE,
            bool debug = false>
   int factor_node_indef_test(T u, T small, bool posdef, bool delays, bool singular, int m, int n, 
                              int blksz, int ncpu, int test=0, int seed=0) {
   
      bool failed = false;

      printf("[factor_node_indef_test] %d x %d, posdef = %d, blksz = %d\n", m, n, posdef, blksz);
      if (debug) printf("[factor_node_indef_test] %d x %d, posdef = %d\n", m, n, posdef);

      ASSERT_TRUE(m >= n);
         
      // Generate test matrix
      int lda = spral::ssids::cpu::align_lda<T>(m);
      T* a = new double[m*lda];

      if (posdef) gen_posdef(m, a, lda);
      else gen_sym_indef(m, a, lda);

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
      // options.failed_pivot_method = FailedPivotMethod::pass;
         
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

      // Make lcol m columns wide for debugging
      size_t len = (lda+2)*m; // Includes D
      if (debug) printf("m = %d, n = %d, lda = %d, len = %zu\n", m, n, lda, len);
      front.lcol = allocT.allocate(len);

      memcpy(front.lcol, a, lda*n*sizeof(T)); // Copy a to l
      // Put nans on the bottom right corner of the LCOL matrix
      // for (int j = n; j < m; ++j) {
      //    for (int i = j; i < m; ++i) {
      //       front.lcol[lda*j+i] = std::numeric_limits<T>::signaling_NaN(); 
      //    }
      // }
      
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

      // Initialize solver (tasking system in particular)
#if defined(SPLDLT_USE_STARPU)
      struct starpu_conf *conf = new starpu_conf;
      starpu_conf_init(conf);
      conf->ncpus = ncpu;
      int ret = starpu_init(conf);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
      delete conf;
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
      
      // Factor front (first and second pass) and from contrib blocks
      factor_front_indef(front, workspaces, pool_alloc, options);

      // By default function calls are asynchronous, so we put a
      // barrier and wait for the DAG to be executed
#if defined(SPLDLT_USE_STARPU)
      starpu_task_wait_for_all();      
#endif
      if(debug) printf("[factor_node_indef_test] factorization done\n");
      if(debug) printf("[factor_node_indef_test] nelim1 = %d\n", front.nelim1);
      
      int nelim = front.nelim; // Number of eliminated columns
      int q1 = front.nelim1;
      int q2 = nelim - front.nelim1;
            
      std::cout << "FIRST FACTOR CALL ELIMINATED " << q1 << " of " << n << " pivots" << std::endl;
      std::cout << "SECOND FACTOR CALL ELIMINATED " << q2 << " of " << n << " pivots" << std::endl;
      
      auto end = std::chrono::high_resolution_clock::now();
      long ttotal = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
      printf("[testing_factor_node_indef] factor time: %e\n", 1e-9*ttotal);

#if defined(SPLDLT_USE_STARPU)
      spldlt::starpu::unregister_node_indef_submit(front);
      starpu_task_wait_for_all(); // Wait for unregistration of handles      
      starpu_data_unregister(sfront.hdl); // Node's symbolic handle
      starpu_data_unregister(front.contrib_hdl);
#endif
      
      // Deinitialize solver (shutdown tasking system in particular)
#if defined(SPLDLT_USE_STARPU)
      starpu_shutdown();
#endif

      if (debug) {
         std::cout << "CB:" << std::endl;
         print_cb("%10.2e", front);
      }

      // Alloc L array for storing factors
      T *l = allocT.allocate(lda*m);
      // Initialize L with zeros
      for (int j=0; j<m; ++j)
         for (int i=j; i<m; ++i)
            l[i+j*lda] = 0.0;
      
      // Copy factors from LCOL into L array
      // Copy only factors (colmuns 1 to n)
      memcpy(l, front.lcol, lda*n*sizeof(T)); // Copy a to l

      if (debug) {
         std::cout << "LCOL:" << std::endl;
         print_mat("%10.2e", m, front.lcol, lda, front.perm);                  
         // std::cout << "L:" << std::endl;
         // print_mat("%10.2e", m, l, lda, front.perm);
      }
      
      // Eliminate remaining columns in L
      if (m > n) {

         // Copy A (columns n+1 to m) into L
         memcpy(&l[lda*n], &a[lda*n], lda*(m-n)*sizeof(T));
         // Add entries (columns n+1 to m) form CB into L
         if (front.nelim > 0) add_cb_to_a(front, l, lda);
         
         // Debug
         // Ignore CB and do update
         // Apply outer product update
         // memcpy(&l[n*lda], &a[n*lda], lda*(m-n)*sizeof(T));
         // do_update<T>(m-n, q1+q2, &l[n*(lda+1)], &l[n], lda, d);
         
         if (debug) {
            std::cout << "D:" << std::endl;
            print_d(m, d);
            std::cout << "L:" << std::endl;
            print_mat("%10.2e", m, l, lda, front.perm);
         }

         if (debug) std::cout << "Eliminate remaining columns using TPP.." << std::endl;
            
         // Finish off with TPP
         T *ld = new T[2*m];
         nelim += ldlt_tpp_factor(
               m-nelim, m-nelim, &front.perm[nelim], &l[nelim*(lda+1)], lda,
               &d[2*nelim], ld, m, options.action, u, small, nelim, &l[nelim], lda);
         delete[] ld;
      }
      
      if (debug) {
         std::cout << "D:" << std::endl;
         print_d(m, d);
         std::cout << "L:" << std::endl;
         print_mat("%10.2e", m, l, lda, front.perm);
      }

      if (debug) std::cout << "nelim = " << nelim << std::endl;
      
      EXPECT_EQ(m, nelim) << "(test " << test << " seed " << seed << ")" << std::endl;
      
      // Perform solve
      T *soln = new T[m];
      solve(m, nelim, front.perm, l, lda, d, b, soln);
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
      delete[] a; allocT.deallocate(front.lcol, len);
      delete[] b;
      delete[] front.perm;
      delete[] soln;

      return failed ? -1 : 0;
   }

   // Run tests for the node factorization
   int run_factor_node_indef_tests();

   }} // namespace spldlt::tests
