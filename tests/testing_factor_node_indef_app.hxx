#pragma once

// SpLDLT
#include "BuddyAllocator.hxx"
#include "sylver/SymbolicFront.hxx"
#include "NumericFront.hxx"
#include "factor_indef.hxx"
#include "common.hxx"

// STD
#include <vector>
#include <cstdio>
#include <chrono>

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "tests/ssids/kernels/AlignedAllocator.hxx"
#include "ssids/cpu/kernels/ldlt_tpp.hxx"
#include "ssids/cpu/kernels/ldlt_app.hxx"
// SSIDS tests
#include "tests/ssids/kernels/framework.hxx"

#if defined(SPLDLT_USE_STARPU)
#include <starpu.h>
#include "StarPU/codelets.hxx"
#include "StarPU/kernels.hxx"
#include "StarPU/kernels_indef.hxx"
#endif


namespace spldlt {
namespace tests {
   
   /// @brief Launch test for the LDL^{T} factorization of a mxn
   /// indefinite node using APP strategy.
   template<typename T,
            int iblksz=sylver::spldlt::INNER_BLOCK_SIZE,
            bool debug = false>
   int factor_node_indef_app_test(
         T u, T small, bool delays, bool singular, int m, int n,
         int blksz=sylver::spldlt::INNER_BLOCK_SIZE, int ncpu=1,
         int test=0, int seed=0) {
      
      
      bool failed = false;

      if (debug) printf("[factor_node_indef_app_test] %d x %d\n", m, n);

      ASSERT_TRUE(m >= n);

      // Generate test matrix
      int lda = spral::ssids::cpu::align_lda<T>(m);
      T* a = new double[m*lda];
      gen_sym_indef(m, a, lda);
      // Modify matrix for test adding delays and singularities if
      // needed
      modify_test_matrix(singular, delays, m, n, a, lda);

      // Generate a RHS based on x=1, b=Ax
      T *b = new T[m];
      gen_rhs(m, a, lda, b);

      // Setup options
      sylver::options_t options;
      options.action = true;
      options.multiplier = 2.0;
      options.small = small;
      options.u = u;
      options.print_level = 0;
      options.small_subtree_threshold = 100*100*100;
      options.nb = blksz;
      options.pivot_method = sylver::PivotMethod::app_block  /*PivotMethod::app_aggressive*/;
      // options.pivot_method = (aggressive) ? PivotMethod::app_aggressive
      //                                     : PivotMethod::app_block;
      options.failed_pivot_method = sylver::FailedPivotMethod::tpp;

      // Setup pool allocator
      using PoolAllocator = sylver::BuddyAllocator<T, std::allocator<T>>;
      PoolAllocator pool_alloc(lda*n);

      sylver::SymbolicFront sfront;
      sfront.nrow = m;
      sfront.ncol = n;

      // Init node
      // Setup allocator for factors
      using FactorAllocator = spral::test::AlignedAllocator<T>;
      FactorAllocator allocT;

      // Numeric front type
      using NumericFrontType = sylver::spldlt::NumericFront<T, FactorAllocator, PoolAllocator>;

      NumericFrontType front(sfront, allocT, pool_alloc, blksz);

      front.allocate(nullptr);

      // Copy the whole matrix into LCOL for debugging
      memcpy(front.lcol, a, lda*m*sizeof(T)); // Copy a to l
      // Put nans on the bottom right corner of the LCOL matrix
      for (int j = n; j < m; ++j) {
         for (int i = j; i < m; ++i) {
            front.lcol[lda*j+i] = std::numeric_limits<T>::signaling_NaN(); 
         }
      }
      
      if (debug) {
         std::cout << "LCOL:" << std::endl;
         print_mat("%10.2e", m, front.lcol, lda);
      }
      

      // Setup permutation vector
      front.perm = new int[m];
      for(int i=0; i<m; i++) front.perm[i] = i;
      T *d = &front.lcol[lda*n];
      // T *d = new T[2*m];      

      // Setup backup
      using Backup = sylver::CopyBackup<T, PoolAllocator>;
      
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

      std::vector<sylver::inform_t> worker_stats(nworkers);
            
      workspaces.reserve(nworkers);
      for(int i = 0; i < nworkers; ++i)
         workspaces.emplace_back(PAGE_SIZE);
      if(debug) printf("[factor_node_indef_app_test] nworkers =  %d\n", nworkers);

      // Init factoriization 
#if defined(SPLDLT_USE_STARPU)
      sylver::spldlt::starpu::codelet_init<T, FactorAllocator, PoolAllocator>();
      sylver::spldlt::starpu::codelet_init_indef<T, iblksz, Backup, FactorAllocator, PoolAllocator>();
      sylver::spldlt::starpu::codelet_init_factor_indef<NumericFrontType, PoolAllocator>();
#endif

#if defined(SPLDLT_USE_STARPU)
      // Register symbolic handles
      starpu_void_data_register(&(sfront.hdl)); // Node's symbolic handle
      // Register StarPU data handles
      front.register_node();
#endif
      
      if(debug) printf("[factor_node_indef_app_test] factor front..\n");

      auto start = std::chrono::high_resolution_clock::now();

      T *upd = nullptr;
      int q1 = 0; // Number of eliminated colmuns (first pass)

      front.nelim(0);

      // Factorization type
      using FactorType = sylver::spldlt::FactorIndefAPP<
         T,
         sylver::spldlt::INNER_BLOCK_SIZE,
         Backup,
         debug,
         FactorAllocator,
         PoolAllocator>;

      // Factor front (first and second pass) and from contrib blocks
      FactorType::factor_front_indef_app(
            front, options, worker_stats, 0.0, upd, 0, workspaces, pool_alloc,
            front.nelim());

#if defined(SPLDLT_USE_STARPU)
      starpu_task_wait_for_all();      
#endif

      auto end = std::chrono::high_resolution_clock::now();
      long ttotal = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();

      if(debug) printf("[factor_node_indef_app_test] factorization done\n");
      printf("[factor_node_indef_app_test] factor time: %e\n", 1e-9*ttotal);
      
#if defined(SPLDLT_USE_STARPU)
      sylver::spldlt::starpu::unregister_node_submit(front);
      starpu_data_unregister_submit(sfront.hdl); // Node's symbolic handle
      starpu_task_wait_for_all(); // Wait for unregistration of handles      
#endif

      // Shutdown runtime system
#if defined(SPLDLT_USE_STARPU)
      starpu_shutdown();
#endif

   }

      // Run tests for the APTP node factorization
      int run_factor_node_indef_app_tests();

   }} // namespace spldlt::tests
