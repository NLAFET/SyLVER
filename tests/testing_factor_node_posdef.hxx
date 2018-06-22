#pragma once

// SpLDLT
#include "SymbolicFront.hxx"
#include "NumericFront.hxx"
#include "factor.hxx"
#if defined(SPLDLT_USE_STARPU)
#include "StarPU/scheduler.hxx"
#include "StarPU/kernels.hxx"
#endif

// SpLDLT tests
// #include "testing_factor_node_posdef.hxx"

// STD
#include <vector>
#include <cstdio>
#include <chrono>

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/BuddyAllocator.hxx"
#include "ssids/cpu/kernels/cholesky.hxx"
// SSIDS tests
#include "tests/ssids/kernels/AlignedAllocator.hxx"

namespace spldlt { namespace tests {

      template<typename T>
      int factor_node_posdef_test(int m, int n, int blksz, int ncpu, int ngpu) {

         bool failed = false;
         int ret;

         printf("[factor_node_posdef_test] m = %d, n =  %d, blksz = %d\n", m, n, blksz);

         ////////////////////////////////////////
         // Setup test matrix and rhs

         // Generate test matrix
         int lda = spral::ssids::cpu::align_lda<T>(m);
         T* a = new double[m*lda];

         gen_posdef(m, a, lda);

         // Generate a RHS based on x=1, b=Ax
         T *b = new T[m];
         gen_rhs(m, a, lda, b);

         // Print out matrices if requested
         // if(debug) {
         //    std::cout << "A:" << std::endl;
         //    print_mat("%10.2e", m, a, lda);
         // }

         // Setup options
         struct cpu_factor_options options;
         options.cpu_block_size = blksz;

         // Setup pool allocator
         typedef spral::ssids::cpu::BuddyAllocator<T, std::allocator<T>> PoolAllocator;
         PoolAllocator pool_alloc(lda*n);
         
         // Setup frontal matrix
         SymbolicFront sfront;
         sfront.nrow = m;
         sfront.ncol = n;
         NumericFront<T, PoolAllocator> front(sfront, pool_alloc, blksz);
         front.ndelay_in = 0; // No incoming delayed columns      
         front.ndelay_out = 0;

         // Setup allocator for factors
         typedef spral::test::AlignedAllocator<T> FactorAllocator;
         FactorAllocator allocT;
         // Allocate factors
         size_t len = lda*m;
         // if (debug) printf("m = %d, n = %d, lda = %d, len = %zu\n", m, n, lda, len);
         front.lcol = allocT.allocate(len);

         // Copy a into l
         memcpy(front.lcol, a, lda*n*sizeof(T));

         // Allocate contribution blocks
         front.alloc_contrib_blocks();

         ////////////////////////////////////////
         // Init runtime system
#if defined(SPLDLT_USE_STARPU)
         struct starpu_conf conf;
         starpu_conf_init(&conf);
         conf.ncpus = ncpu;
#if defined(SPLDLT_USE_GPU)
         conf.ncuda = ngpu;
         // Scheduler
         conf.sched_policy_name = "heteroprio";
         conf.sched_policy_init = &spldlt::starpu::init_heteroprio;
         // conf.sched_policy_name = "ws";
#else
         conf.sched_policy_name = "lws";
#endif

         ret = starpu_init(&conf);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
#if defined(SPLDLT_USE_GPU)
      starpu_cublas_init();
#endif
#endif

         // Get number of workers (CPU and GPU)
         int nworkers = 1;
#if defined(SPLDLT_USE_STARPU)
         nworkers = starpu_worker_get_count();
#endif

         ////////////////////////////////////////
         // Init factor
#if defined(SPLDLT_USE_STARPU)
         spldlt::starpu::codelet_init<T, FactorAllocator, PoolAllocator>();

         // Register symbolic handles
         starpu_void_data_register(&(sfront.hdl)); // Symbolic handle on node
         starpu_void_data_register(&(front.contrib_hdl)); // Symbolic handle on contrib blocks 
         
         spldlt::starpu::register_node(front);
#endif
         
         printf("[factor_node_posdef_test] Factor..\n");
         // Run matrix factorization
         auto start = std::chrono::high_resolution_clock::now();

         factor_front_posdef(front, options);
            
#if defined(SPLDLT_USE_STARPU)
         starpu_task_wait_for_all();      
#endif
         auto end = std::chrono::high_resolution_clock::now();
         long ttotal = 
            std::chrono::duration_cast<std::chrono::nanoseconds>
            (end-start).count();
         
         printf("[factor_node_posdef_test] Done\n");

         // if(debug) printf("[factor_node_indef_test] factorization done\n");

         spldlt::starpu::unregister_node_submit(front);
         starpu_task_wait_for_all(); // Wait for unregistration of handles      
         starpu_data_unregister(sfront.hdl); // Node's symbolic handle
         starpu_data_unregister(front.contrib_hdl);
      
         // Deinitialize solver (including shutdown tasking system)
#if defined(SPLDLT_USE_STARPU)
#if defined(SPLDLT_USE_GPU)
         starpu_cublas_shutdown();
#endif
         starpu_shutdown();
#endif

         ////////////////////////////////////////
         // Check result

         // Alloc L array for storing factors
         T *l = allocT.allocate(lda*m);
         // Initialize L with zeros
         for (int j=0; j<m; ++j)
            for (int i=j; i<m; ++i)
               l[i+j*lda] = 0.0;
      
         // Copy factors from LCOL into L array
         // Copy only factors (colmuns 1 to n)
         memcpy(l, front.lcol, lda*n*sizeof(T)); // Copy a to l
         
         // Eliminate remaining columns in L
         if (m > n) {

            // ...
            // lapack_potrf<double>(FILL_MODE_LWR, m-n, &l[n*lda+n], lda);
         }
         
         int nrhs = 1;
         int ldsoln = m;
         double *soln = new double[nrhs*ldsoln];
         for(int r=0; r<nrhs; ++r)
            memcpy(&soln[r*ldsoln], b, m*sizeof(double));

         printf("[factor_node_posdef_test] Solve..\n");
         
         cholesky_solve_fwd(m, n, l, lda, nrhs, soln, ldsoln);
         host_trsm<double>(SIDE_LEFT, FILL_MODE_LWR, OP_N, DIAG_NON_UNIT, m-n, nrhs, 1.0, &l[n*lda+n], lda, &soln[n], ldsoln);
         host_trsm<double>(SIDE_LEFT, FILL_MODE_LWR, OP_T, DIAG_NON_UNIT, m-n, nrhs, 1.0, &l[n*lda+n], lda, &soln[n], ldsoln);
         cholesky_solve_bwd(m, n, l, lda, nrhs, soln, ldsoln);

         printf("[factor_node_posdef_test] Done\n");

         T bwderr = backward_error(m, a, lda, b, 1, soln, m);
         printf("bwderr = %le\n", bwderr);

         ////////////////////////////////////////
         // Cleanup memory

         delete[] a;
         delete[] b;
         delete[] l;
         delete[] soln;

      }

   }} // namespace spldlt::tests
