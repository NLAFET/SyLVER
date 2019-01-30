/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SpLDLT
#include "SymbolicFront.hxx"
#include "NumericFront.hxx"
#include "factor.hxx"
#include "kernels/llt.hxx"
#if defined(SPLDLT_USE_STARPU)
#include "StarPU/common.hxx"
#include "StarPU/scheduler.h"
#include "StarPU/kernels.hxx"
#endif

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
#include "tests/ssids/kernels/framework.hxx"

namespace spldlt { namespace tests {

      template<typename T>
      int factor_node_posdef_test(int m, int n, int blksz, int ncpu, int ngpu, bool check, bool usetc) {

         bool failed = false;
         int ret;

         printf("[factor_node_posdef_test] m = %d, n =  %d, blksz = %d\n", m, n, blksz);
         std::cout << "[chol_test] usetc = " << usetc << std::endl;

         ////////////////////////////////////////
         // Setup test matrix and rhs

         // Generate test matrix
         int lda = spral::ssids::cpu::align_lda<T>(m);
         T* a = nullptr;
         T* b = nullptr;
         if (check) {
            a = new T[m*lda];
            sylver::tests::gen_posdef(m, a, lda);
            // Generate a RHS based on x=1, b=Ax
            b = new T[m];
            sylver::tests::gen_rhs(m, a, lda, b);
         }

         // Print out matrices if requested
         // if(debug) {
         //    std::cout << "A:" << std::endl;
         //    print_mat("%10.2e", m, a, lda);
         // }

         // Setup options
         struct cpu_factor_options options;
         options.cpu_block_size = blksz;

         // Setup pool allocator
         // typedef spldlt::BuddyAllocator<T,std::allocator<T>> PoolAllocator;
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

         auto start_alloc = std::chrono::high_resolution_clock::now();
         front.lcol = allocT.allocate(len);
         auto end_alloc = std::chrono::high_resolution_clock::now();
         long t_alloc = 
            std::chrono::duration_cast<std::chrono::nanoseconds>
            (end_alloc-start_alloc).count();
         printf("[factor_node_posdef_test] memory alloc factor (s) = %e\n", 1e-9*t_alloc);

         if (check) {
            // Copy a into l
            memcpy(front.lcol, a, lda*n*sizeof(T));
         }
         else {
            ASSERT_TRUE(m == n); // FIXME: does not work for non square fronts
            sylver::tests::gen_posdef(m, front.lcol, lda);
         }

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
         conf.sched_policy_init = &init_heteroprio;
         // conf.sched_policy_name = "ws";

#if defined(HAVE_LAHP)
         if(getenv("USE_LAHETEROPRIO") != NULL
            && (strcmp(getenv("USE_LAHETEROPRIO"),"TRUE")==0||strcmp(getenv("USE_LAHETEROPRIO"),"true")==0)){
            printf("[starpu_f_init_c] use laheteroprio\n");
            conf.sched_policy_name = "laheteroprio";
            conf.sched_policy_init = &init_laheteroprio;
         }
#endif
         
#else
         conf.sched_policy_name = "lws";
#endif

         ret = starpu_init(&conf);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
#if defined(SPLDLT_USE_GPU)
      starpu_cublas_init();
      // Select math mode
      if (usetc) sylver::starpu::enable_tc();
      else       sylver::starpu::disable_tc();
#endif
#endif

         // Get number of workers (CPU and GPU)
         int nworkers = 1;
#if defined(SPLDLT_USE_STARPU)
         nworkers = starpu_worker_get_count();
#endif

#if defined(SPLDLT_USE_STARPU)
#if defined(SPLDLT_USE_GPU)
         auto start_mem_pin = std::chrono::high_resolution_clock::now();
         starpu_memory_pin(front.lcol, lda*n*sizeof(T));
         auto end_mem_pin = std::chrono::high_resolution_clock::now();
         long t_mem_pin = 
            std::chrono::duration_cast<std::chrono::nanoseconds>
            (end_mem_pin-start_mem_pin).count();
         printf("[factor_node_posdef_test] memory pin (s) = %e\n", 1e-9*t_mem_pin);
#endif
#endif

         ////////////////////////////////////////
         // Init factor
#if defined(SPLDLT_USE_STARPU)
         spldlt::starpu::codelet_init<T, FactorAllocator, PoolAllocator>();

#if defined(SPLDLT_USE_GPU)
         cl_update_block.where = STARPU_CPU|STARPU_CUDA;
#endif
         
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

         if (check) {
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
            T *soln = new T[nrhs*ldsoln];
            for(int r=0; r<nrhs; ++r)
               memcpy(&soln[r*ldsoln], b, m*sizeof(T));

            printf("[factor_node_posdef_test] Solve..\n");
            sylver::spldlt::cholesky_solve_fwd(m, n, l, lda, nrhs, soln, ldsoln);
            host_trsm<T>(
                  sylver::SIDE_LEFT, sylver::FILL_MODE_LWR, 
                  sylver::OP_N, sylver::DIAG_NON_UNIT, 
                  m-n, nrhs, 1.0, &l[n*lda+n], lda, 
                  &soln[n], ldsoln);
            host_trsm<T>(
                  sylver::SIDE_LEFT, sylver::FILL_MODE_LWR, 
                  sylver::OP_T, sylver::DIAG_NON_UNIT, 
                  m-n, nrhs, 1.0, &l[n*lda+n], lda, 
                  &soln[n], ldsoln);
            sylver::spldlt::cholesky_solve_bwd(m, n, l, lda, nrhs, soln, ldsoln);

            printf("[factor_node_posdef_test] Done\n");

            T bwderr = sylver::tests::backward_error(m, a, lda, b, 1, soln, m);
            printf("bwderr = %le\n", bwderr);

            delete[] l;
            delete[] soln;
         }

         double flops = ((double)m*n*n)/3.0;
         printf("factor time (s) = %e\n", 1e-9*ttotal);
         printf("GFlop/s = %.3f\n", flops/(double)ttotal);

         ////////////////////////////////////////
         // Cleanup memory

         if (check) {
            delete[] a;
            delete[] b;
         }

      }

   }} // namespace spldlt::tests
