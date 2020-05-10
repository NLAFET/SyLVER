/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SpLDLT
#include "NumericFront.hxx"
#include "factor.hxx"
#include "kernels/llt.hxx"
#if defined(SPLDLT_USE_STARPU)
#include "StarPU/codelets_posdef.hxx"
#include "StarPU/common.hxx"
#include "StarPU/scheduler.h"
#include "StarPU/kernels.hxx"
#endif
#include "sylver/SymbolicFront.hxx"

// STD
#include <vector>
#include <cstdio>
#include <chrono>

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/kernels/cholesky.hxx"
// SSIDS tests
#include "tests/ssids/kernels/AlignedAllocator.hxx"
#include "tests/ssids/kernels/framework.hxx"

namespace spldlt {
namespace tests {

   template<typename T>
   int factor_node_posdef_test(sylver::tests::Options test_options) {

      std::string context = "factor_node_posdef_test";
      bool failed = false;
      int ret;

      // Number of rows
      int const m = test_options.m;
      // Number of columns
      int const n = test_options.n;
      // Block dimensions
      int const blksz = test_options.nb;

      // Use Tensor Cores
      bool const usetc = test_options.usetc;

      // Check residual
      bool const check = test_options.check;

      // Number of CPUs
      int const ncpu = test_options.ncpu;
      // Number of GPUs 
      int const ngpu = test_options.ngpu;
      
      std::cout << "[" << context << "]" << " m = " << m << " n = " << n << std::endl;
      std::cout << "[" << context << "]" << " nb = " << blksz << std::endl;
      std::cout << "[" << context << "]" << " Number of CPUs = " << ncpu << std::endl;
      std::cout << "[" << context << "]" << " Number of GPUs = " << ngpu << std::endl;
      std::cout << "[" << context << "]" << " usetc = " << usetc << std::endl;

      ////////////////////////////////////////
      // Setup test matrix and rhs

      // Generate test matrix
      int lda = spral::ssids::cpu::align_lda<T>(m);
      T* a = nullptr;
      T* b = nullptr;
      if (test_options.check) {
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
      sylver::options_t options;
      options.nb = blksz;

      // Setup pool allocator
      using PoolAllocator = ::sylver::BuddyAllocator<T, std::allocator<T>>; 
      PoolAllocator pool_alloc(lda*n);
         
      // Setup frontal matrix
      sylver::SymbolicFront sfront;
      sfront.nrow = m;
      sfront.ncol = n;

      // Setup allocator for factors
      using FactorAllocator = spral::test::AlignedAllocator<T>;
      FactorAllocator allocT;

      // Numeric front type
      using NumericFrontType = sylver::spldlt::NumericFront<T, FactorAllocator, PoolAllocator>;
      NumericFrontType front(sfront, allocT, pool_alloc, blksz);

      // Allocate front, factors, contrib block and data structures
      front.allocate_posdef();
      
      if (check) {
         // Copy A into L
         memcpy(front.lcol, a, lda*n*sizeof(T));
      }
      else {
         assert(m == n); // FIXME: does not work for non square fronts
         sylver::tests::gen_posdef(m, front.lcol, lda);
      }

      ////////////////////////////////////////
      // Init runtime system
#if defined(SPLDLT_USE_STARPU)
      sylver::starpu::StarPU::ncpu = ncpu;
#if defined(SPLDLT_USE_GPU)
      sylver::starpu::StarPU::ncuda = ngpu;
#else
      sylver::starpu::StarPU::ncuda = 0;
#endif

      switch (test_options.sched) {
      case(sylver::tests::Sched::HP):
         sylver::starpu::StarPU::sched = sylver::starpu::StarPU::Scheduler::HP;
         break;
      case(sylver::tests::Sched::HLWS):
         sylver::starpu::StarPU::sched = sylver::starpu::StarPU::Scheduler::HLWS;
         break;
      case(sylver::tests::Sched::LWS):
         sylver::starpu::StarPU::sched = sylver::starpu::StarPU::Scheduler::LWS;
         break;
      case(sylver::tests::Sched::WS):
         sylver::starpu::StarPU::sched = sylver::starpu::StarPU::Scheduler::WS;
         break;
      default:
         std::runtime_error("Scheduler not available");
      }
      
      sylver::starpu::StarPU::initialize();

// #if defined(SPLDLT_USE_GPU)

//       if (ngpu > 0) {
         
//          // Scheduler
//          conf.sched_policy_name = "heteroprio";
//          conf.sched_policy_init = &init_heteroprio;

// // #if defined(HAVE_LAHP)
// //          if(getenv("USE_LAHETEROPRIO") != NULL
// //             && (strcmp(getenv("USE_LAHETEROPRIO"),"TRUE")==0||strcmp(getenv("USE_LAHETEROPRIO"),"true")==0)){
// //             printf("[starpu_f_init_c] use laheteroprio\n");
// //             conf.sched_policy_name = "laheteroprio";
// //             conf.sched_policy_init = &init_laheteroprio;
// //          }
// // #endif
//       }
//       else {
//          conf.sched_policy_name = "eager";
//       }

// #else
//       conf.sched_policy_name = "lws";
// #endif

      // ret = starpu_init(&conf);
      // STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
#if defined(SPLDLT_USE_GPU)
      // starpu_cublas_init();
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

      std::vector<sylver::inform_t> worker_stats(nworkers);

      ////////////////////////////////////////
      // Init factor
#if defined(SPLDLT_USE_STARPU)
      sylver::spldlt::starpu::codelet_init<T, FactorAllocator, PoolAllocator>();

#if defined(SPLDLT_USE_GPU)
      sylver::spldlt::starpu::cl_update_block.where = STARPU_CPU|STARPU_CUDA;
#endif
      // Register symbolic handles on node
      front.register_symb();
      // Register symbolic handle for contribution block
      front.register_symb_contrib();
      // Register data handles in StarPU         
      // sylver::spldlt::starpu::register_node(front);
      front.register_node_posdef();
#endif
         
      printf("[factor_node_posdef_test] Factor..\n");
      // Run matrix factorization
      auto start = std::chrono::high_resolution_clock::now();

      factor_front_posdef(front, options, worker_stats);

#if defined(SPLDLT_USE_STARPU)
      starpu_task_wait_for_all();      
#endif
      auto end = std::chrono::high_resolution_clock::now();
      long ttotal = 
         std::chrono::duration_cast<std::chrono::nanoseconds>
         (end-start).count();
         
      printf("[factor_node_posdef_test] Done\n");

      // if(debug) printf("[factor_node_indef_test] factorization done\n");

      sylver::spldlt::starpu::unregister_node_submit(front);
      starpu_task_wait_for_all(); // Wait for unregistration of handles      
      starpu_data_unregister(sfront.hdl); // Node's symbolic handle
      starpu_data_unregister(front.contrib_hdl());
      
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
         sylver::host_trsm<T>(
               sylver::SIDE_LEFT, sylver::FILL_MODE_LWR, 
               sylver::operation::OP_N, sylver::DIAG_NON_UNIT, 
               m-n, nrhs, 1.0, &l[n*lda+n], lda, 
               &soln[n], ldsoln);
         sylver::host_trsm<T>(
               sylver::SIDE_LEFT, sylver::FILL_MODE_LWR, 
               sylver::operation::OP_T, sylver::DIAG_NON_UNIT, 
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

      return 0;
   }

}} // namespace spldlt::tests
