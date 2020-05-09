#pragma once

// SyLVER
#include "BuddyAllocator.hxx"
#include "sylver/SymbolicFront.hxx"
#include "factor_indef.hxx"
#include "sylver/StarPU/hlws.hxx"
#include "sylver/StarPU/starpu.hxx"
// SpLDLT tests
#include "common.hxx"

// STD
#include <vector>
#include <cstdio>
#include <chrono>
#include <iostream>

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/kernels/ldlt_tpp.hxx"
#include "ssids/cpu/kernels/ldlt_app.hxx"
// SSIDS tests
#include "tests/ssids/kernels/framework.hxx"
#include "tests/ssids/kernels/AlignedAllocator.hxx"

#if defined(SPLDLT_USE_STARPU)
#include <starpu.h>
#if defined(SPLDLT_USE_GPU)
#include <starpu_cublas.h>
#include <starpu_cublas_v2.h>
#endif
#include "StarPU/codelets.hxx"
#include "StarPU/factor_failed.hxx"
#include "StarPU/kernels.hxx"
#include "StarPU/kernels_indef.hxx"
#include "StarPU/scheduler.h"
#endif

namespace spldlt {
namespace tests {
   
template<
   typename T,
   int iblksz=sylver::spldlt::INNER_BLOCK_SIZE,
   bool debug = false>
int factor_node_indef_test(
      T u, T small, bool posdef, bool delays, bool singular, int m, int n, 
      int blksz, int ncpu, int ngpu=0, int test=0, int seed=0) {
   
   bool failed = false;

   std::cout << "[factor_node_indef_test] "
             << m << " x " << n << ", blksz = " << blksz
             << ", posdef = " << posdef
             << std::endl;
      
   // We don't allow these cases
   ASSERT_TRUE(n > 0);
   ASSERT_TRUE(m > 0);
   ASSERT_TRUE(m >= n);
         
   // Generate test matrix
   int lda = spral::ssids::cpu::align_lda<T>(m);
   T* a = new double[m*lda];

   if (posdef) sylver::tests::gen_posdef(m, a, lda);
   else        sylver::tests::gen_sym_indef(m, a, lda);

   modify_test_matrix(singular, delays, m, n, a, lda);

   // Generate a RHS based on x=1, b=Ax
   T *b = new T[m];
   sylver::tests::gen_rhs(m, a, lda, b);

   // Print out matrices if requested
   if(debug) {
      std::cout << "A:" << std::endl;
      spldlt::tests::print_mat("%10.2e", m, a, lda);
   }

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
   // options.failed_pivot_method = FailedPivotMethod::pass;
         
   // Setup allocator for factors
   using FactorAllocator = spral::test::AlignedAllocator<T>;
   // Pool allocator type
   using PoolAllocator = ::sylver::BuddyAllocator<T, std::allocator<T>>;
   // Numeric front type
   using NumericFrontType = sylver::spldlt::NumericFront<T, FactorAllocator, PoolAllocator>;
   

   FactorAllocator factor_alloc;
   PoolAllocator pool_alloc(lda*n);

   sylver::SymbolicFront sfront;
   sfront.nrow = m;
   sfront.ncol = n;
   NumericFrontType front(sfront, factor_alloc, pool_alloc, blksz);

   // Init node


   // Make lcol m columns wide for debugging
   size_t len = (lda+2)*m; // Includes D
   if (debug) printf("m = %d, n = %d, lda = %d, len = %zu\n", m, n, lda, len);
   front.lcol = factor_alloc.allocate(len);

   memcpy(front.lcol, a, lda*n*sizeof(T)); // Copy a to l
   // Put nans on the bottom right corner of the LCOL matrix
   // for (int j = n; j < m; ++j) {
   //    for (int i = j; i < m; ++i) {
   //       front.lcol[lda*j+i] = std::numeric_limits<T>::signaling_NaN(); 
   //    }
   // }
      
   if (debug) {
      std::cout << "LCOL:" << std::endl;
      spldlt::tests::print_mat("%10.2e", m, front.lcol, lda);
   }
      
   // Setup permutation vector
   front.perm = new int[m];
   for(int i=0; i<m; i++) front.perm[i] = i;
   T *d = &front.lcol[lda*n];

   // Backup type: copy backup
   using Backup = sylver::CopyBackup<T, PoolAllocator>;

   // CopyBackup<T, PoolAllocator> backup(m, n, blksz);
   front.alloc_backup(); // TODO only if piv strategy is APTP
   // Setup cdata
   front.alloc_cdata(); // TODO only if piv strategy is APTP
   // Allocate block structure
   front.alloc_blocks();
   // Allocate contribution blocks
   front.alloc_contrib_blocks();

   // Initialize solver (tasking system in particular)
#if defined(SPLDLT_USE_STARPU)

   sylver::starpu::StarPU::ncpu = ncpu;
   sylver::starpu::StarPU::ncuda = ngpu;

//    struct starpu_conf *conf = new starpu_conf;
//    starpu_conf_init(conf);
//    conf->ncpus = ncpu;
// #if defined(SPLDLT_USE_GPU)
//    conf->ncuda = ngpu;

//    // Select scheduling strategy in StarPU
//    if (ngpu > 0) {
//       conf->sched_policy_name = "lws";
//       // conf->sched_policy_name = "heteroprio";
//       // conf->sched_policy_init = &init_heteroprio;
//    }
//    else{
//       // If no GPU is enabled, use Eager scheduler
//       conf->sched_policy_name = "eager"; // FIXME: use lws scheduler?
//    }
         
// #else
//    conf->sched_policy_name = "lws";
// #endif
//    int ret = starpu_init(conf);
//    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
//    delete conf;
// #if defined(SPLDLT_USE_GPU)
//    starpu_cublas_init();
// #endif

   // sylver::starpu::StarPU::conf.sched_policy_name = "eager";
   // sylver::starpu::StarPU::conf.sched_policy_name = NULL;
   // sylver::starpu::StarPU::conf.sched_policy =
   //    &sylver::starpu::HeteroLwsScheduler::starpu_sched_policy();
   
   sylver::starpu::StarPU::initialize();
   
#endif

   int nworkers = 1;
#if defined(SPLDLT_USE_STARPU)
   nworkers = starpu_worker_get_count();
   // #else
   //       nworkers = omp_get_num_threads();
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

   // Setup workspaces and thread stats
   std::vector<sylver::inform_t> worker_stats(nworkers);
   std::vector<spral::ssids::cpu::Workspace> workspaces;
   const int PAGE_SIZE = 8*1024*1024; // 8 MB

   workspaces.reserve(nworkers);
   for(int i = 0; i < nworkers; ++i)
      workspaces.emplace_back(PAGE_SIZE);
   if(debug) printf("[factor_node_indef_test] nworkers =  %d\n", nworkers);

   // Register worksapce handle
   starpu_matrix_data_register (
         &sylver::spldlt::starpu::workspace_hdl, -1, (uintptr_t) NULL, blksz, blksz, blksz,
         sizeof(T));

   // Init factorization 
#if defined(SPLDLT_USE_STARPU)
   sylver::spldlt::starpu::codelet_init<T, FactorAllocator, PoolAllocator>();
   sylver::spldlt::starpu::codelet_init_indef<T, iblksz, Backup, FactorAllocator, PoolAllocator>();
   sylver::spldlt::starpu::codelet_init_factor_indef<NumericFrontType, PoolAllocator>();
   sylver::spldlt::starpu::codelet_init_factor_failed<NumericFrontType>();

   // if (ngpu > 0) {
   //    // extern struct starpu_codelet cl_update_contrib_block_app;

   //    // Force update_contrib taks on the CPU/GPU
   //    // cl_update_contrib_block_app.where = STARPU_CPU;
   //    cl_update_contrib_block_app.where = STARPU_CUDA;
      
   //    // Force UpdateN taks on the CPU/GPU
   //    // cl_updateN_block_app.where = STARPU_CPU; 
   //    // cl_updateN_block_app.where = STARPU_CUDA;
   //    cl_updateN_block_app.where = STARPU_CUDA | STARPU_CPU;
   // }
#endif

#if defined(SPLDLT_USE_STARPU)
   // Register symbolic handles on node
   front.register_symb();
   // Register symbolic handle for contribution block
   front.register_symb_contrib();
   // Register StarPU data handles
   sylver::spldlt::starpu::register_node_indef(front);
#endif
      
   if(debug) printf("[factor_node_indef_test] factor front..\n");

   auto start = std::chrono::high_resolution_clock::now();
      
   // Factor front (first and second pass) and from contrib blocks
   sylver::spldlt::factor_front_indef(front, workspaces, pool_alloc, options, worker_stats);

   // By default function calls are asynchronous, so we put a
   // barrier and wait for the DAG to be executed
#if defined(SPLDLT_USE_STARPU)
   starpu_task_wait_for_all();      
#endif
   auto end = std::chrono::high_resolution_clock::now();
   long ttotal = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();

   if(debug) printf("[factor_node_indef_test] factorization done\n");
   if(debug) printf("[factor_node_indef_test] nelim first pass = %d\n", front.nelim_first_pass());
      
   int nelim = front.nelim(); // Number of eliminated columns
   int q1 = front.nelim_first_pass();
   int q2 = nelim - front.nelim_first_pass();
            
   std::cout << "FIRST FACTOR CALL ELIMINATED " << q1 << " of " << n << " pivots" << std::endl;
   std::cout << "SECOND FACTOR CALL ELIMINATED " << q2 << " of " << n << " pivots" << std::endl;
      
#if defined(SPLDLT_USE_STARPU)
   sylver::spldlt::starpu::unregister_node_indef(front, false); // Synchronous unregister
   // starpu_task_wait_for_all(); // Wait for unregistration of handles      
   starpu_data_unregister(sfront.hdl); // Node's symbolic handle
   starpu_data_unregister(front.contrib_hdl());

   starpu_data_unregister(sylver::spldlt::starpu::workspace_hdl);
#endif

#if defined(SPLDLT_USE_STARPU)
#if defined(SPLDLT_USE_GPU)
   // auto start_mem_pin = std::chrono::high_resolution_clock::now();
   starpu_memory_unpin(front.lcol, lda*n*sizeof(T));
   // auto end_mem_pin = std::chrono::high_resolution_clock::now();
   // long t_mem_pin = 
   //    std::chrono::duration_cast<std::chrono::nanoseconds>
   //    (end_mem_pin-start_mem_pin).count();
   // printf("[factor_node_posdef_test] memory pin (s) = %e\n", 1e-9*t_mem_pin);
#endif
#endif

   // Deinitialize solver (including shutdown tasking system)
#if defined(SPLDLT_USE_STARPU)
#if defined(SPLDLT_USE_GPU)
   starpu_cublas_shutdown();
#endif
   starpu_shutdown();
#endif

   if (debug) {
      std::cout << "CB:" << std::endl;
      print_cb("%10.2e", front);
   }

   // Alloc L array for storing factors
   T *l = factor_alloc.allocate(lda*m);
   // Initialize L with zeros
   for (int j=0; j<m; ++j)
      for (int i=j; i<m; ++i)
         l[i+j*lda] = 0.0;
      
   // Copy factors from LCOL into L array
   // Copy only factors (colmuns 1 to n)
   memcpy(l, front.lcol, lda*n*sizeof(T)); // Copy a to l

   if (debug) {
      std::cout << "LCOL:" << std::endl;
      spldlt::tests::print_mat("%10.2e", m, front.lcol, lda, front.perm);                  
      // std::cout << "L:" << std::endl;
      // print_mat("%10.2e", m, l, lda, front.perm);
   }
      
   // Eliminate remaining columns in L
   if (m > n) {

      // Copy A (columns n+1 to m) into L
      memcpy(&l[lda*n], &a[lda*n], lda*(m-n)*sizeof(T));
      // Add entries (columns n+1 to m) form CB into L
      if (front.nelim() > 0) add_cb_to_a(front, l, lda);
         
      // Debug
      // Ignore CB and do update
      // Apply outer product update
      // memcpy(&l[n*lda], &a[n*lda], lda*(m-n)*sizeof(T));
      // do_update<T>(m-n, q1+q2, &l[n*(lda+1)], &l[n], lda, d);
         
      if (debug) {
         std::cout << "D:" << std::endl;
         print_d(m, d);
         std::cout << "L:" << std::endl;
         spldlt::tests::print_mat("%10.2e", m, l, lda, front.perm);
      }

      if (debug) std::cout << "Eliminate remaining columns using TPP.." << std::endl;
            
      // Finish off with TPP
      T *ld = new T[2*m];
      nelim += spral::ssids::cpu::ldlt_tpp_factor(
            m-nelim, m-nelim, &front.perm[nelim], &l[nelim*(lda+1)], lda,
            &d[2*nelim], ld, m, options.action, u, small, nelim, &l[nelim], lda);
      delete[] ld;
   }
      
   if (debug) {
      std::cout << "D:" << std::endl;
      print_d(m, d);
      std::cout << "L:" << std::endl;
      spldlt::tests::print_mat("%10.2e", m, l, lda, front.perm);
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
   T bwderr = sylver::tests::backward_error(m, a, lda, b, 1, soln, m);
   /*if(debug)*/ printf("bwderr = %le\n", bwderr);
   EXPECT_LE(u*bwderr, 5e-14) << "(test " << test << " seed " << seed << ")" << std::endl;

   ////////////////////////////////////////
   // Print results

   double flops = ((double)m*n*n)/3.0;
   printf("[factor_node_indef_test]\n");
   printf("factor time (s) = %e\n", 1e-9*ttotal);
   printf("GFlop/s = %.3f\n", flops/(double)ttotal);


   // Cleanup memory
   factor_alloc.deallocate(l, m*lda);
   delete[] a; factor_alloc.deallocate(front.lcol, len);
   delete[] b;
   delete[] front.perm;
   delete[] soln;

   return failed ? -1 : 0;
}

   // Run tests for the node factorization
   int run_factor_node_indef_tests();

}} // namespace spldlt::tests
