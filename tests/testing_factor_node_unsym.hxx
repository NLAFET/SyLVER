#pragma once

// SpLDLT
#include "SymbolicFront.hxx"
#include "NumericFront.hxx"
#include "kernels/factor_unsym.hxx"

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"

namespace spldlt { 
   namespace tests {

      /// @param T working precision
      /// @param FactorAllocator memory allocator for factors
      /// @param m number of rows/columns in the front
      /// @param k number of columns to eliminate
      template<typename T, typename FactorAllocator>
      int factor_node_unsym_test(
            struct spral::ssids::cpu::cpu_factor_options& options, int m, int k, 
            int ncpu, int ngpu, bool diagdom, bool check) {

         bool failed = false;
         int blksz = options.cpu_block_size;

         printf("[factor_node_unsym_test] m = %d\n", m);
         printf("[factor_node_unsym_test] k = %d\n", k);
         printf("[factor_node_unsym_test] blksz = %d\n", blksz);

         // We don't allow these cases
         ASSERT_TRUE(k > 0);
         ASSERT_TRUE(m > 0);
         ASSERT_TRUE(ncpu > 0);
         ASSERT_TRUE(ngpu >= 0);

         ASSERT_TRUE(m >= k);

         ////////////////////////////////////////
         // Init test problem
         // Generate test matrix
         int lda = spral::ssids::cpu::align_lda<T>(m);
         T* a = nullptr;
         T* b = nullptr;

         ////////////////////////////////////////
         // Setup front

         // Setup pool allocator
         typedef spral::ssids::cpu::BuddyAllocator<T, std::allocator<T>> PoolAllocator;
         PoolAllocator pool_alloc((m-k)*(m-k));

         // Setup symbolic front
         SymbolicFront sfront;
         sfront.nrow = m;
         sfront.ncol = k;
         NumericFront<T, PoolAllocator> front(sfront, pool_alloc, blksz);
         front.ndelay_in = 0; // No incoming delayed columns      
         front.ndelay_out = 0;

         // Allocate facotrs
         FactorAllocator factor_alloc;
         size_t lenL = lda*k; // Size of L
         // Allocate L factor (size m x k). Contains U factors in the upper triangular part 
         front.lcol = factor_alloc.allocate(lenL);
         // Allocate U factor (size k x (m-k))
         int ldu = spral::ssids::cpu::align_lda<T>(k);
         size_t lenU = ldu*(m-k); // Size of U
         front.ucol = factor_alloc.allocate(lenU);
         
         // Generate factors
         if (check) {
            
            a = new T[lda*m];

            if (diagdom) gen_unsym_diagdom(m, a, lda);
            else         gen_mat(m, m, a, lda);

            b = new T[m];
            gen_unsym_rhs(m, a, lda, b);
            
            // Copy A in L
            memcpy(front.lcol, a, lda*k*sizeof(T));
            // Copy A in U
            if (m > k) {
               for (int j = 0; j < m-k; ++j)
                  memcpy(&front.ucol[j*ldu], &a[j*lda], k*sizeof(T));
            }
            
         }
         else {
            
            ASSERT_TRUE(m == k); // FIXME: does not work for non square fronts
            if (diagdom) gen_unsym_diagdom(m, front.lcol, lda);
            else         gen_mat(m, m, front.lcol, lda);
         }

         // Allocate contribution blocks
         front.alloc_contrib_blocks_unsym();

         ////////////////////////////////////////
         // Launch runtime system
#if defined(SPLDLT_USE_STARPU)
         struct starpu_conf conf;
         starpu_conf_init(&conf);
         conf.ncpus = ncpu;
         conf.sched_policy_name = "lws";
         int ret;
         ret = starpu_init(&conf);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
#endif         

         ////////////////////////////////////////
         // Setup factor
#if defined(SPLDLT_USE_STARPU)
         // TODO
#endif

         ////////////////////////////////////////
         // Perform factorization

         printf("[factor_node_unsym_test] Factor..\n");
         auto start = std::chrono::high_resolution_clock::now();         
         

         // Wait for completion
#if defined(SPLDLT_USE_STARPU)
         starpu_task_wait_for_all();      
#endif
         auto end = std::chrono::high_resolution_clock::now();
         long ttotal = 
            std::chrono::duration_cast<std::chrono::nanoseconds>
            (end-start).count();

         ////////////////////////////////////////
         // Shutdown runtime system
#if defined(SPLDLT_USE_STARPU)
         starpu_shutdown();
#endif

         ////////////////////////////////////////
         // Print results
         printf("factor time (s) = %e\n", 1e-9*ttotal);

         ////////////////////////////////////////
         // Cleanup memory

         // Cleanup contribution blocks
         front.free_contrib_blocks_unsym();

         // Cleanup factors
         factor_alloc.deallocate(front.lcol, lenL);
         factor_alloc.deallocate(front.ucol, lenU);

         if (check) {
            delete[] a;
            delete[] b;
         }
         
         return failed ? -1 : 0;
      }

   } // namespace tests
} // namespace spldlt
