#pragma once

// SpLDLT
#include "SymbolicFront.hxx"
#include "NumericFront.hxx"
#include "factor_unsym.hxx"
#include "kernels/lu_nopiv.hxx"

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
         typedef spldlt::BuddyAllocator<T,std::allocator<T>> PoolAllocator;
         // typedef spral::ssids::cpu::BuddyAllocator<T, std::allocator<T>> PoolAllocator;
         PoolAllocator pool_alloc(m*m);

         // Setup symbolic front
         SymbolicFront sfront;
         sfront.nrow = m;
         sfront.ncol = k;
         NumericFront<T, PoolAllocator> front(sfront, pool_alloc, blksz);
         front.ndelay_in = 0; // No incoming delayed columns      
         front.ndelay_out = 0;

         // Allocate factors
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

            // if (diagdom) gen_unsym_diagdom(m, a, lda);
            if (diagdom) gen_unsym_diagdomblock(m, a, lda, blksz);
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

         // Alloc blocks in the factors
         front.alloc_blocks_unsym();
         
         // Allocate contribution blocks
         front.alloc_contrib_blocks_unsym();

         if (options.pivot_method == PivotMethod::app_block) {
            // Allocate backups of blocks
            front.alloc_backup_blocks_unsym();
            // Setup colulm data
            front.alloc_cdata(); // TODO only if piv strategy is APTP
         }
      

         // Setup permutation vector
         // Row permutation
         front.perm = new int[m];
         for(int i=0; i<m; i++) front.perm[i] = i;
         // Column permutation
         front.cperm = new int[m];
         for(int i=0; i<m; i++) front.cperm[i] = i;

         
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
         // Create workspaces
         int nworkers = 1;
#if defined(SPLDLT_USE_STARPU)
         nworkers = starpu_worker_get_count();
         // #else
         //       nworkers = omp_get_num_threads();
#endif
         std::vector<ThreadStats> worker_stats(nworkers);
         std::vector<spral::ssids::cpu::Workspace> workspaces;
         const int PAGE_SIZE = 8*1024*1024; // 8 MB

         workspaces.reserve(nworkers);
         for(int i = 0; i < nworkers; ++i)
            workspaces.emplace_back(PAGE_SIZE);
         printf("[factor_node_indef_test] nworkers =  %d\n", nworkers);

         ////////////////////////////////////////
         // Register data in runtime system
#if defined(SPLDLT_USE_STARPU)
         // TODO
#endif

         ////////////////////////////////////////
         // Perform factorization

         printf("[factor_node_unsym_test] Factor..\n");
         auto start = std::chrono::high_resolution_clock::now();         

         switch (options.pivot_method) {
         case PivotMethod::app_aggressive:
            // Front factorization using restricted pivoting
            factor_front_unsym_rp(front, workspaces);
            break;
         case PivotMethod::app_block:
            factor_front_unsym_app(options, front, workspaces);
            break;
         default:
            printf("[factor_node_unsym_test] Pivot method not implemented\n");
         }
         
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
         // Check solution results

         if (check) {

            int *rperm = front.perm;
            int *cperm = front.cperm;
            
            // Print permutation matrix         
            printf("perm = \n");
            for (int i=0; i<m; ++i)
               printf(" %d ", rperm[i]);
            printf("\n");

            // Alloc lu array for storing factors
            T *lu = factor_alloc.allocate(lda*m);
            // Initialize array with zeros
            for (int j=0; j<m; ++j)
               for (int i=0; i<m; ++i)
                  lu[i+j*lda] = 0.0;

            // Copy factors from lcol into lu array
            // Copy only factors (colmuns 1 to k)
            memcpy(lu, front.lcol, lda*k*sizeof(T)); // Copy lcol to lu

            if (m > k) {
               
               // TODO
            }

            int nrhs = 1;
            int ldsoln = m;
            T *soln = new T[nrhs*ldsoln];
            
            // Setup permuted rhs 
            T *pb = new T[m];
            for (int i=0; i<m; ++i)
               for (int r=0; r<nrhs; ++r)
                  pb[r*ldsoln+i] = b[r*ldsoln+rperm[i]];

            // Copy rhs into solution vector
            for(int r=0; r<nrhs; ++r)
               memcpy(&soln[r*ldsoln], pb, m*sizeof(T));

            // Perform solve
            // Fwd substitutuion
            lu_nopiv_fwd(m, k, lu, lda, nrhs, soln, ldsoln);
            // Bwd substitutuion
            lu_nopiv_bwd(m, k, lu, lda, nrhs, soln, ldsoln);

            // Permute x
            T *psoln = new T[m];
            for (int i=0; i<m; ++i)
               for (int r=0; r<nrhs; ++r)
                  psoln[r*ldsoln+i] = soln[r*ldsoln+cperm[i]];

            // Calculate bwd error
            double bwderr = unsym_backward_error(
                  m, k, a, lda, b, nrhs, psoln, ldsoln);

            printf("bwderr = %le\n", bwderr);

         }
         
         ////////////////////////////////////////
         // Print results
         printf("factor time (s) = %e\n", 1e-9*ttotal);

         ////////////////////////////////////////
         // Cleanup memory

         if (options.pivot_method == PivotMethod::app_block) {
            // Allocate backups of blocks
            front.release_backup_blocks_unsym();
            // Free column data
            front.free_cdata();
         }

         // Cleanup data structures
         front.free_contrib_blocks_unsym();
         
         front.free_blocks_unsym();

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
