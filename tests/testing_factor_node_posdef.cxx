// SpLDLT
#include "SymbolicFront.hxx"
#include "NumericFront.hxx"
#if defined(SPLDLT_USE_STARPU)
#include "StarPU/scheduler.hxx"
#include "StarPU/kernels.hxx"
#endif

// SpLDLT tests
#include "testing_factor_node_posdef.hxx"

// STD
#include <vector>
#include <cstdio>
#include <chrono>

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/BuddyAllocator.hxx"
// SSIDS tests
#include "tests/ssids/kernels/AlignedAllocator.hxx"

namespace spldlt { namespace tests {

      template<typename T>
      int factor_node_posdef_test(int m, int n, int blksz, int ncpu, int ngpu) {

         bool failed = false;
         int ret;

         printf("[factor_node_posdef_test] m = %d, n =  %d, blksz = %d\n", m, n, blksz);

         ////////////////////////////////////////
         // Setup test matrix

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

         memcpy(front.lcol, a, lda*n*sizeof(T)); // Copy a to l

         // Allocate contribution blocks
         front.alloc_contrib_blocks();

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
         // Init factor
#if defined(SPLDLT_USE_STARPU)
         spldlt::starpu::codelet_init<T, FactorAllocator, PoolAllocator>();
         
         spldlt::starpu::register_node(front);
#endif
         
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

         // if(debug) printf("[factor_node_indef_test] factorization done\n");
      
      }


   }}
