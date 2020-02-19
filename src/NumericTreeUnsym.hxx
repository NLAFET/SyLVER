/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "BuddyAllocator.hxx"
#include "sylver_ciface.hxx"
#include "SymbolicTree.hxx"
#include "SymbolicFront.hxx"
#include "NumericFront.hxx"
#include "tasks/tasks_unsym.hxx"

// STD 
#include <stdexcept>
#include <vector>

// SSIDS
#include "ssids/cpu/ThreadStats.hxx"
// StarPU
#if defined(SPLDLT_USE_STARPU)
#include <starpu.h>
#include "StarPU/kernels_unsym.hxx"
#endif

namespace sylver { 
namespace splu {

   /// @brief Represents the matrix factors organised in an
   /// assembly tree.
   template<typename T, typename FactorAllocator, bool diagdom, size_t PAGE_SIZE>
   class NumericTreeUnsym {
      typedef spldlt::BuddyAllocator<T,std::allocator<T>> PoolAllocator;
   public:
      // Delete copy constructors for safety re allocated memory
      NumericTreeUnsym(const NumericTreeUnsym&) =delete;
      NumericTreeUnsym& operator=(const NumericTreeUnsym&) =delete;

      /// @brief Compute the factors associated with the given
      /// symbolic tree.
      ///
      /// @param val Values of A.
      /// @param scaling Scaling vector (NULL if none) 
      NumericTreeUnsym(
            sylver::SymbolicTree& symbolic_tree, 
            T *val,
            T *scaling,
            struct options_c &options
            )
         : symb_(symbolic_tree),
           factor_alloc_(symbolic_tree.get_factor_mem_est(1.1)),
           pool_alloc_(symbolic_tree.get_pool_size<T>())
      {

         // Blocking size
         int blksz = options.nb;

         printf("[NumericTreeUnsym] u = %e, small = %e\n", options.u, options.small);
         printf("[NumericTreeUnsym] nb = %d\n", options.nb);

         // Workers workspaces
         std::vector<spral::ssids::cpu::Workspace> workspaces;
         // Workers Stats
         std::vector<spral::ssids::cpu::ThreadStats> worker_stats;

         // Initialize factorization
         factorize_init(blksz, workspaces, worker_stats);
            
         // Compute factors. This call is asynchronous 
         if (diagdom) factor_mf_diagdom(val, scaling);
         else         factor_mf();

#if defined(SPLDLT_USE_STARPU)
         starpu_task_wait_for_all();
#endif
            
         factorize_fini(worker_stats);
                     
      }

      /// @Brief Factorize assembly tree using a multifrontal mode
      /// in the case of diagonally dominant matrix.
      void factor_mf_diagdom(
            T *val, 
            T *scaling) {
         
         printf("[factor_mf_diagdom]\n");
            
#if defined(SPLDLT_USE_STARPU)
         // TODO move hdl registration to activate task

         // Register symbolic handles prior to the factorization
         for(int ni = 0; ni < symb_.nnodes()+1; ++ni) {
            // Register node symbolic handle in StarPU (fully summed
            // part)
            fronts_[ni].register_symb();
            // Register contribution block symbolic handle in StarPU
            fronts_[ni].register_symb_contrib();
            // starpu_void_data_register(&(symb_[ni].hdl)); // Register node symbolic handle
            // starpu_void_data_register(&(fronts_[ni].contrib_hdl())); // Register contribution block symbolic handle
         }
#endif
            
         for(int ni = 0; ni < symb_.nnodes()+1; ++ni) {
               
            sylver::SymbolicFront& sfront = symb_[ni];
            spldlt::NumericFront<T,PoolAllocator>& front = fronts_[ni];

            // Skip if current node is within a subtree 
            if (sfront.is_in_subtree()) continue;

            activate_init_front_unsym_task(front, factor_alloc_, true);
               
         }
            
      }

      void factor_mf() {
         throw std::runtime_error("Oops, not Implemented yet\n");
      }

   private:

      /// @brief Initialize structure for storing the fronts
      /// 
      /// @param blksz Block size 
      void factorize_init(
            int blksz, 
            std::vector<spral::ssids::cpu::Workspace>& workspaces,
            std::vector<spral::ssids::cpu::ThreadStats>& worker_stats) {

         printf("[factorize_init]\n");

         // 
         // Init fronts
         //

         // Associate symbolic fronts to numeric ones; copy tree structure
         fronts_.reserve(symb_.nnodes()+1);

         for(int ni=0; ni<symb_.nnodes()+1; ++ni) {

            fronts_.emplace_back(symb_[ni], pool_alloc_, blksz);

            auto* fc = symb_[ni].first_child;
            fronts_[ni].first_child = fc ? &fronts_[fc->idx] : nullptr;
            auto* nc = symb_[ni].next_child;
            fronts_[ni].next_child = nc ? &fronts_[nc->idx] :  nullptr;
         }

         //
         // Init workspaces
         // 

         int nworkers = 1;
#if defined(SPLDLT_USE_STARPU)
         // nworkers = starpu_cpu_worker_get_count();
         nworkers = starpu_worker_get_count();
#endif
         printf("[factorize_init] nworkers = %d\n", nworkers);         
         printf("[factorize_init] blksz = %d\n", blksz);         

         // Prepare workspaces
         workspaces.reserve(nworkers);
         for(int i = 0; i < nworkers; ++i)
            workspaces.emplace_back(PAGE_SIZE);

#if defined(SPLDLT_USE_STARPU)
         // Register worksapce handle which is currently only used for
         // CUDA kernels
         starpu_matrix_data_register (
               &sylver::splu::starpu::workspace_hdl,
               -1, (uintptr_t) NULL,
               blksz, blksz, blksz,
               sizeof(T));
#endif

         //
         // Init stats
         // 

         worker_stats.reserve(nworkers);
         for(int i = 0; i < nworkers; ++i)
            worker_stats.emplace_back();
      }

      /// @brief Terminate factorization
      void factorize_fini(
            std::vector<spral::ssids::cpu::ThreadStats>& worker_stats) {
            
#if defined(SPLDLT_USE_STARPU)
         starpu_data_unregister(sylver::splu::starpu::workspace_hdl);
#endif
            
         // Reduce thread_stats
         spral::ssids::cpu::ThreadStats stats = spral::ssids::cpu::ThreadStats(); // initialise
         for(auto tstats : worker_stats)
            stats += tstats;
         if(stats.flag < 0) return;
            
         for(int ni=0; ni<symb_.nnodes(); ni++) {
            int m = symb_[ni].nrow + fronts_[ni].ndelay_in();
            stats.maxfront = std::max(stats.maxfront, m);
            // int n = symb_[ni].ncol + fronts_[ni].ndelay_in;
            // int ldl = align_lda<T>(m);
            // T *d = fronts_[ni].lcol + n*ldl;
            // for(int i=0; i<fronts_[ni].nelim; ) {
            //    T a11 = d[2*i];
            //    T a21 = d[2*i+1];
            //    if(i+1==fronts_[ni].nelim || std::isfinite(d[2*i+2])) {
            //       // 1x1 pivot (or zero)
            //       if(a11 == 0.0) {
            //          // NB: If we reach this stage, options.action must be true.
            //          stats.flag = spral::ssids::cpu::Flag::WARNING_FACT_SINGULAR;
            //          stats.num_zero++;
            //       }
            //       if(a11 < 0.0) stats.num_neg++;
            //       i++;
            //    } else {
            //       // 2x2 pivot
            //       T a22 = d[2*i+3];
            //       stats.num_two++;
            //       T det = a11*a22 - a21*a21; // product of evals
            //       T trace = a11 + a22; // sum of evals
            //       if(det < 0) stats.num_neg++;
            //       else if(trace < 0) stats.num_neg+=2;
            //       i+=2;
            //    }
            // }
         }
      }
         
   private:
      sylver::SymbolicTree& symb_; ///< Structure holding symbolic factorization data 
      std::vector<spldlt::NumericFront<T,PoolAllocator>> fronts_; // Vector
      // containing frontal matrices
      FactorAllocator factor_alloc_; ///< Allocator specific to
      // permanent memory
      PoolAllocator pool_alloc_; ///< Allocator specific to temporay
      // memory e.g. contribution blocks.
   };

}} // End of namespace sylver::splu
