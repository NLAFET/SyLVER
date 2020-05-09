/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "assemble.hxx"
#include "BuddyAllocator.hxx"
#include "factor.hxx"
#include "NumericFront.hxx"
#if defined(SPLDLT_USE_STARPU)
#include "StarPU/codelets_posdef.hxx"
#endif
#include "sylver_ciface.hxx"
#include "tasks/assemble_block.hxx"
#include "tasks/tasks.hxx"
#include "sylver/SymbolicTree.hxx"

#include <cstddef>
#include <iostream>

#include "ssids/cpu/kernels/cholesky.hxx"
#include "ssids/cpu/Workspace.hxx"

namespace sylver {
namespace spldlt {

   template<
      typename T, // Working precision
      std::size_t PAGE_SIZE,
      typename FactorAllocator // Allocator for factor entries
      >
   class NumericTreePosdef {
      using PoolAllocator = ::sylver::BuddyAllocator<T,std::allocator<T>>;
      using NumericFrontType = NumericFront<T, FactorAllocator, PoolAllocator>;
   public:
      // Delete copy constructors for safety re allocated memory
      NumericTreePosdef(const NumericTreePosdef&) =delete;
      NumericTreePosdef& operator=(const NumericTreePosdef&) =delete;

      // Note: scaling might be usful even in a positive definite
      // context
      NumericTreePosdef(
            void* fkeep, 
            sylver::SymbolicTree& symbolic_tree, 
            T *aval,
            T* scaling,
            void** child_contrib,
            sylver::options_t& options,
            sylver::inform_t& inform)
         : fkeep_(fkeep), symb_(symbolic_tree),
           factor_alloc_(symbolic_tree.get_factor_mem_est(1.1)),
           pool_alloc_(symbolic_tree.get_pool_size<T>())
      {
         // Blocking size
         int const blksz = options.nb;

         // Create numeric fronts and associate symbolic fronts to
         // numeric ones. Copy tree structure.
         fronts_.reserve(symbolic_tree.nnodes()+1);
         for(int ni=0; ni<symb_.nnodes()+1; ++ni) {
            fronts_.emplace_back(symbolic_tree[ni], factor_alloc_, pool_alloc_, blksz);
            auto* fc = symbolic_tree[ni].first_child;
            fronts_[ni].first_child = fc ? &fronts_[fc->idx] : nullptr;
            auto* nc = symbolic_tree[ni].next_child;
            fronts_[ni].next_child = nc ? &fronts_[nc->idx] :  nullptr;
         }

         // Number of workers involved in the factorization
         int nworkers = 1;
#if defined(SPLDLT_USE_STARPU)
         // nworkers = starpu_cpu_worker_get_count();
         nworkers = starpu_worker_get_count();
#endif

         // std::cout << "[NumericTree] print_level = " << options.print_level << std::endl;
         if (options.print_level > 1) {
            printf("[NumericTree] nworkers = %d\n", nworkers);         
            printf("[NumericTree] blksz = %d\n", blksz);
            std::cout << "[NumericTree] action = " << options.action << std::endl;
         }

         std::vector<sylver::inform_t> worker_stats(nworkers);
         std::vector<spral::ssids::cpu::Workspace> workspaces;
         // Prepare workspaces
         workspaces.reserve(nworkers);
         for(int i = 0; i < nworkers; ++i)
            workspaces.emplace_back(PAGE_SIZE);

#if defined(SPLDLT_USE_STARPU)
         // TODO: make sure workspaces are not needed for posdef
         // factor.
         //
         // Register workspace handle which is currently only used
         // forCUDA kernels
         //
         // starpu_matrix_data_register (
         //       &spldlt::starpu::workspace_hdl,
         //       -1, (uintptr_t) NULL,
         //       blksz, blksz, blksz,
         //       sizeof(T));

         // Initialize StarPU codelets
         sylver::spldlt::starpu::codelets_init_posdef
            <T, FactorAllocator, PoolAllocator>();
#endif

         // Launch factorization
         auto start = std::chrono::high_resolution_clock::now();
         factor_mf(aval, scaling, child_contrib, workspaces, options,
                   worker_stats);
         auto end = std::chrono::high_resolution_clock::now();
         long ttotal = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
         if (options.print_level > 1) printf("[NumericTree] Task submission: %e\n", 1e-9*ttotal);

#if defined(SPLDLT_USE_STARPU)
         starpu_task_wait_for_all();

         // starpu_data_unregister(spldlt::starpu::workspace_hdl);
#endif

         // Initialize inform
         inform = sylver::inform_t();
         // Reduce thread_stats
         for(auto tinform : worker_stats)
            inform += tinform;
         if(inform.flag < 0) return;
         
      }

   private:

      // Multifrontal Cholesky factorization 
      void factor_mf(
            T *aval, // Numerical values of matrix A
            T *scaling, void** child_contrib,
            std::vector<spral::ssids::cpu::Workspace>& workspaces,
            sylver::options_t& options, // SyLVER options
            std::vector<sylver::inform_t>& worker_stats // factorizatino statistics
            ) {

         // printf("[factor_mf_posdef] nparts = %d\n", symb_.nparts_);

         // Blocking size
         int const blksz = options.nb;

#if defined(SPLDLT_USE_STARPU)
         // TODO move hdl registration to activate task
         for(int ni = 0; ni < symb_.nnodes(); ++ni) {
            // Register symbolic handles on node
            // starpu_void_data_register(&(symb_[ni].hdl));
            fronts_[ni].register_symb();
            // Register symbolic handle for contribution block
            // starpu_void_data_register(&(fronts_[ni].contrib_hdl()));
            fronts_[ni].register_symb_contrib();
         }
#endif


#if defined(SPLDLT_USE_STARPU) && defined(SPLDLT_USE_OMP)

         struct starpu_cluster_machine *clusters;
         clusters = starpu_cluster_machine(
               // HWLOC_OBJ_SOCKET, 
               // HWLOC_OBJ_NUMANODE,
               HWLOC_OBJ_MACHINE,
               // STARPU_CLUSTER_PARTITION_ONE, STARPU_CLUSTER_NB, 2,
               STARPU_CLUSTER_TYPE, STARPU_CLUSTER_OPENMP,
               0);
         // printf("[factor_mf_posdef] machine id = %d\n", clusters->id);
         starpu_cluster_print(clusters);
         // starpu_uncluster_machine(clusters);
         auto subtree_start = std::chrono::high_resolution_clock::now();                  

#endif
         
         for(int p = 0; p < symb_.nsubtrees(); ++p) {
            int root = symb_.subtrees()[p]-1; // subtrees is 1-indexed
            // printf("[factor_mf] nsubtrees = %d, p = %d, root = %d\n", symb_.nsubtrees_, p, root);
            factor_subtree_task(
                  symb_.akeep(), fkeep_, symb_[root], aval, scaling, p, 
                  child_contrib, &options, worker_stats);
         }

#if defined(SPLDLT_USE_STARPU) && defined(SPLDLT_USE_OMP)

         starpu_task_wait_for_all(); // FIXME Can we avoid this synchronization
         
         starpu_uncluster_machine(clusters);
         auto subtree_end = std::chrono::high_resolution_clock::now();                  
         long t_subtree = std::chrono::duration_cast<std::chrono::nanoseconds>(subtree_end-subtree_start).count();
         if (options.print_level > 1) printf("[factor_mf_posdef] factor subtrees: %e\n", 1e-9*t_subtree);

#endif

         // Allocate mapping array
         int *map = new int[symb_.n+1];

         // Loop over node in the assemnly tree
         for(int ni = 0; ni < symb_.nnodes(); ++ni) {
            
            NumericFrontType& front = fronts_[ni];
            sylver::SymbolicFront& sfront = symb_[ni];
            
            // Skip iteration if node is in a subtree
            if (sfront.is_in_subtree()) continue;
            
            // Activate frontal matrix
            // activate_front(true, front, child_contrib, factor_alloc_);
            front.activate_posdef();
            
            // Initialize frontal matrix 
            // init_node(sfront, fronts_[ni], aval); // debug
            init_node_task(front, aval, scaling, sylver::spldlt::INIT_PRIO);
            // sylver_task_wait_for_all();

            // build lookup vector, allowing for insertion of delayed vars
            // Note that while rlist[] is 1-indexed this is fine so long as lookup
            // is also 1-indexed (which it is as it is another node's rlist[]
            for(int i=0; i<sfront.ncol; i++)
               map[ sfront.rlist[i] ] = i;
            for(int i=sfront.ncol; i<sfront.nrow; i++)
               map[ sfront.rlist[i] ] = i + fronts_[ni].ndelay_in();

            // Assemble front: fully-summed columns 
            for (auto* child=fronts_[ni].first_child; child!=NULL; child=child->next_child) {
           
               sylver::SymbolicFront& child_sfront = symb_[child->symb().idx]; // Children symbolic node

               int ldcontrib = child_sfront.nrow - child_sfront.ncol;
               // Handle expected contributions (only if something there)
               if (ldcontrib>0) {

                  int cm = child_sfront.nrow - child_sfront.ncol;
                  // int* cache = work.get_ptr<int>(cm); // TODO move cache array
                  // Compute column mapping from child front into parent 
                  child_sfront.map = new int[cm];
                  for (int i=0; i<cm; i++)
                     child_sfront.map[i] = map[ child_sfront.rlist[child_sfront.ncol+i] ];

                  // Skip iteration if child node is in a subtree
                  if (child_sfront.exec_loc != -1) {

                     // Assemble contribution block from subtrees into
                     // fully-summed coefficients
                     assemble_subtree_task(
                           fronts_[ni], child_sfront, child_contrib, 
                           child_sfront.contrib_idx, child_sfront.map, 
                           sylver::spldlt::ASSEMBLE_PRIO);
                     
                  }
                  else {

                     int csa = child_sfront.ncol / blksz;
                     int cnr = (child_sfront.nrow-1) / blksz + 1; // number of block rows
                     // Loop over blocks in contribution blocks
                     for (int jj = csa; jj < cnr; ++jj) {
                        for (int ii = jj; ii < cnr; ++ii) {
                           // assemble_block(fronts_[ni], *child, ii, jj, child_sfront.map);
                           assemble_block_task(
                                 fronts_[ni], *child,
                                 ii, jj, 
                                 child_sfront.map,
                                 sylver::spldlt::ASSEMBLE_PRIO);
                        }
                     }
                  }
               }
            } // Loop over child nodes
            // sylver_task_wait_for_all();

            // Compute factors and Schur complement
            factor_front_posdef(fronts_[ni], options, worker_stats);
            // sylver_task_wait_for_all();

            // Assemble front: non fully-summed columns i.e. contribution block 
            for (auto* child=fronts_[ni].first_child; child!=NULL; child=child->next_child) {
               
               // SymbolicNode const& csnode = child->symb;
               sylver::SymbolicFront& child_sfront = symb_[child->symb().idx];
               
               int ldcontrib = child_sfront.nrow - child_sfront.ncol;
               // Handle expected contributions (only if something there)
               // if (child->contrib) {
               if (ldcontrib>0) {
                  // Skip iteration if child node is in a subtree
                  if (child_sfront.is_in_subtree()) {
                     
                     // Assemble contribution block from subtrees into non
                     // fully-summed coefficients
                     assemble_contrib_subtree_task(
                           fronts_[ni], child_sfront, child_contrib, 
                           child_sfront.contrib_idx, child_sfront.map, 
                           sylver::spldlt::ASSEMBLE_PRIO);

                  }
                  else {
                     
                     // int cm = csnode.nrow - csnode.ncol;
                     // int* cache = work.get_ptr<int>(cm); // TODO move cache array
                     // for (int i=0; i<cm; i++)
                     // csnode.map[i] = map[ csnode.rlist[csnode.ncol+i] ];

                     int csa = child_sfront.ncol / blksz;
                     // Number of block rows in child node
                     int cnr = (child_sfront.nrow-1) / blksz + 1; 
                     // int cnc = (csnode.ncol-1) / blksz + 1; // number of block columns in child node
                     // Lopp over blocks in contribution blocks
                     for (int jj = csa; jj < cnr; ++jj) {                     
                        // int c_sa = (csnode.ncol > jj*blksz) ? 0 : (jj*blksz-csnode.ncol); // first col in block
                        // int c_en = std::min((jj+1)*blksz-csnode.ncol, cm); // last col in block
                        // assemble_expected_contrib(c_sa, c_en, nodes_[ni], *child, map, cache);
                        for (int ii = jj; ii < cnr; ++ii) {
                           // assemble_contrib_block(nodes_[ni], *child, ii, jj, csnode.map, blksz)
                           assemble_contrib_block_task(
                                 fronts_[ni], *child, ii, jj, 
                                 child_sfront.map, workspaces,
                                 sylver::spldlt::ASSEMBLE_PRIO);
                        }
                     }
                  }
               }

               if (!child_sfront.is_in_subtree()) {
                  // fini_node(*child);
                  fini_node_task(*child, true);
               }
#if defined(SPLDLT_USE_STARPU)
               // Unregister symbolic handle on child node
               child->unregister_submit_symb();
               child->unregister_submit_symb_contrib();
#endif


            } // Loop over child nodes

         } // Loop over nodes in the assemnly tree

         // Finish root node
         NumericFrontType& front = fronts_[symb_.nnodes()];
         for (auto* child=front.first_child; child!=NULL; child=child->next_child) {
            sylver::SymbolicFront const& child_sfront = symb_[child->symb().idx];
            if (!child_sfront.is_in_subtree()) {
               // fini_node(*child);
               fini_node_task(*child, true);
            }
#if defined(SPLDLT_USE_STARPU)
            // Unregister symbolic handle on child node
            child->unregister_submit_symb();
            child->unregister_submit_symb_contrib();
#endif
         }

      }

   public:
      
      void solve_fwd(int nrhs, double* x, int ldx) const {

         // Allocate memory
         double* xlocal = new double[nrhs*symb_.n];
        
         // Main loop
         for(int ni=0; ni<symb_.nnodes(); ++ni) {

            // Skip iteration if node is in a subtree
            if (symb_[ni].exec_loc != -1) continue;

            int m = symb_[ni].nrow;
            int n = symb_[ni].ncol;
            int nelim = n;
            int ndin = 0;
            int ldl = align_lda<T>(m);

            // posdef there is no permutation
            int const *map = symb_[ni].rlist;

            // Gather into dense vector xlocal
            // FIXME: don't bother copying elements of x > m, just use beta=0
            //        in dgemm call and then add as we scatter
            for(int r = 0; r < nrhs; ++r) {
               for(int i = 0; i < m; ++i) {
                  xlocal[r*symb_.n+i] = x[r*ldx + map[i]-1]; // Fortran indexed
               }
            }
            
            // Perform dense solve
            spral::ssids::cpu::cholesky_solve_fwd(
                  m, n, fronts_[ni].lcol, ldl, nrhs, xlocal, symb_.n);

            // tests/debug
            // spral::ssids::cpu::cholesky_solve_fwd(
            //       m+ndin, nelim, fronts_[ni].lcol, ldl, nrhs, xlocal, symb_.n);


            /* Scatter result */
            for(int r=0; r<nrhs; ++r) {
               for(int i=0; i<m; ++i) {
                  x[r*ldx + map[i]-1] = xlocal[r*symb_.n+i];
               }
            }
         }

         /* Cleanup memory */
         delete[] xlocal;
      }

      void solve_bwd(
            int nrhs, double* x, int ldx) const {
         
         // Allocate memory - map only needed for indef bwd/diag_bwd solve
         double* xlocal = new double[nrhs*symb_.n];
         int* map_alloc = nullptr;

         // Perform solve
         for(int ni=symb_.nnodes()-1; ni>=0; --ni) {
            
            // Skip iteration if node is in a subtree
            if (symb_[ni].exec_loc != -1) continue;

            int m = symb_[ni].nrow;
            int n = symb_[ni].ncol;
            int nelim = n;

            // Mapping array: no permutation
            int const *map = symb_[ni].rlist;

            /* Gather into dense vector xlocal */
            int blkm = m;
            int ldl = align_lda<T>(m);
            for(int r=0; r<nrhs; ++r)
               for(int i=0; i<blkm; ++i)
                  xlocal[r*symb_.n+i] = x[r*ldx + map[i]-1];

            // Perform dense solve
            spral::ssids::cpu::cholesky_solve_bwd(
                  m, n, fronts_[ni].lcol, ldl, nrhs, xlocal, symb_.n);

            // tests/debug
            // spral::ssids::cpu::cholesky_solve_bwd(
            //       m+ndin, nelim, fronts_[ni].lcol, ldl, nrhs, xlocal, symb_.n);

            /* Scatter result (only first nelim entries have changed) */
            for(int r=0; r<nrhs; ++r)
               for(int i=0; i<nelim; ++i)
                  x[r*ldx + map[i]-1] = xlocal[r*symb_.n+i];
         }

         /* Cleanup memory */
         delete[] xlocal;
      }

   private:
      void* fkeep_;
      sylver::SymbolicTree& symb_;
      std::vector<NumericFrontType> fronts_;
      FactorAllocator factor_alloc_; // Allocator specific to
      // permanent memory
      PoolAllocator pool_alloc_; // Allocator specific to temporay
      // memory e.g. contribution blocks.

   };
   
}} // End of namespace sylver::spldlt
