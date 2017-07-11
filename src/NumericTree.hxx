/// \file
/// \copyright 2016- The Science and Technology Facilities Council (STFC)
/// \author    Florent Lopez

#pragma once

#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/factor.hxx"
// #include "ssids/cpu/BuddyAllocator.hxx"
// #include "ssids/cpu/NumericNode.hxx"
#include "ssids/cpu/ThreadStats.hxx"
#include "ssids/cpu/kernels/cholesky.hxx"


#include "BuddyAllocator.hxx"
#include "Workspace.hxx"
#include "SymbolicSNode.hxx"
#include "SymbolicTree.hxx"
#include "NumericNode.hxx"
#include "kernels/assemble.hxx"
#include "kernels/common.hxx"
#include "tasks.hxx"

#if defined(SPLDLT_USE_STARPU)
#include <starpu.h>
#include "StarPU/kernels.hxx"
#endif

// profiling
#include <chrono>

using namespace spral::ssids::cpu;
using namespace spldlt;

#if defined(SPLDLT_USE_STARPU)
using namespace spldlt::starpu;
#endif

namespace spldlt {

   template<typename T,
            size_t PAGE_SIZE,
            typename FactorAllocator,
            bool posdef> //< true for Cholesky factoriztion, false for indefinte
   class NumericTree {
      typedef spldlt::BuddyAllocator<T,std::allocator<T>> PoolAllocator;
   public:
      // Delete copy constructors for safety re allocated memory
      NumericTree(const NumericTree&) =delete;
      NumericTree& operator=(const NumericTree&) =delete;

      // FIXME: idealy symbolic_tree should be constant but we
      // currently modify it in order to add the runtime system info
      NumericTree(SymbolicTree& symbolic_tree, T *aval, 
                  struct cpu_factor_options const& options)
         : symb_(symbolic_tree), 
           factor_alloc_(symbolic_tree.get_factor_mem_est(1.0)),
           pool_alloc_(symbolic_tree.get_pool_size<T>())
      {

         // Blocking size
         int blksz = options.cpu_block_size;

         // printf("[NumericTree] block size: %d\n",  options.cpu_block_size);
         // Associate symbolic nodes to numeric ones; copy tree structure
         nodes_.reserve(symbolic_tree.nnodes_+1);
         for(int ni=0; ni<symb_.nnodes_+1; ++ni) {
            nodes_.emplace_back(symbolic_tree[ni], pool_alloc_, blksz);
            auto* fc = symbolic_tree[ni].first_child;
            nodes_[ni].first_child = fc ? &nodes_[fc->idx] : nullptr;
            auto* nc = symbolic_tree[ni].next_child;
            nodes_[ni].next_child = nc ? &nodes_[nc->idx] :  nullptr;
         }

         // Allocate workspace
         // spldlt::Workspace work(PAGE_SIZE);
         // spldlt::Workspace colmap(PAGE_SIZE);
         // spldlt::Workspace rowmap(PAGE_SIZE);
         
         // printf("[NumericTree] nnodes: %d\n", symb_.nnodes_);

#if defined(SPLDLT_USE_STARPU)
         // Initialize factorization with StarPU
         // Init codelet
         codelet_init<T,PoolAllocator>();
         // Init scratch memory data
         // Init workspace
         // int ldw = align_lda<T>(blksz);
         // starpu_matrix_data_register(
         //       &(work.hdl), -1, 0,
         //       blksz, blksz, blksz,
         //       sizeof(T));
         // Init colmap workspace (int array)
         // starpu_vector_data_register(
         //       &(colmap.hdl), -1, 0, blksz, 
         //       sizeof(int));
         // Init rowmap workspace (int array)
         // starpu_vector_data_register(
         //       &(rowmap.hdl), -1, 0, blksz,
         //       sizeof(int));
#endif

         auto start = std::chrono::high_resolution_clock::now();
         
         // Perform the factorization of the numeric tree
         // factor(aval, work, rowmap, colmap, options);

         factor_mf(aval, options);

         auto end = std::chrono::high_resolution_clock::now();
         long ttotal = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
         printf("[NumericTree] task submission time: %e\n", 1e-9*ttotal);

#if defined(SPLDLT_USE_STARPU)
         // starpu_data_unregister_submit(work.hdl);
         // starpu_data_unregister_submit(colmap.hdl);
         // starpu_data_unregister_submit(rowmap.hdl);
#endif        

#if defined(SPLDLT_USE_STARPU)
         starpu_task_wait_for_all();
#endif
      }

      void solve_fwd(int nrhs, double* x, int ldx) const {

         // printf("[NumericTree] solve fwd, nrhs: %d\n", nrhs);
         // for (int i = 0; i < ldx; ++i) printf(" %10.4f", x[i]);
         // printf("[NumericTree] solve fwd, posdef: %d\n", posdef);

         /* Allocate memory */
         double* xlocal = new double[nrhs*symb_.n];
         int* map_alloc = (!posdef) ? new int[symb_.n] : nullptr; // only indef
        
         /* Main loop */
         for(int ni=0; ni<symb_.nnodes_; ++ni) {
            int m = symb_[ni].nrow;
            int n = symb_[ni].ncol;
            int nelim = (posdef) ? n
               : nodes_[ni].nelim;
            int ndin = (posdef) ? 0
               : nodes_[ni].ndelay_in;
            int ldl = align_lda<T>(m+ndin);
            // printf("[NumericTree] solve fwd, node: %d, nelim: %d, ldl: %d\n", ni, nelim, ldl);
            /* Build map (indef only) */
            int const *map;
            if(!posdef) {
               // indef need to allow for permutation and/or delays
               for(int i=0; i<n+ndin; ++i)
                  map_alloc[i] = nodes_[ni].perm[i];
               for(int i=n; i<m; ++i)
                  map_alloc[i+ndin] = symb_[ni].rlist[i];
               map = map_alloc;
            } else {
               // posdef there is no permutation
               map = symb_[ni].rlist;
            }

            /* Gather into dense vector xlocal */
            // FIXME: don't bother copying elements of x > m, just use beta=0
            //        in dgemm call and then add as we scatter
            for(int r=0; r<nrhs; ++r)
               for(int i=0; i<m+ndin; ++i)
                  xlocal[r*symb_.n+i] = x[r*ldx + map[i]-1]; // Fortran indexed

            /* Perform dense solve */
            if(posdef) {
               cholesky_solve_fwd(m, n, nodes_[ni].lcol, ldl, nrhs, xlocal, symb_.n);
            } else { /* indef */
               ldlt_app_solve_fwd(m+ndin, nelim, nodes_[ni].lcol, ldl, nrhs,
                                  xlocal, symb_.n);
            }

            /* Scatter result */
            for(int r=0; r<nrhs; ++r)
               for(int i=0; i<m+ndin; ++i)
                  x[r*ldx + map[i]-1] = xlocal[r*symb_.n+i];
         }

         /* Cleanup memory */
         if(!posdef) delete[] map_alloc; // only used in indef case
         delete[] xlocal;
      }

      template <bool do_diag, bool do_bwd>
      void solve_diag_bwd_inner(int nrhs, double* x, int ldx) const {
         if(posdef && !do_bwd) return; // diagonal solve is a no-op for posdef

         /* Allocate memory - map only needed for indef bwd/diag_bwd solve */
         double* xlocal = new double[nrhs*symb_.n];
         int* map_alloc = (!posdef && do_bwd) ? new int[symb_.n]
            : nullptr;

         /* Perform solve */
         for(int ni=symb_.nnodes_-1; ni>=0; --ni) {
            int m = symb_[ni].nrow;
            int n = symb_[ni].ncol;
            int nelim = (posdef) ? n
               : nodes_[ni].nelim;
            int ndin = (posdef) ? 0
               : nodes_[ni].ndelay_in;

            /* Build map (indef only) */
            int const *map;
            if(!posdef) {
               // indef need to allow for permutation and/or delays
               if(do_bwd) {
                  for(int i=0; i<n+ndin; ++i)
                     map_alloc[i] = nodes_[ni].perm[i];
                  for(int i=n; i<m; ++i)
                     map_alloc[i+ndin] = symb_[ni].rlist[i];
                  map = map_alloc;
               } else { // if only doing diagonal, only need first nelim<=n+ndin
                  map = nodes_[ni].perm;
               }
            } else {
               // posdef there is no permutation
               map = symb_[ni].rlist;
            }

            /* Gather into dense vector xlocal */
            int blkm = (do_bwd) ? m+ndin
               : nelim;
            int ldl = align_lda<T>(m+ndin);
            for(int r=0; r<nrhs; ++r)
               for(int i=0; i<blkm; ++i)
                  xlocal[r*symb_.n+i] = x[r*ldx + map[i]-1];

            /* Perform dense solve */
            if(posdef) {
               cholesky_solve_bwd(m, n, nodes_[ni].lcol, ldl, nrhs, xlocal, symb_.n);
            } else {
               if(do_diag) ldlt_app_solve_diag(
                     nelim, &nodes_[ni].lcol[(n+ndin)*ldl], nrhs, xlocal, symb_.n
                     );
               if(do_bwd) ldlt_app_solve_bwd(
                     m+ndin, nelim, nodes_[ni].lcol, ldl, nrhs, xlocal, symb_.n
                     );
            }

            /* Scatter result (only first nelim entries have changed) */
            for(int r=0; r<nrhs; ++r)
               for(int i=0; i<nelim; ++i)
                  x[r*ldx + map[i]-1] = xlocal[r*symb_.n+i];
         }

         /* Cleanup memory */
         if(!posdef && do_bwd) delete[] map_alloc; // only used in indef case
         delete[] xlocal;
      }

      void solve_diag(int nrhs, double* x, int ldx) const {
         solve_diag_bwd_inner<true, false>(nrhs, x, ldx);
      }

      void solve_diag_bwd(int nrhs, double* x, int ldx) const {
         solve_diag_bwd_inner<true, true>(nrhs, x, ldx);
      }

      void solve_bwd(int nrhs, double* x, int ldx) const {
         solve_diag_bwd_inner<false, true>(nrhs, x, ldx);
      }

   private:

      // Factorization using a multifrontal mode
      //    Note: Asynchronous routine i.e. no barrier at the end 
      void factor_mf(T *aval, struct cpu_factor_options const& options) {

         int blksz = options.cpu_block_size;

         int INIT_PRIO = 4;
         int ASSEMBLE_PRIO = 4;

         // Allocate mapping array
         // TODO use proper allocator 
         int *map = new int[symb_.n+1];

         // Loop over node in the assemnly tree
         for(int ni = 0; ni < symb_.nnodes_; ++ni) {
            
            SymbolicSNode &snode = symb_[ni];

            // Activate frontal matrix
            activate_front(snode, nodes_[ni], blksz, factor_alloc_, pool_alloc_);
            
            // Initialize frontal matrix 
            // init_node(snode, nodes_[ni], aval);
            init_node_task(snode, nodes_[ni], aval, INIT_PRIO);

            // Assemble front: fully-summed columns
            // typedef typename std::allocator_traits<PoolAllocator>::template rebind_alloc<int> PoolAllocInt;

            /* Build lookup vector, allowing for insertion of delayed vars */
            /* Note that while rlist[] is 1-indexed this is fine so long as lookup
             * is also 1-indexed (which it is as it is another node's rlist[] */
            // #if defined(SPLDLT_USE_STARPU)

// #else            
//             std::vector<int, PoolAllocInt> map(symb_.n+1, PoolAllocInt(pool_alloc_));
// #endif
            for(int i=0; i<snode.ncol; i++)
               map[ snode.rlist[i] ] = i;
            for(int i=snode.ncol; i<snode.nrow; i++)
               map[ snode.rlist[i] ] = i + nodes_[ni].ndelay_in;
            
            for (auto* child=nodes_[ni].first_child; child!=NULL; child=child->next_child) {
               
               // SymbolicNode const& csnode = child->symb;
               SymbolicSNode &csnode = symb_[child->symb.idx];

               int ldcontrib = csnode.nrow - csnode.ncol;
               // Handle expected contributions (only if something there)
               // if (child->contrib) {
               if (ldcontrib>0) {
                  
                  int cm = csnode.nrow - csnode.ncol;
                  // int* cache = work.get_ptr<int>(cm); // TODO move cache array
                  // printf("[factor_mf] cm: %d\n", cm);

                  // Compute column mapping from child front into parent 
                  csnode.map = new int[cm];
                  for (int i=0; i<cm; i++) {
                     csnode.map[i] = map[ csnode.rlist[csnode.ncol+i] ];
                     // printf("map[%d] = %d\n", i, csnode.map[i]);
                  }

                  int csa = csnode.ncol / blksz;
                  int cnr = (csnode.nrow-1) / blksz + 1; // number of block rows in child node
                  // int cnc = (csnode.ncol-1) / blksz + 1; // number of block columns in child node
                  // printf("[factor_mf] csa: %d, cnr: %d\n", csa, cnr);
                  // printf("[factor_mf] ncol: %d\n", csnode.ncol);

                  // Lopp over blocks in contribution blocks
                  for (int jj=csa; jj<cnr; ++jj) {
                     for (int ii=jj; ii<cnr; ++ii) {
                        // assemble_block(nodes_[ni], *child, ii, jj, csnode.map, blksz);
                        assemble_block_task(snode, nodes_[ni], csnode, *child, ii, jj, csnode.map, blksz, ASSEMBLE_PRIO);
                     }
                  }
               }
            }

            // Compute factors and Schur complement 
            factorize_front_posdef(snode, nodes_[ni], options);

            // Assemble front: non fully-summed columns i.e. contribution block 
            for (auto* child=nodes_[ni].first_child; child!=NULL; child=child->next_child) {
               
               // SymbolicNode const& csnode = child->symb;
               SymbolicSNode &csnode = symb_[child->symb.idx];

               int ldcontrib = csnode.nrow - csnode.ncol;
               // Handle expected contributions (only if something there)
               // if (child->contrib) {
               if (ldcontrib>0) {
                  // int cm = csnode.nrow - csnode.ncol;
                  // int* cache = work.get_ptr<int>(cm); // TODO move cache array
                  // for (int i=0; i<cm; i++)
                  // csnode.map[i] = map[ csnode.rlist[csnode.ncol+i] ];

                  int csa = csnode.ncol / blksz;
                  int cnr = (csnode.nrow-1) / blksz + 1; // number of block rows in child node
                  // int cnc = (csnode.ncol-1) / blksz + 1; // number of block columns in child node
                  // Lopp over blocks in contribution blocks
                  for (int jj = csa; jj < cnr; ++jj) {                     
                     // int c_sa = (csnode.ncol > jj*blksz) ? 0 : (jj*blksz-csnode.ncol); // first col in block
                     // int c_en = std::min((jj+1)*blksz-csnode.ncol, cm); // last col in block
                     // assemble_expected_contrib(c_sa, c_en, nodes_[ni], *child, map, cache);
                     // int ii = 0;
                     for (int ii = jj; ii < cnr; ++ii) {
                        // assemble_contrib_block(nodes_[ni], *child, ii, jj, csnode.map, blksz)
                        assemble_contrib_block_task(snode, nodes_[ni], csnode, *child, ii, jj, csnode.map, blksz, ASSEMBLE_PRIO);
                     }
                  }
               }
               
               // fini_node(*child);
               fini_node_task(csnode, *child, INIT_PRIO);      

#if defined(SPLDLT_USE_STARPU)
               unregister_node_submit(csnode, *child, blksz);

               // Unregister symbolic handle on child node
               starpu_data_unregister_submit(csnode.hdl);
#endif

// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif
            } // loop over children nodes

         } // loop over nodes

// #if defined(SPLDLT_USE_STARPU)
//             starpu_task_wait_for_all();
// #endif
      }

      /* Factorization using a supernodal mode. 
         Note: Asynchronous routine i.e. no barrier at the end. 
       */
      void factor(
            T *aval,
            Workspace &work,
            Workspace &rowmap,
            Workspace &colmap,
            struct cpu_factor_options const& options) {

         /* Task priorities:
            init: 4
            facto: 3
            solve: 2
            udpate: 1
            udpate_between: 0
         */

         int blksz = options.cpu_block_size; 

         // profiling
         std::chrono::high_resolution_clock::time_point start, end;
         long tapply = 0, tinit = 0;

         // start = std::chrono::high_resolution_clock::now();

         /* Initialize nodes because right-looking update */
         for(int ni = 0; ni < symb_.nnodes_; ++ni) {

            SymbolicSNode &snode = symb_[ni];
            
            activate_node(snode, nodes_[ni], blksz, factor_alloc_, pool_alloc_);

            //             alloc_node(snode, nodes_[ni], factor_alloc_, pool_alloc_);

            // #if defined(SPLDLT_USE_STARPU)
            //             starpu_void_data_register(&(snode.hdl));

            //             /* Register blocks in StarPU */
            //             // printf("[NumericTree] regiter node: %d\n", ni);
            //             register_node(snode, nodes_[ni], blksz);

            //             // activate_node(snode, nodes_[ni], blksz);
            // #endif
            
            init_node_task(snode, nodes_[ni], aval, 4);
         }
         
         // end = std::chrono::high_resolution_clock::now();
         // tinit = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
         
         // #if defined(SPLDLT_USE_STARPU)
         //          starpu_task_wait_for_all();
         // #endif
         
         /* Loop over singleton nodes in order */
         for(int ni = 0; ni < symb_.nnodes_; ++ni) {

            // #if defined(SPLDLT_USE_STARPU)
            //          starpu_task_wait_for_all();
            // #endif

            /* Factorize node */
            factorize_node_posdef(symb_[ni], nodes_[ni], options);

            // #if defined(SPLDLT_USE_STARPU)
            //          starpu_task_wait_for_all();
            // #endif

            // start = std::chrono::high_resolution_clock::now();
 
            // Apply factorization operation to ancestors
            apply_node(symb_[ni], nodes_[ni],
                       symb_.nnodes_, symb_, nodes_,
                       blksz, work, rowmap, colmap);
            
            // #if defined(SPLDLT_USE_STARPU)
            //          starpu_task_wait_for_all();
            // #endif
            
            // end = std::chrono::high_resolution_clock::now();
            // tapply += std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();

         }

         // printf("[NumericTree] task apply submission time: %e\n", 1e-9*tapply);
         // printf("[NumericTree] task init submission time: %e\n", 1e-9*tinit);

      }

      SymbolicTree& symb_;
      FactorAllocator factor_alloc_;
      PoolAllocator pool_alloc_;
      std::vector<spldlt::NumericNode<T,PoolAllocator>> nodes_;
   };

} /* end of namespace spldlt */
