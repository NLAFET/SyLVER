/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Jonathan Hogg
/// @author Florent Lopez
#pragma once

// SyLVER
#include "kernels/common.hxx"
#include "kernels/ldlt_app.hxx"
#include "NumericFront.hxx"
#include "Tile.hxx"
// STD
#include <assert.h>
#include <string>
// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
// #include "ssids/cpu/kernels/assemble.hxx"
#include "ssids/contrib.h"

namespace sylver {
namespace spldlt {

#if defined(SPLDLT_USE_STARPU)

namespace starpu {

   extern starpu_data_handle_t workspace_hdl;      

   // Register handles for a node in StarPU
   template <typename T, typename PoolAlloc>
   void register_node(
         NumericFront<T, PoolAlloc> &front) {
         
      sylver::SymbolicFront& sfront = front.symb();
      int blksz = front.blksz();

      int const m = front.nrow();
      int const n = front.ncol();
      T *a = front.lcol;
      int const lda = front.ldl();
      int const nr = front.nr(); // number of block rows
      int const nc = front.nc(); // number of block columns
      // sfront.handles.reserve(nr*nc);
      sfront.handles.resize(nr*nc); // Allocate handles

      for(int j = 0; j < nc; ++j) {
         int blkn = std::min(blksz, n - j*blksz);

         for(int i = j; i < nr; ++i) {
            int blkm = std::min(blksz, m - i*blksz);

            // TODO remove the following register
            starpu_matrix_data_register(
                  &(sfront.handles[i + j*nr]), // StarPU handle ptr 
                  STARPU_MAIN_RAM, // memory 
                  reinterpret_cast<uintptr_t>(&a[(j*blksz)*lda+(i*blksz)]),
                  lda, blkm, blkn,
                  sizeof(T));

            // Register StarPU handle for block (i,j)
            front.blocks[j*nr+i].register_handle(); 

         }
      }

      int const ldcontrib = m-n;         
      // Allocate and init handles in contribution blocks         
      if (ldcontrib>0) {
         // Index of first block in contrib
         int rsa = n/blksz;
         // Number of block in contrib
         // int ncontrib = nr-rsa;

         for(int j = rsa; j < nr; j++) {
            for(int i = j; i < nr; i++) {
               // Register block in StarPU
               front.contrib_block(i, j).register_handle();
               // front.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].register_handle();
            }
         }
      }
   }

   ////////////////////////////////////////////////////////////////////////////////   
   // register_node_indef

   /// @brief Register handles for a node in StarPU.
   template <typename T, typename PoolAlloc>
   void register_node_indef(NumericFront<T, PoolAlloc>& front) {

      // Note: blocks are already registered when allocated
         
      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> IntAlloc;

      sylver::SymbolicFront& sfront = front.symb();
      int blksz = front.blksz();
      int m = front.nrow();
      int n = front.ncol();
      T *a = front.lcol;
      int lda = spral::ssids::cpu::align_lda<T>(m);
      int nr = front.nr(); // number of block rows
      int nc = front.nc(); // number of block columns
      spldlt::ldlt_app_internal::ColumnData<T, IntAlloc>& cdata = *front.cdata;

      // Block diagonal matrix 
      T *d = &a[n*lda];

      // sfront.handles.reserve(nr*nc);
      sfront.handles.resize(nr*nc); // allocate handles
      // printf("[register_front] sfront.handles size = %d\n", sfront.handles.size());
      for(int j = 0; j < nc; ++j) {

         int blkn = std::min(blksz, n - j*blksz);

         // Register cdata for APP factorization.
         // FIXME: Only if pivot_method is APP
         cdata[j].register_handle(); // Symbolic handle on column j
         cdata[j].register_d_hdl(d, 2*std::min((j+1)*blksz, n)); // Handle on diagonal D 
         // cdata[j].register_d_hdl(d, 2*n); // Handle on diagonal D 

         for(int i = j; i < nr; ++i) {
            int blkm = std::min(blksz, m - i*blksz);

            // TODO remove sfront.handles registration for indef case
            starpu_matrix_data_register(
                  &(sfront.handles[i + j*nr]), // StarPU handle ptr 
                  STARPU_MAIN_RAM, // memory 
                  reinterpret_cast<uintptr_t>(&a[(j*blksz)*lda+(i*blksz)]),
                  lda, blkm, blkn,
                  sizeof(T));

            // Register StarPU handle for block (i,j)
            front.blocks[j*nr+i].register_handle(); 
         }
      }

      int ldcontrib = m-n;
         
      // Allocate and init handles in contribution blocks         
      if (ldcontrib>0 && front.contrib_blocks.size()>0) {
         // Index of first block in contrib
         int rsa = n/blksz;
         // Number of block in contrib
         int ncontrib = nr-rsa;

         for(int j = rsa; j < nr; j++) {
            for(int i = j; i < nr; i++) {
               // Register block in StarPU
               // front.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].register_handle();
               front.contrib_block(i, j).register_handle();
            }
         }
      }

      // T *contrib = node.contrib;

      // // Allocate and init handles in contribution blocks         
      // if (contrib) {
      //    // Index of first block in contrib
      //    int rsa = n/blksz;
      //    // Number of block in contrib
      //    int ncontrib = nr-rsa;
      //    snode.contrib_handles.resize(ncontrib*ncontrib);

      //    for(int j = rsa; j < nr; j++) {
      //       // First col in contrib block
      //       int first_col = std::max(j*blksz, n);
      //       // Block width
      //       int blkn = std::min((j+1)*blksz, m) - first_col;

      //       for(int i = j; i < nr; i++) {
      //          // First col in contrib block
      //          int first_row = std::max(i*blksz, n);
      //          // Block height
      //          int blkm = std::min((i+1)*blksz, m) - first_row;

      //          // starpu_matrix_data_register(
      //          //       &(snode.contrib_handles[(i-rsa)+(j-rsa)*ncontrib]), // StarPU handle ptr
      //          //       STARPU_MAIN_RAM, // memory 
      //          //       reinterpret_cast<uintptr_t>(&contrib[(first_col-n)*ldcontrib+(first_row-n)]),
      //          //       ldcontrib, blkm, blkn, sizeof(T));
                  
      //          node.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].register_handle();
      //       }
      //    }

      // }
   }

   /// @brief Unregister StarPU data handles associated with a node
   template <typename T, typename PoolAlloc, bool async=true>
   void unregister_node_indef(
         NumericFront<T, PoolAlloc> &node
         ) {

      // printf("[unregister_node_indef] nodeidx = %d\n", node.symb.idx);
         
      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> IntAlloc;

      // Get node info
      sylver::SymbolicFront &snode = node.symb();
      int const blksz = node.blksz();
      int const m = node.nrow();
      int const n = node.ncol();
      int const nr = node.nr(); // number of block rows
      int const nc = node.nc(); // number of block columns

      assert(node.cdata); // Make sure cdata is allocated

      spldlt::ldlt_app_internal::
         ColumnData<T, IntAlloc>& cdata = *node.cdata;
         
      // Unregister block handles in the factors
      for(int j = 0; j < nc; ++j) {

         // FIXME: only if PivotMethod is APP
         cdata[j].template unregister_handle<async>();

         cdata[j].template unregister_d_hdl<async>(); // Unregister handle on diagonal D

         for(int i = j; i < nr; ++i) {
               
            if (async) starpu_data_unregister_submit(snode.handles[i + j*nr]);
            else       starpu_data_unregister(snode.handles[i + j*nr]);

            // Unregister block (i,j)
            node.blocks[j*nr+i].template unregister_handle<async>();
         }
      }

      // Unregister block handles in the contribution blocks
      int ldcontrib = m-n;

      if (ldcontrib>0) {
         // Index of first block in contrib
         int rsa = n/blksz;
         // Number of block in contrib
         int ncontrib = nr-rsa;

         for(int j = rsa; j < nr; j++) {
            for(int i = j; i < nr; i++) {

               // Register block in StarPU
               // node.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].template unregister_handle<async>();
               node.contrib_block(i, j).template unregister_handle<async>();
            }
         }

      }
   }

   /// @brief Unregister StarPU data handles associated with a node
   template <typename T, typename PoolAlloc, bool async=true>
   void unregister_node_posdef(
         NumericFront<T, PoolAlloc> &node
         ) {

      // Get node info
      sylver::SymbolicFront &snode = node.symb();
      int const blksz = node.blksz();
      int const m = node.nrow();
      int const n = node.ncol();
      int const nr = node.nr(); // number of block rows
      int const nc = node.nc(); // number of block columns

      // Unregister block handles in the factors
      for(int j = 0; j < nc; ++j) {
         for(int i = j; i < nr; ++i) {

            // TODO remove snode.handles array
            if (async) starpu_data_unregister_submit(snode.handles[i + j*nr]);
            else       starpu_data_unregister(snode.handles[i + j*nr]);

            // Unregister block (i,j)
            node.blocks[j*nr+i].template unregister_handle<async>();
         }
      }
         
      // Unregister block handles in the contribution blocks
      int ldcontrib = m-n;

      if (ldcontrib>0) {
         // Index of first block in contrib
         int rsa = n/blksz;
         // Number of block in contrib
         int ncontrib = nr-rsa;

         for(int j = rsa; j < nr; j++) {
            for(int i = j; i < nr; i++) {

               // Register block in StarPU
               // node.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].template unregister_handle<async>();
               node.contrib_block(i, j).template unregister_handle<async>();
            }
         }

      }         
   }      

} // End of namespace sylver::spldlt::starpu

#endif

   ////////////////////////////////////////////////////////////

   // @brief Retrieve contrib data from subtree for assembly
   // operations
   // Note: See assemble.cxx for specialized routine
   template <typename T>
   void contrib_get_data(
         const void *const contrib, int *const n,
         const T* *const val, int *const ldval, const int* *const rlist,
         int *const ndelay, const int* *const delay_perm,
         const T* *const delay_val, int *const lddelay);
   
   ////////////////////////////////////////////////////////////

   // Activate frontal matrix: allocate data structures
   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void activate_front(
         bool posdef,
         NumericFront<T, PoolAlloc> &front,
         void** child_contrib,
         FactorAlloc& factor_alloc) {

      // Allocate frontal matrix
      if (posdef) alloc_front_posdef(front, factor_alloc);
      else        alloc_front_indef(front, child_contrib, factor_alloc);

#if defined(SPLDLT_USE_STARPU)
      // Register symbolic handle for current node in StarPU
      // starpu_void_data_register(&(sfront.hdl));
      // Register block handles
      // register_node(sfront, front, blksz);

      if (posdef) spldlt::starpu::register_node(front);
      else        spldlt::starpu::register_node_indef(front);
#endif
   }

   ////////////////////////////////////////////////////////////////////////////////   
   // Allocate memory assocaited with the frontal matrix front
   template<typename T, 
            typename FactorAlloc, 
            typename PoolAlloc>
   void alloc_front_posdef(
         NumericFront<T,PoolAlloc>& front,
         FactorAlloc& factor_alloc) {

      std::string const context = "alloc_front_posdef";
      
      T *scaling = NULL; // No scaling

      // Rebind allocators
      typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<T> FATypeTraits;
      typename FATypeTraits::allocator_type factor_alloc_type(factor_alloc);
      typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<int> FAIntTraits;
      typename FAIntTraits::allocator_type factor_alloc_int(factor_alloc);
      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;
      // printf("[alloc_front] posdef = %d\n", posdef);

      sylver::SymbolicFront const& sfront = front.symb();

      front.ndelay_in(0);

      int const nrow = front.nrow();
      int const ncol = front.ncol();

      /* Get space for node now we know it size using Fortran allocator + zero it*/
      // NB L is  nrow x ncol and D is 2 x ncol (but no D if posdef)
      size_t ldl = spral::ssids::cpu::align_lda<T>(nrow);
      size_t len = ldl * ncol;  // posdef
      front.lcol = FATypeTraits::allocate(factor_alloc_type, len);

#if defined(SPLDLT_USE_STARPU)
#if defined(SPLDLT_USE_GPU)
      int ret = starpu_memory_pin(front.lcol, len*sizeof(T));
      // STARPU_CHECK_RETURN_VALUE(ret, "starpu_memory_pin");
#endif
#endif
      int err;
      // Get space for contribution block + (explicitly do not zero it!)
      front.alloc_contrib_blocks();
      // Allocate cdata (required for allocating blocks)
      // FIXME not needed for posdef case
      front.alloc_cdata();
      // Allocate frontal matrix blocks
      err = front.alloc_blocks(); // FIXME specialize for posdef case
      sylver::sylver_check_error(err, context, "Failed to allocate blocks");
   }

   ////////////////////////////////////////////////////////////////////////////////
   // alloc_front_indef

   template<typename T, 
            typename FactorAlloc, 
            typename PoolAlloc>
   void alloc_front_indef(
         NumericFront<T,PoolAlloc>& front,
         void** child_contrib,
         FactorAlloc& factor_alloc) {

      // TODO add scaling
      T *scaling = NULL; // No scaling

      // Rebind allocators
      typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<T> FATypeTraits;
      typename FATypeTraits::allocator_type factor_alloc_type(factor_alloc);
      typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<int> FAIntTraits;
      typename FAIntTraits::allocator_type factor_alloc_int(factor_alloc);
      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;
      // printf("[alloc_front] posdef = %d\n", posdef);

      sylver::SymbolicFront const& sfront = front.symb();

      front.ndelay_out(0);
      front.ndelay_in(0);
      // Count incoming delays and determine size of node
      for(auto* child=front.first_child; child!=NULL; child=child->next_child) {
         // Make sure we're not in a subtree
         if (child->symb().exec_loc == -1) {
            front.ndelay_in_add(child->ndelay_out());
         } 
         else {
            int cn, ldcontrib, ndelay, lddelay;
            T const *cval, *delay_val;
            int const *crlist, *delay_perm;
            // spral_ssids_contrib_get_data(
            contrib_get_data(
                  child_contrib[child->symb().contrib_idx], &cn, &cval,
                  &ldcontrib, &crlist, &ndelay, &delay_perm, &delay_val,
                  &lddelay);
            // front.ndelay_in += ndelay;
            front.ndelay_in_add(ndelay);
         }
      }

      int nrow = front.nrow();
      int ncol = front.ncol();

      // Get space for node now we know it size using factor
      // allocator
      // NB L is  nrow x ncol and D is 2 x ncol
      size_t ldl = spral::ssids::cpu::align_lda<T>(nrow);
      size_t len =  (ldl+2) * ncol; // indef (includes D)

      front.lcol = FATypeTraits::allocate(factor_alloc_type, len);
      assert(front.lcol != nullptr);

#if defined(SPLDLT_USE_STARPU)
#if defined(SPLDLT_USE_GPU)
      int ret = starpu_memory_pin(front.lcol, len*sizeof(T));
      // STARPU_CHECK_RETURN_VALUE(ret, "starpu_memory_pin");
#endif
#endif
      
      // Get space for contribution block + (explicitly do not zero it!)
      front.alloc_contrib_blocks();

      // Alloc + set perm for expected eliminations at this node
      // (delays are set when they are imported from children)
      front.perm = FAIntTraits::allocate(factor_alloc_int, ncol); // ncol fully summed variables
      for(int i=0; i<sfront.ncol; i++)
         front.perm[i] = sfront.rlist[i];

      // TODO: Backup is needed only when pivot_method is set to APTP

      // Allocate backups
      front.alloc_backup();      
      // Allocate cdata
      front.alloc_cdata();
      // Allocate frontal matrix blocks
      front.alloc_blocks();

   }

   // Taken from SSIDS for debugging purpose
   /**
    * \brief Add \f$A\f$ to a given block column of a node.
    *
    * \param from First column of target block column.
    * \param to One more than last column of target block column.
    * \param node Supernode to add to.
    * \param aval Values of \f$A\f$.
    * \param ldl Leading dimension of node.
    * \param scaling Scaling to apply (none if null).
    */
   template <typename T, typename NumericNode>
   void init_a_block(int from, int to, NumericNode& node, T const* aval,
                    T const* scaling) {
      SymbolicNode const& snode = node.symb();
      size_t ldl = node.ldl();
      if(scaling) {
         /* Scaling to apply */
         for(int i=from; i<to; ++i) {
            long src  = snode.amap[2*i+0] - 1; // amap contains 1-based values
            long dest = snode.amap[2*i+1] - 1; // amap contains 1-based values
            int c = dest / snode.nrow;
            int r = dest % snode.nrow;
            long k = c*ldl + r;
            if(r >= snode.ncol) k += node.ndelay_in();
            T rscale = scaling[ snode.rlist[r]-1 ];
            T cscale = scaling[ snode.rlist[c]-1 ];
            node.lcol[k] = rscale * aval[src] * cscale;
         }
      } else {
         /* No scaling to apply */
         for(int i=from; i<to; ++i) {
            long src  = snode.amap[2*i+0] - 1; // amap contains 1-based values
            long dest = snode.amap[2*i+1] - 1; // amap contains 1-based values
            int c = dest / snode.nrow;
            int r = dest % snode.nrow;
            assert(c < node.ncol());
            assert(r < node.nrow());
            long k = c*ldl + r;

            if(r >= snode.ncol) k += node.ndelay_in();

            node.lcol[k] = aval[src];
         }
      }
   }


   ////////////////////////////////////////////////////////////////////////////////
   template <typename T,
             typename PoolAlloc>
   void init_node(
         NumericFront<T,PoolAlloc>& front,
         T const* aval, T const* scaling) {

      sylver::SymbolicFront const& sfront = front.symb();

      // printf("[init_node] node idx = %d, num_a = %d\n", sfront.idx+1, sfront.num_a);
      /* Add A */
      // add_a_block<T, NumericNode<T,PoolAlloc>>(0, snode.num_a, node, aval, NULL);  
      // spral::ssids::cpu::add_a_block(0, sfront.num_a, front, aval, scaling);
      spldlt::init_a_block(0, sfront.num_a, front, aval, scaling);

   }
   
   // Terminate node.
   // Deallocate contribution block
   template <typename T, typename PoolAlloc>
   void fini_node(NumericFront<T,PoolAlloc>& node) {

      // Cleanup memory
      node.free_contrib_blocks();
      node.free_cdata();
      node.free_backup();
   }

   ////////////////////////////////////////////////////////////////////////////////   
   // Assemble block

   template <typename T, typename PoolAlloc>
   void assemble_block(NumericFront<T,PoolAlloc>& node, 
                       NumericFront<T,PoolAlloc> const& cnode, 
                       int ii, int jj, int const* cmap) {
      
      sylver::SymbolicFront const& csnode = cnode.symb();
      int const blksz = cnode.blksz();
      
      // Source node info
      int const cnrow = cnode.nrow(); // Number of rows (including delays) 
      int const cncol = cnode.ncol(); // Number of cols (including delays)

      // Source block
      auto const& blk = cnode.contrib_block(ii, jj);
      int const blk_lda = blk.lda;
      int const blk_m = blk.m;
      int const blk_n = blk.n;

      // index of first column in CB
      int col_sa = (cncol > jj*blksz) ? 0 : (jj*blksz-cncol);
      // Global index of last column
      // int c_en = std::min((jj+1)*blksz-csnode.ncol, cm); 
      // int r_sa = (csnode.ncol > ii*blksz) ? 0 : (ii*blksz-csnode.ncol); // first col in block
      // Global index of last row
      // int r_en = std::min((ii+1)*blksz-csnode.ncol, cm);
      
      // Index of first row in CB
      int row_sa = (cncol > ii*blksz) ? 0 : (ii*blksz-cncol);

      // printf("[assemble_block] row_sa: %d, col_sa: %d\n", row_sa, col_sa);

      for (int j = 0; j < blk_n; j++) {

         int c = cmap[ col_sa + j ]; // Destination column in parent front

         T *src = &(blk.a[j*blk_lda]); // Source column
                  
         // Enusre the destination column is in the fully-summed elements
         if (c < node.symb().ncol) {
            
            // printf("[assemble_block] c: %d\n", c);

            int ldd = node.ldl();
            T *dest = &node.lcol[c*ldd];
            
            int i_sa = ( ii == jj ) ? j : 0;

            for (int i = i_sa ; i < blk_m; i++) {

               // printf("[assemble_block] i: %d, j: %d\n", i, j);

               dest[ cmap[ row_sa + i ] ] += src[i];
            }
         }
      }

   }   
   
   // // Assemble block fully-summed coefficient    
   // // ii block row index
   // // jj block col index
   // template <typename T, typename PoolAlloc>
   // void assemble_block(NumericFront<T,PoolAlloc>& node, 
   //                     NumericFront<T,PoolAlloc>& cnode, 
   //                     int ii, int jj, int *cmap, int blksz) {
      
   //    SymbolicFront const& csnode = cnode.symb;
      
   //    int cm = csnode.nrow - csnode.ncol;

   //    // colum indexes
   //    int c_sa = (csnode.ncol > jj*blksz) ? 0 : (jj*blksz-csnode.ncol); // first col in block
   //    int c_en = std::min((jj+1)*blksz-csnode.ncol, cm); // last col in block
   //    // row indexes
   //    // int r_sa = (csnode.ncol > ii*blksz) ? 0 : (ii*blksz-csnode.ncol); // first col in block
   //    int r_en = std::min((ii+1)*blksz-csnode.ncol, cm); // last row in block

   //    // loop over column in block
   //    for (int j=c_sa; j<c_en; j++) {

   //       // int c = map[ csnode.rlist[csnode.ncol+j] ];
   //       int c = cmap[ j ];
   //       T *src = &(cnode.contrib[j*cm]);
   //       // T *src = &(src_blk.a[(j-c_sa)*ld_src_blk]);
                        
   //       int ncol = node.symb.ncol; // no delays!

   //       // printf("[factor_mf] c: %d, ncol: %d\n", c, ncol);

   //       if (c < ncol) {
   //          int ldd = node.get_ldl();
   //          T *dest = &node.lcol[c*ldd];

   //          // int const* idx = &cache[j];                           
   //          // loop over rows in block

   //          int r_sa = (ii==jj) ? j : (ii*blksz-csnode.ncol); // first row in block

   //          for (int i=r_sa; i<r_en; i++) {

   //             // int ii = map[ csnode.rlist[csnode.ncol+col+row] ];
   //             // dest[ idx[i] ] += src[i];

   //             dest[ map[ csnode.rlist[csnode.ncol+i] ] ] += src[i];               
   //             // dest[ cmap[i] ] += src[i-r_sa];
   //          }
   //       }
   //    }
   // }

   ////////////////////////////////////////////////////////////////////////////////   
   // Assemble contrib block
   
   template <typename T, typename PoolAlloc>
   void assemble_contrib_block(
         NumericFront<T,PoolAlloc>& node, 
         NumericFront<T,PoolAlloc> const& cnode, 
         int ii, int jj, int const* cmap) {

      // printf("[assemble_contrib_block]\n");

      int blksz = node.blksz();

      sylver::SymbolicFront const& csnode = cnode.symb();
      
      int const ncol = node.ncol();
      int const nrow = node.nrow();

      // Source block
      int const cncol = cnode.ncol();
      int const cnrow = cnode.nrow();

      auto const& src_blk = cnode.contrib_block(ii, jj);
      int src_blk_lda = src_blk.lda;
      int src_blk_m = src_blk.m;
      int src_blk_n = src_blk.n;
      
      // Colum indexes
      int col_sa = (cncol > jj*blksz) ? 0 : (jj*blksz-cncol); // first col in block
      // int c_en = std::min((jj+1)*blksz-csnode.ncol, cm); // last col in block
      // row indexes
      // int r_sa = (csnode.ncol > ii*blksz) ? 0 : (ii*blksz-csnode.ncol); // first col in block
      // int r_en = std::min((ii+1)*blksz-csnode.ncol, cm); // last col in block

      // First row index in CB of source block
      int row_sa = (cncol > ii*blksz) ? 0 : (ii*blksz-cncol);

      // loop over columns in block jj
      for (int j = 0; j < src_blk_n; j++) {

         // int c = map[ csnode.rlist[csnode.ncol+j] ];
         int c = cmap[ col_sa + j ]; // Destination column in parent front

         // Make sure column is whitin the nodes dimensions
         assert((c >= 0) && (c < node.nrow()));
         // Make sure column is not in the delays
         assert((c < node.symb().ncol) || (c >= node.ncol()));

         int cc = c / blksz;
         int dest_col_sa = (ncol > cc*blksz) ? 0 : (cc*blksz-ncol); // first col in block

         // j-th column in source block 
         T *src = &(src_blk.a[j*src_blk_lda]);

         // Enusre the destination column is in the contribution blocks
         if (c >= node.ncol()) {

            // int ldd = node.symb.nrow - node.symb.ncol;
            // T *dest = &node.contrib[(c-ncol)*ldd];

            // int const* idx = &cache[j];               
            // loop over rows in block

            // int r_sa = (csnode.ncol > ii*blksz) ? 0 : (ii*blksz-csnode.ncol); // first col in block
            
            // In the case where the source block is on the diagonal,
            // we only assemble the cofficients below the diagonal
            int i_sa = ( ii == jj ) ? j : 0;

            for (int i = i_sa; i < src_blk_m; i++) {

               int r = cmap[ row_sa + i ]; // Destination row in parent front
               int rr = r / blksz;
               // First row index in CB of destination block
               int dest_row_sa = (ncol > rr*blksz) ? 0 : (rr*blksz-ncol);
               // sylver::Tile<T, PoolAlloc> &dest_blk = node.contrib_blocks[(rr-sa)+(cc-sa)*ncontrib];
               auto& dest_blk = node.contrib_block(rr, cc);
               int dest_blk_lda = dest_blk.lda;
               assert((c - ncol - dest_col_sa) >= 0);
               T *dest = &dest_blk.a[ (c - ncol - dest_col_sa)*dest_blk_lda ];
               assert((r - ncol - dest_row_sa) >= 0);   
               dest[ r - ncol - dest_row_sa ] += src[i];
            }
         }
      }      
   }

   ////////////////////////////////////////////////////////////////////////////////   
   // Assemble contrib block with block-column memory layout

   template <typename T, typename PoolAlloc>
   void assemble_contrib_block_1d(
         NumericFront<T,PoolAlloc>& node, 
         NumericFront<T,PoolAlloc>& cnode, 
         int ii, int jj, int const* cmap
         // spral::ssids::cpu::Workspace &work
         ) {
      
      // printf("[assemble_contrib_block_1d]\n");

      int const blksz = node.blksz();
      
      // Destination node
      int const ncol = node.ncol();

      // Source node
      int const cncol = cnode.ncol();
      int const cnrow = cnode.nrow();

      sylver::Tile<T, PoolAlloc>& src_blk = cnode.contrib_block(ii, jj);
      // Get source block info
      int src_blk_lda = src_blk.lda;
      int src_blk_m = src_blk.m;
      int src_blk_n = src_blk.n;

      // Index of first col in block-column
      int col_sa = std::max(0, jj*blksz-cncol);

      // Index of first col in block-column
      int row_sa = std::max(0, ii*blksz-cncol);

      // loop over columns in block jj
      for (int j=0; j<src_blk_n; j++) {

         // j-th column in source block 
         T *src = &(src_blk.a[j*src_blk_lda]);
         
         // Destination column in parent front
         int c = cmap[ col_sa + j ];
         
         // Enusre the destination column is in the contribution blocks
         if (c >= node.ncol()) {

            // Block-column index
            int cc = c / blksz;
            // First col in block-column cc
            int dest_col_sa = std::max(0, cc*blksz-ncol);

            // Get diag block
            int diag_row_sa =  std::max(0, cc*blksz-ncol);
            auto& diag_blk = node.contrib_block(cc, cc);
            int diag_blk_lda = diag_blk.lda;

            assert((c - ncol - dest_col_sa) >= 0);

            T *dest = &diag_blk.a[ (c - ncol - dest_col_sa)*diag_blk_lda ];

            // In the case where the source block is on the diagonal,
            // we only assemble the cofficients below the diagonal
            int i_sa = ( ii==jj ) ? j : 0;

            // int* cache = work.get_ptr<int>(src_blk_m-i_sa);
            // for (int i=0; i<src_blk_m-i_sa; i++)
            //    cache[i] = cmap[ row_sa + i_sa + i ] - ncol - diag_row_sa;
               
            // spral::ssids::cpu::asm_col(src_blk_m-i_sa, cache, &src[i_sa], dest);
            
            // for (int i=0; i<src_blk_m-i_sa; i++) {
            //    dest[ cache[i] ] += src[i];
            // }
            
            for (int i=i_sa; i<src_blk_m; i++) {
               // Destination row in parent front
               // int r = cmap[ row_sa + i ];

               // assert(r-ncol-diag_row_sa >= 0);
               
               // dest[ r - ncol - diag_row_sa ] += src[i];

               assert((cmap[ row_sa + i ]-ncol-diag_row_sa) >= 0);

               dest[ cmap[ row_sa + i ] - ncol - diag_row_sa ] += src[i];
                              
            }

         }
      }   
   }
   
   // template <typename T, typename PoolAlloc>
   // void assemble_contrib_block(spldlt::NumericNode<T,PoolAlloc>& node, 
   //                             spldlt::NumericNode<T,PoolAlloc> const& cnode, 
   //                             int ii, int jj, int *cmap, int blksz) {


   //    SymbolicNode const& csnode = cnode.symb;
      
   //    int cm = csnode.nrow - csnode.ncol;
   //    int ncol = node.symb.ncol; // no delays!

   //    // colum indexes
   //    int c_sa = (csnode.ncol > jj*blksz) ? 0 : (jj*blksz-csnode.ncol); // first col in block
   //    int c_en = std::min((jj+1)*blksz-csnode.ncol, cm); // last col in block
   //    // row indexes
   //    // int r_sa = (csnode.ncol > ii*blksz) ? 0 : (ii*blksz-csnode.ncol); // first col in block
   //    int r_en = std::min((ii+1)*blksz-csnode.ncol, cm); // last col in block

   //    // loop over columns in block jj
   //    for (int j=c_sa; j<c_en; j++) {

   //       // int c = map[ csnode.rlist[csnode.ncol+j] ];
   //       int c = cmap[j] ;
   //       T *src = &(cnode.contrib[j*cm]);
                                 
   //       if (c >= ncol) {
   //          int ldd = node.symb.nrow - node.symb.ncol;
   //          T *dest = &node.contrib[(c-ncol)*ldd];

   //          // int const* idx = &cache[j];               
   //          // loop over rows in block

   //          // int r_sa = (csnode.ncol > ii*blksz) ? 0 : (ii*blksz-csnode.ncol); // first col in block
   //          int r_sa = (ii == jj) ? j : (ii*blksz-csnode.ncol); // first col in block

   //          for (int i=r_sa; i<r_en; i++) {

   //             // printf("[factor_mf] i: %d\n", i);

   //             // int ii = map[ csnode.rlist[csnode.ncol+col+row] ];
   //             // dest[ idx[i] ] += src[i];

   //             // dest[ map[ csnode.rlist[csnode.ncol+i] ] - ncol ] += src[i];
   //             dest[ cmap[ i ] - ncol ] += src[i];
   //          }
   //       }
   //    }

   // }

   ////////////////////////////////////////////////////////////////////////////////   
   // Assemble subtree

   // template <typename T, typename PoolAlloc>
   // void assemble_subtree (
   //       NumericFront<T,PoolAlloc>& node,
   //       SymbolicFront const& csnode,
   //       void** child_contrib, 
   //       int contrib_idx // Index of subtree to assemble
   //       ) {

   //    SymbolicFront const& snode = node.symb;

   //    // Retreive contribution block from subtrees
   //    int cn, ldcontrib, ndelay, lddelay;
   //    double const *cval, *delay_val;
   //    int const *crlist, *delay_perm;
   //    spral_ssids_contrib_get_data(
   //          child_contrib[contrib_idx], &cn, &cval, &ldcontrib, &crlist,
   //          &ndelay, &delay_perm, &delay_val, &lddelay
   //          );

   //    /* Handle delays - go to back of node
   //     * (i.e. become the last rows as in lower triangular format) */

   //    if(!cval) return; // child was all delays, nothing more to do

   //    for(int j = 0; j < cn; ++j) {
               
   //       int c = csnode.map[ j ]; // Destination column
                  
   //       T const* src = &cval[j*ldcontrib];

   //       if (c < snode.ncol) {

   //          int ldd = node.get_ldl();
   //          T *dest = &node.lcol[c*ldd];

   //          for (int i = j ; i < cn; ++i) {
   //             // Assemble destination block
   //             dest[ csnode.map[ i ]] += src[i];
   //          }
   //       }
   //    }

   // }

   ////////////////////////////////////////////////////////////
   // assemble_contrib_subtree_block
   template <typename T, typename PoolAlloc>
   void assemble_contrib_subtree_block(
         NumericFront<T,PoolAlloc>& node,
         sylver::SymbolicFront const& csnode,
         void** child_contrib,
         int contrib_idx, // Index of subtree to assemble
         int ii, // Block-row index
         int jj // Block-column index
         ) {

      sylver::SymbolicFront const& snode = node.symb();
      int const blksz = node.blksz();

      /* Initialise variables */
      int const ncol = node.ncol();
      int const nrow = node.nrow();

      // Retreive contribution block from subtrees
      int cn, ldcontrib, ndelay, lddelay;
      T const *cval, *delay_val;
      int const *crlist, *delay_perm;
      // spral_ssids_contrib_get_data(
      contrib_get_data(
            child_contrib[contrib_idx], &cn, &cval, &ldcontrib, &crlist,
            &ndelay, &delay_perm, &delay_val, &lddelay
            );
      // printf("[assemble_contrib_subtree] ndelay = %d, cval = %p\n", ndelay, cval);
      if(!cval) return; // child was all delays, nothing more to do

      // Loop over columns of block-column `jj` 
      for(int j = jj*blksz;
          j < std::min((jj+1)*blksz,cn); ++j) {               

         int c = csnode.map[ j ]; // Destination column

         T const* src = &cval[j*ldcontrib];

         // Make sure that the destination column is in the
         // contribution of parent node
         if (c >= snode.ncol) {

            int cc = c / blksz; // Destination block column
            int dest_col_sa = (ncol > cc*blksz) ? 0 : (cc*blksz-ncol); // First col in block

            // Loop over rows of block-row `ii`
            for (int i = std::max(ii*blksz, j) ;
                 i < std::min((ii+1)*blksz,cn); ++i) {

               // Destination row in parent front
               int r = csnode.map[ i ];
               int rr = r / blksz; // Destination block row
               int dest_row_sa = (ncol > rr*blksz) ? 0 : (rr*blksz-ncol);

               // Destination block in contributions of parent node
               auto& dest_blk = node.contrib_block(rr, cc);

               int dest_blk_lda = dest_blk.lda;
               assert( (c - ncol - dest_col_sa) >= 0 );
               T *dest = &dest_blk.a[ (c - ncol - dest_col_sa)*dest_blk_lda ];
               // Assemble destination block
               assert( (r - ncol - dest_row_sa) >= 0 );
               dest[ r - ncol - dest_row_sa ] += src[i];
               
            }
         }         
      }
   }
   
   ////////////////////////////////////////////////////////////////////////////////   
   // Assemble contrib subtree
   template <typename T, typename PoolAlloc>
   void assemble_contrib_subtree(
         NumericFront<T,PoolAlloc>& node,
         sylver::SymbolicFront const& csnode,
         void** child_contrib, 
         int contrib_idx// Index of subtree to assemble
         ) {

      int const blksz = node.blksz();

      /* Initialise variables */
      int const ncol = node.ncol();

      // Retreive contribution block from subtrees
      int cn, ldcontrib, ndelay, lddelay;
      T const *cval, *delay_val;
      int const *crlist, *delay_perm;
      // spral_ssids_contrib_get_data(
      contrib_get_data(
            child_contrib[contrib_idx], &cn, &cval, &ldcontrib, &crlist,
            &ndelay, &delay_perm, &delay_val, &lddelay
            );
      // printf("[assemble_contrib_subtree] ndelay = %d, cval = %p\n", ndelay, cval);
      if(!cval) return; // child was all delays, nothing more to do

      for(int j = 0; j < cn; ++j) {

         int c = csnode.map[ j ]; // Destination column

         // Make sure column is whitin the nodes dimensions
         assert((c >= 0) && (c < node.nrow()));
         // Make sure column is not in the delays
         assert((c < node.symb().ncol) || (c >= node.ncol()));

         T const* src = &cval[j*ldcontrib];

         if (c >= node.symb().ncol) {

            int cc = c / blksz; // Destination block column
            int dest_col_sa = (ncol > cc*blksz) ? 0 : (cc*blksz-ncol); // First col in block

            for (int i = j; i < cn; ++i) {
               int r = csnode.map[ i ]; // Destination row in parent front
               int rr = r / blksz; // Destination block row
               // First row index in CB of destination block
               int dest_row_sa = (ncol > rr*blksz) ? 0 : (rr*blksz-ncol);

               sylver::Tile<T, PoolAlloc> &dest_blk = node.contrib_block(rr, cc);
               int dest_blk_lda = dest_blk.lda;
               assert( (c - ncol - dest_col_sa) >= 0 );
               T *dest = &dest_blk.a[ (c - ncol - dest_col_sa)*dest_blk_lda ];
               // Assemble destination block
               assert( (r - ncol - dest_row_sa) >= 0 );
               dest[ r - ncol - dest_row_sa ] += src[i];
            }
         }
      }
      /* Free memory from child contribution block */
      // TODO
      // spral_ssids_contrib_free_dbl(child_contrib[contrib_idx]);
   }

   ////////////////////////////////////////////////////////////
   // assemble_delays_subtree

   /// @brief Assemble delays from a subtree to its parent
   template <typename T, typename PoolAlloc>
   void assemble_delays_subtree(
         NumericFront<T,PoolAlloc>& node,
         sylver::SymbolicFront const& csnode,
         void** child_contrib,
         int contrib_idx, // Index of subtree to assemble
         int delay_col
         ) {
      
      int ncol = node.ncol();
      size_t ldl = node.ldl(); // Leading dimension

      // Retreive contribution block from subtrees
      int cn, ldcontrib, ndelay, lddelay;
      T const *cval, *delay_val;
      int const *crlist, *delay_perm;
      // spral_ssids_contrib_get_data(
      contrib_get_data(
            child_contrib[csnode.contrib_idx], &cn, &cval, &ldcontrib, &crlist,
            &ndelay, &delay_perm, &delay_val, &lddelay
            );

      /* Handle delays - go to back of node
       * (i.e. become the last rows as in lower triangular format) */
      for(int i=0; i<ndelay; i++) {
         // Add delayed rows (from delayed cols)
         T *dest = &node.lcol[delay_col*(ldl+1)];
         T const* src = &delay_val[i*(lddelay+1)];
         node.perm[delay_col] = delay_perm[i];
         for(int j=0; j<ndelay-i; j++) {
            dest[j] = src[j];
         }
         // Add child's non-fully summed rows (from delayed cols)
         dest = node.lcol;
         src = &delay_val[i*lddelay+ndelay];
         for(int j=0; j<cn; j++) {
            // int r = cache[j];
            int r = csnode.map[j];
            if(r < ncol) dest[r*ldl+delay_col] = src[j];
            else         dest[delay_col*ldl+r] = src[j];
         }
         delay_col++;
      }

   }

   ////////////////////////////////////////////////////////////
   // assemble_subtree_block
   
   /// @brief Assemble block (i,j) from subtree `csnode` to
   /// fully-summed coeficients of parent node `node`
   ///
   /// @param i Block-row index of block to be assembled
   /// @param i Block-column index of block to be assembled
   template <typename T, typename PoolAlloc>
   void assemble_subtree_block(
         NumericFront<T,PoolAlloc>& node,
         sylver::SymbolicFront const& csnode,
         void** child_contrib, 
         int contrib_idx,// Index of subtree to assemble
         int ii, // Block-row index
         int jj // Block-column index
         ) {

      int const blksz = node.blksz();

      // Retreive contribution block from subtrees
      int cn, ldcontrib, ndelay, lddelay;
      T const *cval, *delay_val;
      int const *crlist, *delay_perm;
      // spral_ssids_contrib_get_data(
      contrib_get_data(
            child_contrib[csnode.contrib_idx],
            &cn, &cval, &ldcontrib, &crlist,
            &ndelay, &delay_perm, &delay_val,
            &lddelay);

      if(!cval) return; // child was all delays, nothing more to do
      /* Handle expected contribution */
      // for(int j = 0; j < cn; ++j) {
      // Loop over columns of block-column `jj` 
      for(int j = jj*blksz;
          j < std::min((jj+1)*blksz,cn); ++j) {               
         int c = csnode.map[ j ]; // Destination column                  
         T const* src = &cval[j*ldcontrib];
         // Make sure we stay in fully-summed coefficients
         if (c < node.symb().ncol) {
            int ldd = node.ldl();
            T *dest = &node.lcol[c*ldd];

            // for (int i = j ; i < cn; ++i) {
            // Loop over rows of block-row `ii`
            for (int i = std::max(ii*blksz, j) ;
                 i < std::min((ii+1)*blksz,cn); ++i) {
               // Assemble destination block
               dest[ csnode.map[ i ]] += src[i];
            }
         }
      }
   }
   
   ////////////////////////////////////////////////////////////
   // assemble_subtree
   
   /// @brief Assemble a subtree to its parent
   template <typename T, typename PoolAlloc>
   void assemble_subtree(
         NumericFront<T,PoolAlloc>& node,
         sylver::SymbolicFront const& csnode,
         void** child_contrib, 
         int contrib_idx// Index of subtree to assemble
         ) {

      int const ncol = node.ncol();
      size_t const ldl = node.ldl(); // Leading dimension

      // Retreive contribution block from subtrees
      int cn, ldcontrib, ndelay, lddelay;
      T const *cval, *delay_val;
      int const *crlist, *delay_perm;
      // spral_ssids_contrib_get_data(
      contrib_get_data(
            child_contrib[csnode.contrib_idx], &cn, &cval, &ldcontrib, &crlist,
            &ndelay, &delay_perm, &delay_val, &lddelay
            );
      // int *cache = new int[cn];
      // for(int j=0; j<cn; ++j)
      //    cache[j] = map[ crlist[j] ];

      // printf("[assemble_subtree] ndelay = %d, delay_col = %d\n", ndelay, delay_col);

      /* Handle delays - go to back of node
       * (i.e. become the last rows as in lower triangular format) */
      // for(int i=0; i<ndelay; i++) {
      //    // Add delayed rows (from delayed cols)
      //    T *dest = &node.lcol[delay_col*(ldl+1)];
      //    T const* src = &delay_val[i*(lddelay+1)];
      //    node.perm[delay_col] = delay_perm[i];
      //    for(int j=0; j<ndelay-i; j++) {
      //       dest[j] = src[j];
      //    }
      //    // Add child's non-fully summed rows (from delayed cols)
      //    dest = node.lcol;
      //    src = &delay_val[i*lddelay+ndelay];
      //    for(int j=0; j<cn; j++) {
      //       // int r = cache[j];
      //       int r = csnode.map[j];
      //       if(r < ncol) dest[r*ldl+delay_col] = src[j];
      //       else         dest[delay_col*ldl+r] = src[j];
      //    }
      //    delay_col++;
      // }
      if(!cval) return; // child was all delays, nothing more to do
      /* Handle expected contribution */
      for(int j = 0; j < cn; ++j) {               
         int c = csnode.map[ j ]; // Destination column                  
         T const* src = &cval[j*ldcontrib];
         if (c < node.symb().ncol) {
            int ldd = node.ldl();
            T *dest = &node.lcol[c*ldd];

            for (int i = j ; i < cn; ++i) {
               // Assemble destination block
               dest[ csnode.map[ i ]] += src[i];
            }
         }
      }
   }

   ///////////////////////////////////////////////////////////   
   // @brief Copy delays columns from a chil node to its parent
   template <typename T, typename PoolAlloc>
   void assemble_delays(
         // std::vector<int, PoolAllocInt> map,
         NumericFront<T,PoolAlloc>& cnode,
         int delay_col,
         NumericFront<T,PoolAlloc>& node
         ) {

      // printf("[assemble_delays]\n");

      sylver::SymbolicFront &csnode = cnode.symb(); // Child symbolic node

      int ncol = node.ncol();
      size_t ldl = node.ldl();

      for(int i=0; i<cnode.ndelay_out(); i++) {
         // Add delayed rows (from delayed cols)
         T *dest = &node.lcol[delay_col*(ldl+1)];
         int lds = align_lda<T>(csnode.nrow + cnode.ndelay_in());
         T *src = &cnode.lcol[(cnode.nelim()+i)*(lds+1)];
         node.perm[delay_col] = cnode.perm[cnode.nelim()+i];
         for(int j=0; j<cnode.ndelay_out()-i; j++) {
            dest[j] = src[j];
         }
         // Add child's non-fully summed rows (from delayed cols)
         dest = node.lcol;
         src = &cnode.lcol[cnode.nelim()*lds + cnode.ndelay_in() +i*lds];
         for(int j=csnode.ncol; j<csnode.nrow; j++) {
            // int r = map[ csnode.rlist[j] ];
            int r = csnode.map[j-csnode.ncol];
            if(r < ncol) dest[r*ldl+delay_col] = src[j];
            else         dest[delay_col*ldl+r] = src[j];
         }
         delay_col++;
      }

   }

   ///////////////////////////////////////////////////////////
   /// @brief Assemble contributions from children node and subtrees
   /// into the fully-summed columns
   template <typename T, typename PoolAlloc>
   void assemble_notask(
         int n,
         NumericFront<T,PoolAlloc>& node,
         void** child_contrib,
         PoolAlloc const& pool_alloc
         ) {

      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;

      int const blksz = node.blksz();
      sylver::SymbolicFront const& snode = node.symb();

      int nrow = node.nrow();
      int ncol = node.ncol();
      size_t ldl = node.ldl();

      /*
       * Add children
       */
      int delay_col = snode.ncol;

      // printf("[assemble]\n");

      // Allocate mapping array
      // int *map = new int[n+1];
      std::vector<int, PoolAllocInt> map(n+1, PoolAllocInt(pool_alloc));

      // build lookup vector, allowing for insertion of delayed vars
      // Note that while rlist[] is 1-indexed this is fine so long as lookup
      // is also 1-indexed (which it is as it is another node's rlist[]
      for(int i=0; i<snode.ncol; i++)
         map[ snode.rlist[i] ] = i;
      for(int i=snode.ncol; i<snode.nrow; i++)
         map[ snode.rlist[i] ] = i + node.ndelay_in();
      
      // Assemble front: fully-summed columns 
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {

         sylver::SymbolicFront& csnode = child->symb(); // Children symbolic node

         int cm = csnode.nrow - csnode.ncol;
         csnode.map = new int[cm];
         for (int i=0; i<cm; i++)
            csnode.map[i] = map[ csnode.rlist[csnode.ncol+i] ];

         int ldcontrib = csnode.nrow - csnode.ncol;
         if (csnode.exec_loc == -1) {
            // Assemble contributions from child front
            assemble_delays(*child, delay_col, node);
            
            delay_col += child->ndelay_out();

            // Handle expected contributions (only if something there)
            if (ldcontrib>0) {
                  
               int cnrow = child->nrow();
               int cncol = child->ncol();
                  
               int csa = cncol / blksz;
               int cnr = child->nr(); // number of block rows
               // Loop over blocks in contribution blocks
               for (int jj = csa; jj < cnr; ++jj) {
                  for (int ii = jj; ii < cnr; ++ii) {
                     assemble_block(node, *child, ii, jj, csnode.map);
                  }
               }
            }
               
         }
         else {
            // Assemble contributions from subtree

            // Retreive contribution block from subtrees
            int cn, ldcontrib, ndelay, lddelay;
            double const *cval, *delay_val;
            int const *crlist, *delay_perm;
            // spral_ssids_contrib_get_data(
            contrib_get_data(
                  child_contrib[csnode.contrib_idx], &cn, &cval, &ldcontrib, &crlist,
                  &ndelay, &delay_perm, &delay_val, &lddelay
                  );
            // int *cache = new int[cn];
            // for(int j=0; j<cn; ++j)
            //    cache[j] = map[ crlist[j] ];

            // printf("[assemble] contrib_idx = %d, ndelay = %d\n", csnode.contrib_idx, ndelay);

            /* Handle delays - go to back of node
             * (i.e. become the last rows as in lower triangular format) */
            for(int i=0; i<ndelay; i++) {
               // Add delayed rows (from delayed cols)
               T *dest = &node.lcol[delay_col*(ldl+1)];
               T const* src = &delay_val[i*(lddelay+1)];
               node.perm[delay_col] = delay_perm[i];
               for(int j=0; j<ndelay-i; j++) {
                  dest[j] = src[j];
               }
               // Add child's non-fully summed rows (from delayed cols)
               dest = node.lcol;
               src = &delay_val[i*lddelay+ndelay];
               for(int j=0; j<cn; j++) {
                  // int r = cache[j];
                  int r = csnode.map[j];
                  if(r < ncol) dest[r*ldl+delay_col] = src[j];
                  else         dest[delay_col*ldl+r] = src[j];
               }
               delay_col++;
            }
            if(!cval) continue; // child was all delays, nothing more to do
            /* Handle expected contribution */
            for(int j = 0; j < cn; ++j) {               
               int c = csnode.map[ j ]; // Destination column                  
               T const* src = &cval[j*ldcontrib];
               if (c < snode.ncol) {
                  int ldd = node.ldl();
                  T *dest = &node.lcol[c*ldd];

                  for (int i = j ; i < cn; ++i) {
                     // Assemble destination block
                     dest[ csnode.map[ i ]] += src[i];
                  }
               }
            }

         }
      }
   } // assemble

   ////////////////////////////////////////////////////////////////////////////////   
   // assemble_contrib_notask
   //
   // Assemble contributions from children node and subtrees into the
   // contribution blocks
   template <typename T, typename PoolAlloc>
   void assemble_contrib_notask(
         NumericFront<T,PoolAlloc>& node,
         void** child_contrib) {

      // Assemble front: non fully-summed columns i.e. contribution block 
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {

         sylver::SymbolicFront const& child_sfront = child->symb();
         // SymbolicFront &child_sfront = symb_[child->symb.idx];

         int ldcontrib = child_sfront.nrow - child_sfront.ncol;
         // Handle expected contributions (only if something there)
         // if (child->contrib) {
         if (ldcontrib>0) {
            // Skip iteration if child node is in a subtree
            if (child_sfront.exec_loc != -1) {                     
               // Assemble contribution block from subtrees into non
               // fully-summed coefficients
               assemble_contrib_subtree(
                     node, child_sfront, child_contrib, 
                     child_sfront.contrib_idx);

            }
            else {                     

               int cncol = child->ncol();
               int cnrow = child->nrow();
               int blksz = child->blksz();
               
               int csa = cncol / blksz;
               // Number of block rows in child node
               int cnr = child->nr();
               // Lopp over blocks in contribution blocks
               for (int jj = csa; jj < cnr; ++jj) {                     
                  for (int ii = jj; ii < cnr; ++ii) {

#if defined(MEMLAYOUT_1D)
         assemble_contrib_block_1d(node, *child, ii, jj, child_sfront.map);
#else
         assemble_contrib_block(node, *child, ii, jj, child_sfront.map);
#endif
                  }
               }
            }
         }
      } // Loop over child nodes

   } // assemble_contrib_notask

}} // End of namespace sylver::spldlt

