/// \file
/// \copyright 2016- The Science and Technology Facilities Council (STFC)
/// \author Jonathan Hogg
/// \author Florent Lopez

#pragma once

#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/kernels/assemble.hxx"

#include <assert.h>

#include "kernels/ldlt_app.hxx"
#include "NumericFront.hxx"

namespace spldlt {

#if defined(SPLDLT_USE_STARPU)

   namespace starpu {

      extern starpu_data_handle_t workspace_hdl;      

      // Register handles for a node in StarPU
      template <typename T, typename PoolAlloc>
      void register_node(
            NumericFront<T, PoolAlloc> &front) {
         
         SymbolicFront& sfront = front.symb;
         int blksz = front.blksz;

         int m = front.get_nrow();
         int n = front.get_ncol();
         T *a = front.lcol;
         int lda = spral::ssids::cpu::align_lda<T>(m);
         int nr = (m-1) / blksz + 1; // number of block rows
         int nc = (n-1) / blksz + 1; // number of block columns
         // sfront.handles.reserve(nr*nc);
         sfront.handles.resize(nr*nc); // allocate handles
         // printf("[register_front] sfront.handles size = %d\n", sfront.handles.size());
         for(int j = 0; j < nc; ++j) {

            int blkn = std::min(blksz, n - j*blksz);

            for(int i = j; i < nr; ++i) {
               int blkm = std::min(blksz, m - i*blksz);

               starpu_matrix_data_register(
                     &(sfront.handles[i + j*nr]), // StarPU handle ptr 
                     STARPU_MAIN_RAM, // memory 
                     reinterpret_cast<uintptr_t>(&a[(j*blksz)*lda+(i*blksz)]),
                     lda, blkm, blkn,
                     sizeof(T));
               // printf("[register_front] blk idx = %d, hdl = %p\n", i + j*nr, &(sfront.handles[i + j*nr]));

            }
         }

         int ldcontrib = m-n;
         
         // Allocate and init handles in contribution blocks         
         if (ldcontrib>0) {
            // Index of first block in contrib
            int rsa = n/blksz;
            // Number of block in contrib
            int ncontrib = nr-rsa;

            for(int j = rsa; j < nr; j++) {
               for(int i = j; i < nr; i++) {
                  // Register block in StarPU
                  front.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].register_handle();
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

      ////////////////////////////////////////////////////////////////////////////////   
      // register_node_indef
      //
      /// @brief Register handles for a node in StarPU.
      template <typename T, typename PoolAlloc>
      void register_node_indef(NumericFront<T, PoolAlloc>& front) {

         // Note: blocks are already registered when allocated
         
         typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> IntAlloc;

         SymbolicFront& sfront = front.symb;
         int blksz = front.blksz;
         int m = front.get_nrow();
         int n = front.get_ncol();
         T *a = front.lcol;
         int lda = spral::ssids::cpu::align_lda<T>(m);
         int nr = front.get_nr(); // number of block rows
         int nc = front.get_nc(); // number of block columns
         spldlt::ldlt_app_internal::ColumnData<T, IntAlloc>& cdata = *front.cdata;

         // Block diagonal matrix 
         T *d = &a[n*lda];

         // // Register worksapce handle
         // starpu_matrix_data_register (
         //       &spldlt::starpu::workspace_hdl,
         //       -1, (uintptr_t) NULL,
         //       blksz, blksz, blksz,
         //       sizeof(T));

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

               // register StarPU handle for block (i, j)
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
                  front.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].register_handle();
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
         SymbolicFront &snode = node.symb;
         int blksz = node.blksz;
         int m = node.get_nrow();
         int n = node.get_ncol();
         int nr = node.get_nr(); // number of block rows
         int nc = node.get_nc(); // number of block columns

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
                  node.contrib_blocks[(i-rsa)+(j-rsa)*ncontrib].template unregister_handle<async>();
               }
            }

         }
      }

   } // namespace spldlt::starpu
#endif
   
   ////////////////////////////////////////////////////////////////////////////////
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

      T *scaling = NULL;

      /* Rebind allocators */
      typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<double> FADoubleTraits;
      typename FADoubleTraits::allocator_type factor_alloc_double(factor_alloc);
      typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<int> FAIntTraits;
      typename FAIntTraits::allocator_type factor_alloc_int(factor_alloc);
      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;
      // printf("[alloc_front] posdef = %d\n", posdef);

      SymbolicFront const& sfront = front.symb;

      front.ndelay_in = 0;

      int nrow = front.get_nrow();
      int ncol = front.get_ncol();

      /* Get space for node now we know it size using Fortran allocator + zero it*/
      // NB L is  nrow x ncol and D is 2 x ncol (but no D if posdef)
      size_t ldl = spral::ssids::cpu::align_lda<double>(nrow);
      size_t len = ldl    * ncol;  // posdef
      front.lcol = FADoubleTraits::allocate(factor_alloc_double, len);

      // /* Get space for contribution block + (explicitly do not zero it!) */
      // node.alloc_contrib();
      front.alloc_contrib_blocks();

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

      T *scaling = NULL;

      /* Rebind allocators */
      typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<double> FADoubleTraits;
      typename FADoubleTraits::allocator_type factor_alloc_double(factor_alloc);
      typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<int> FAIntTraits;
      typename FAIntTraits::allocator_type factor_alloc_int(factor_alloc);
      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;
      // printf("[alloc_front] posdef = %d\n", posdef);

      SymbolicFront const& sfront = front.symb;

      front.ndelay_out = 0;
      front.ndelay_in = 0;
      /* Count incoming delays and determine size of node */
      for(auto* child=front.first_child; child!=NULL; child=child->next_child) {
         // Make sure we're not in a subtree
         if (child->symb.exec_loc == -1) { 
            front.ndelay_in += child->ndelay_out;
         } 
         else {
            int cn, ldcontrib, ndelay, lddelay;
            double const *cval, *delay_val;
            int const *crlist, *delay_perm;
            spral_ssids_contrib_get_data(
                  child_contrib[child->symb.contrib_idx], &cn, &cval, &ldcontrib, &crlist,
                  &ndelay, &delay_perm, &delay_val, &lddelay
                  );
            front.ndelay_in += ndelay;
         }
      }

      int nrow = front.get_nrow();
      int ncol = front.get_ncol();

      /* Get space for node now we know it size using Fortran allocator + zero it*/
      // NB L is  nrow x ncol and D is 2 x ncol (but no D if posdef)
      size_t ldl = spral::ssids::cpu::align_lda<T>(nrow);
      // size_t len = posdef ?  ldl    * ncol  // posdef
      //    : (ldl+2) * ncol; // indef (includes D)
      size_t len =  (ldl+2) * ncol; // indef (includes D)

      front.lcol = FADoubleTraits::allocate(factor_alloc_double, len);
      // front.lcol = new T[len];
      // DEBUG
      // TODO remove explicit zeroing
      // for(int i=0; i<len; ++i)
      //    front.lcol[i] = 0.0;
      
      // Get space for contribution block + (explicitly do not zero it!)
      front.alloc_contrib_blocks();
      // DEBUG
      // TODO remove explicit zeroing      
      // front.zero_contrib_blocks();
      /* Alloc + set perm for expected eliminations at this node (delays are set
         * when they are imported from children) */
      front.perm = FAIntTraits::allocate(factor_alloc_int, ncol); // ncol fully summed variables
      for(int i=0; i<sfront.ncol; i++)
         front.perm[i] = sfront.rlist[i];

      // TODO: Only if pivot_method is APP
      // Allocate backups
      front.alloc_backup();      
      // Allocate cdata
      front.alloc_cdata();
      // Allocate block structure
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
      SymbolicNode const& snode = node.symb;
      size_t ldl = node.get_ldl();
      if(scaling) {
         /* Scaling to apply */
         for(int i=from; i<to; ++i) {
            long src  = snode.amap[2*i+0] - 1; // amap contains 1-based values
            long dest = snode.amap[2*i+1] - 1; // amap contains 1-based values
            int c = dest / snode.nrow;
            int r = dest % snode.nrow;
            long k = c*ldl + r;
            if(r >= snode.ncol) k += node.ndelay_in;
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
            assert(c < node.get_ncol());
            assert(r < node.get_nrow());
            long k = c*ldl + r;
            // printf("[init_a_block] k = %d, src = %d\n", k, src);
            if(r >= snode.ncol) k += node.ndelay_in;
            // if(node.ndelay_in>0) printf("[init_a_block] ndelay_in = %d\n", node.ndelay_in);
            // printf("[init_a_block] ldl = %d\n", ldl);
            node.lcol[k] = aval[src];
         }
      }
   }


   ////////////////////////////////////////////////////////////////////////////////
   template <typename T,
             typename PoolAlloc>
   void init_node(
         NumericFront<T,PoolAlloc>& front,
         T const* aval) {

      T *scaling = nullptr;
      SymbolicFront const& sfront = front.symb;

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
      
      SymbolicFront const& csnode = cnode.symb;
      int blksz = cnode.blksz;
      
      // Source node
      int cnrow = cnode.get_nrow(); // Number of rows (including delays) 
      int cncol = cnode.get_ncol(); // Number of cols (including delays)

      int cm = csnode.nrow - csnode.ncol;
      int csa = cncol / blksz; // Index of first block in contrib
      int cnr = cnode.get_nr(); // number of block rows in child node
      int cncontrib = cnr-csa;
      // Source block
      Tile<T, PoolAlloc> const& blk = cnode.contrib_blocks[(ii-csa)+(jj-csa)*cncontrib];
      int blk_lda = blk.lda;
      int blk_m = blk.m;
      int blk_n = blk.n;

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
         
         int ncol = node.symb.ncol; // no delays!
         // printf("[assemble_block] ncol: %d\n", ncol);
         
         // Enusre the destination column is in the fully-summed elements
         if (c < ncol) {
            
            // printf("[assemble_block] c: %d\n", c);

            int ldd = node.get_ldl();
            T *dest = &node.lcol[c*ldd];
            
            int i_sa = ( ii == jj ) ? j : 0;

            for (int i = i_sa ; i < blk_m; i++) {

               // printf("[assemble_block] i: %d, j: %d\n", i, j);

               dest[ cmap[ row_sa + i ] ] += src[i];
            }
         }
      }

      // // loop over column in block
      // for (int j=c_sa; j<c_en; j++) {

      //    // int c = map[ csnode.rlist[csnode.ncol+j] ];
      //    int c = cmap[ j ];
      //    // T *src = &(cnode.contrib[j*cm]);
      //    T *src = &(src_blk.a[(j-c_sa)*ld_src_blk]);
                        
      //    int ncol = node.symb.ncol; // no delays!

      //    // printf("[factor_mf] c: %d, ncol: %d\n", c, ncol);

      //    if (c < ncol) {
      //       int ldd = node.get_ldl();
      //       T *dest = &node.lcol[c*ldd];

      //       // int const* idx = &cache[j];                           
      //       // loop over rows in block

      //       int r_sa = (ii==jj) ? j : (ii*blksz-csnode.ncol); // first row in block

      //       for (int i=r_sa; i<r_en; i++) {

      //          // int ii = map[ csnode.rlist[csnode.ncol+col+row] ];
      //          // dest[ idx[i] ] += src[i];

      //          // dest[ map[ csnode.rlist[csnode.ncol+i] ] ] += src[i];               
      //          dest[ cmap[i] ] += src[i-r_sa];
      //       }
      //    }
      // }

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

      int blksz = node.blksz;

      SymbolicFront const& csnode = cnode.symb;
      
      int cm = csnode.nrow - csnode.ncol;
      int ncol = node.get_ncol();
      int nrow = node.get_nrow();

      // Source block
      int cncol = cnode.get_ncol();
      int cnrow = cnode.get_nrow();

      int csa = cncol / blksz; // Index of first block in contrib
      int cnr = cnode.get_nr(); // Number of block rows in child node
      int cncontrib = cnr-csa;
      Tile<T, PoolAlloc> const& src_blk = cnode.contrib_blocks[(ii-csa)+(jj-csa)*cncontrib];
      int src_blk_lda = src_blk.lda;
      int src_blk_m = src_blk.m;
      int src_blk_n = src_blk.n;

      // Destination block
      int sa = ncol / blksz; // Index of first block in contrib
      int nr = node.get_nr(); // Number of block rows in node
      int ncontrib = nr-sa;
      
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
         int cc = c / blksz;
         int dest_col_sa = (ncol > cc*blksz) ? 0 : (cc*blksz-ncol); // first col in block

         // j-th column in source block 
         T *src = &(src_blk.a[j*src_blk_lda]);

         // Enusre the destination column is in the contribution blocks
         if (c >= ncol) {

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
               Tile<T, PoolAlloc> &dest_blk = node.contrib_blocks[(rr-sa)+(cc-sa)*ncontrib];
               int dest_blk_lda = dest_blk.lda;
               T *dest = &dest_blk.a[ (c - ncol - dest_col_sa)*dest_blk_lda ];

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

      int blksz = node.blksz;
      
      // Destination node
      int ncol = node.get_ncol();

      // Source node
      int cncol = cnode.get_ncol();
      int cnrow = cnode.get_nrow();

      Tile<T, PoolAlloc>& src_blk = cnode.get_contrib_block(ii, jj);
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
         if (c >= ncol) {

            // Block-column index
            int cc = c / blksz;
            // First col in block-column cc
            int dest_col_sa = std::max(0, cc*blksz-ncol);

            // Get diag block
            int diag_row_sa =  std::max(0, cc*blksz-ncol);
            Tile<T, PoolAlloc>& diag_blk = node.get_contrib_block(cc, cc);
            int diag_blk_lda = diag_blk.lda;

            assert(c - ncol - dest_col_sa >= 0);

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

               assert(cmap[ row_sa + i ]-ncol-diag_row_sa >= 0);

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

   ////////////////////////////////////////////////////////////////////////////////   
   // Assemble contrib subtree
   template <typename T, typename PoolAlloc>
   void assemble_contrib_subtree(
         NumericFront<T,PoolAlloc>& node,
         SymbolicFront const& csnode,
         void** child_contrib, 
         int contrib_idx// Index of subtree to assemble
         ) {

      SymbolicFront const& snode = node.symb;
      int blksz = node.blksz;

      /* Initialise variables */
      int ncol = node.get_ncol();
      int nrow = node.get_nrow();

      // Retreive contribution block from subtrees
      int cn, ldcontrib, ndelay, lddelay;
      double const *cval, *delay_val;
      int const *crlist, *delay_perm;
      spral_ssids_contrib_get_data(
            child_contrib[contrib_idx], &cn, &cval, &ldcontrib, &crlist,
            &ndelay, &delay_perm, &delay_val, &lddelay
            );
      // printf("[assemble_contrib_subtree] ndelay = %d, cval = %p\n", ndelay, cval);
      if(!cval) return; // child was all delays, nothing more to do

      int sa = ncol / blksz; // Index of first block in contrib
      int nr = (nrow-1) / blksz + 1;
      int ncontrib = nr-sa;
      for(int j = 0; j < cn; ++j) {

         int c = csnode.map[ j ]; // Destination column

         T const* src = &cval[j*ldcontrib];

         if (c >= snode.ncol) {

            int cc = c / blksz; // Destination block column
            int dest_col_sa = (ncol > cc*blksz) ? 0 : (cc*blksz-ncol); // First col in block

            for (int i = j; i < cn; ++i) {
               int r = csnode.map[ i ]; // Destination row in parent front
               int rr = r / blksz; // Destination block row
               // First row index in CB of destination block
               int dest_row_sa = (ncol > rr*blksz) ? 0 : (rr*blksz-ncol);
               Tile<T, PoolAlloc> &dest_blk = node.contrib_blocks[(rr-sa)+(cc-sa)*ncontrib];
               int dest_blk_lda = dest_blk.lda;
               T *dest = &dest_blk.a[ (c - ncol - dest_col_sa)*dest_blk_lda ];
               // Assemble destination block
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
         SymbolicFront const& csnode,
         void** child_contrib,
         int contrib_idx, // Index of subtree to assemble
         int delay_col
         ) {
      
      int ncol = node.get_ncol();
      size_t ldl = node.get_ldl(); // Leading dimension

      // Retreive contribution block from subtrees
      int cn, ldcontrib, ndelay, lddelay;
      double const *cval, *delay_val;
      int const *crlist, *delay_perm;
      spral_ssids_contrib_get_data(
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
   // assemble_subtree
   
   /// @brief Assemble a subtree to its parent
   template <typename T, typename PoolAlloc>
   void assemble_subtree(
         NumericFront<T,PoolAlloc>& node,
         SymbolicFront const& csnode,
         void** child_contrib, 
         int contrib_idx// Index of subtree to assemble
         ) {

      SymbolicFront snode = node.symb; // Symbolic node
      int ncol = node.get_ncol();
      size_t ldl = node.get_ldl(); // Leading dimension

      // Retreive contribution block from subtrees
      int cn, ldcontrib, ndelay, lddelay;
      double const *cval, *delay_val;
      int const *crlist, *delay_perm;
      spral_ssids_contrib_get_data(
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
         if (c < snode.ncol) {
            int ldd = node.get_ldl();
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

      SymbolicFront &csnode = cnode.symb; // Child symbolic node

      int ncol = node.get_ncol();
      size_t ldl = node.get_ldl();

      for(int i=0; i<cnode.ndelay_out; i++) {
         // Add delayed rows (from delayed cols)
         T *dest = &node.lcol[delay_col*(ldl+1)];
         int lds = align_lda<T>(csnode.nrow + cnode.ndelay_in);
         T *src = &cnode.lcol[(cnode.nelim+i)*(lds+1)];
         node.perm[delay_col] = cnode.perm[cnode.nelim+i];
         for(int j=0; j<cnode.ndelay_out-i; j++) {
            dest[j] = src[j];
         }
         // Add child's non-fully summed rows (from delayed cols)
         dest = node.lcol;
         src = &cnode.lcol[cnode.nelim*lds + cnode.ndelay_in +i*lds];
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

      int blksz = node.blksz;
      SymbolicFront snode = node.symb;

      int nrow = node.get_nrow();
      int ncol = node.get_ncol();
      size_t ldl = node.get_ldl();

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
         map[ snode.rlist[i] ] = i + node.ndelay_in;
      
      // Assemble front: fully-summed columns 
      for (auto* child=node.first_child; child!=NULL; child=child->next_child) {

         SymbolicFront &csnode = child->symb; // Children symbolic node

         int cm = csnode.nrow - csnode.ncol;
         csnode.map = new int[cm];
         for (int i=0; i<cm; i++)
            csnode.map[i] = map[ csnode.rlist[csnode.ncol+i] ];

         int ldcontrib = csnode.nrow - csnode.ncol;
         if (csnode.exec_loc == -1) {
            // Assemble contributions from child front

            // printf("[assemble] child->ndelay_out = %d\n", child->ndelay_out);

            /* Handle delays - go to back of node
             * (i.e. become the last rows as in lower triangular format) */
            // for(int i=0; i<child->ndelay_out; i++) {
            //    // Add delayed rows (from delayed cols)
            //    T *dest = &node.lcol[delay_col*(ldl+1)];
            //    int lds = align_lda<T>(csnode.nrow + child->ndelay_in);
            //    T *src = &child->lcol[(child->nelim+i)*(lds+1)];
            //    node.perm[delay_col] = child->perm[child->nelim+i];
            //    for(int j=0; j<child->ndelay_out-i; j++) {
            //       dest[j] = src[j];
            //    }
            //    // Add child's non-fully summed rows (from delayed cols)
            //    dest = node.lcol;
            //    src = &child->lcol[child->nelim*lds + child->ndelay_in +i*lds];
            //    for(int j=csnode.ncol; j<csnode.nrow; j++) {
            //       int r = map[ csnode.rlist[j] ];
            //       // int r = csnode.map[j];
            //       if(r < ncol) dest[r*ldl+delay_col] = src[j];
            //       else         dest[delay_col*ldl+r] = src[j];
            //    }
            //    delay_col++;
            // }

            assemble_delays(*child, delay_col, node);
            
            delay_col += child->ndelay_out;

            // Handle expected contributions (only if something there)
            if (ldcontrib>0) {
               // int *cache = new int[cm];
               // spral::ssids::cpu::assemble_expected(0, cm, node, *child, map, cache);
               // delete cache;
                  
               int cnrow = child->get_nrow();
               int cncol = child->get_ncol();
                  
               int csa = cncol / blksz;
               int cnr = child->get_nr(); // number of block rows
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
            spral_ssids_contrib_get_data(
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
                  int ldd = node.get_ldl();
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

         SymbolicFront const& child_sfront = child->symb;
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

               int cncol = child->get_ncol();
               int cnrow = child->get_nrow();
               int blksz = child->blksz;
               
               int csa = cncol / blksz;
               // Number of block rows in child node
               int cnr = child->get_nr();
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

} /* end of namespace spldlt */

