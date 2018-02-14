/// \file
/// \copyright 2016- The Science and Technology Facilities Council (STFC)
/// \author Jonathan Hogg
/// \author Florent Lopez

#pragma once

#include "ssids/cpu/cpu_iface.hxx"
// #include "ssids/cpu/NumericNode.hxx"
// #include "ssids/cpu/SymbolicNode.hxx"
// #include "ssids/cpu/Workspace.hxx"
#include "ssids/cpu/kernels/assemble.hxx"

#include "NumericFront.hxx"

// using namespace spral::ssids::cpu;

namespace spldlt {

   // /* alloc node */
   // template <typename T,
   //           typename FactorAlloc,
   //           typename PoolAlloc,
   //           int posdef = true>
   // void alloc_node(
   //       SymbolicNode const& snode,
   //       spldlt::NumericNode<T,PoolAlloc>& node,
   //       FactorAlloc& factor_alloc,
   //       PoolAlloc& pool_alloc
   //       ) {

   //    T *scaling = NULL;

   //    /* Rebind allocators */
   //    typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<double> FADoubleTraits;
   //    typename FADoubleTraits::allocator_type factor_alloc_double(factor_alloc);
   //    typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<int> FAIntTraits;
   //    typename FAIntTraits::allocator_type factor_alloc_int(factor_alloc);
   //    typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;

   //    /* Count incoming delays and determine size of node */
   //    node.ndelay_in = 0;
      
   //    int nrow = snode.nrow + node.ndelay_in;
   //    int ncol = snode.ncol + node.ndelay_in;

   //    /* Get space for node now we know it size using Fortran allocator + zero it*/
   //    // NB L is  nrow x ncol and D is 2 x ncol (but no D if posdef)
   //    size_t ldl = align_lda<double>(nrow);
   //    size_t len = posdef ?  ldl    * ncol  // posdef
   //       : (ldl+2) * ncol; // indef (includes D)
   //    node.lcol = FADoubleTraits::allocate(factor_alloc_double, len);
   // }

   ////////////////////////////////////////////////////////////////////////////////   
   // Allocate memory assocaited with the frontal matrix front
   template<typename T, 
            typename FactorAlloc, 
            typename PoolAlloc>
   void alloc_front_posdef(
         NumericFront<T,PoolAlloc>& front,
         FactorAlloc& factor_alloc,
         PoolAlloc& pool_alloc) {

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

      int nrow = sfront.nrow + front.ndelay_in;
      int ncol = sfront.ncol + front.ndelay_in;

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
         FactorAlloc& factor_alloc,
         PoolAlloc& pool_alloc) {

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
      /* Count incoming delays and determine size of node */
      for(auto* child=front.first_child; child!=NULL; child=child->next_child) {
         front.ndelay_in += child->ndelay_out;
      }
      for(int contrib_idx : sfront.contrib) {
         int cn, ldcontrib, ndelay, lddelay;
         double const *cval, *delay_val;
         int const *crlist, *delay_perm;
         spral_ssids_contrib_get_data(
               child_contrib[contrib_idx], &cn, &cval, &ldcontrib, &crlist,
               &ndelay, &delay_perm, &delay_val, &lddelay
               );
         front.ndelay_in += ndelay;
      }

      printf("[alloc_front] ndelay_in = %d\n", front.ndelay_in);
      // front.ndelay_in = 0; // debug

      int nrow = sfront.nrow + front.ndelay_in;
      int ncol = sfront.ncol + front.ndelay_in;

      /* Get space for node now we know it size using Fortran allocator + zero it*/
      // NB L is  nrow x ncol and D is 2 x ncol (but no D if posdef)
      size_t ldl = spral::ssids::cpu::align_lda<double>(nrow);
      // size_t len = posdef ?  ldl    * ncol  // posdef
      //    : (ldl+2) * ncol; // indef (includes D)
      size_t len =  (ldl+2) * ncol; // indef (includes D)

      front.lcol = FADoubleTraits::allocate(factor_alloc_double, len);

      // /* Get space for contribution block + (explicitly do not zero it!) */
      // node.alloc_contrib();
      front.alloc_contrib_blocks();

      /* Alloc + set perm for expected eliminations at this node (delays are set
         * when they are imported from children) */
      front.perm = FAIntTraits::allocate(factor_alloc_int, ncol); // ncol fully summed variables
      for(int i=0; i<sfront.ncol; i++)
         front.perm[i] = sfront.rlist[i];

   }

   ////////////////////////////////////////////////////////////////////////////////
   template <typename T,
             // typename FactorAlloc,
             typename PoolAlloc>
   void init_node(
         SymbolicFront const& sfront,
         NumericFront<T,PoolAlloc>& front,
         // FactorAlloc& factor_alloc,
         // PoolAlloc& pool_alloc,
         //         Workspace *work,
         T const* aval) {

      // printf("[kernels] init node\n");
      bool posdef = true;
      T *scaling = NULL;

      // /* Count incoming delays and determine size of node */
      // node.ndelay_in = 0;
      
      // int nrow = snode.nrow + node.ndelay_in;
      // int ncol = snode.ncol + node.ndelay_in;

      // /* Get space for node now we know it size using Fortran allocator + zero it*/
      // // NB L is  nrow x ncol and D is 2 x ncol (but no D if posdef)
      // size_t ldl = align_lda<double>(nrow);
      // size_t len = posdef ?  ldl    * ncol  // posdef
      //    : (ldl+2) * ncol; // indef (includes D)
      // node.lcol = FADoubleTraits::allocate(factor_alloc_double, len);

      /* Add A */
      // add_a_block<T, NumericNode<T,PoolAlloc>>(0, snode.num_a, node, aval, NULL);  
      spral::ssids::cpu::add_a_block(0, sfront.num_a, front, aval, scaling);
   }

   // Terminate node.
   // Deallocate contribution block
   template <typename T, typename PoolAlloc>
   void fini_node(NumericFront<T,PoolAlloc>& node) {

      // printf("[fini_node]\n");
      
      // deallocate contribution block
      // node.free_contrib();

      // deallocate contribution block
      node.free_contrib_blocks();
   }

   ////////////////////////////////////////////////////////////////////////////////   
   // Assemble block

   template <typename T, typename PoolAlloc>
   void assemble_block(NumericFront<T,PoolAlloc>& node, 
                       NumericFront<T,PoolAlloc>& cnode, 
                       int ii, int jj, int *cmap, int blksz) {
      
      SymbolicFront const& csnode = cnode.symb;

      // printf("[assemble_block] ii: %d, jj: %d\n", ii, jj);
      
      // Source node
      int cm = csnode.nrow - csnode.ncol;
      int csa = csnode.ncol / blksz; // Index of first block in contrib
      int cnr = (csnode.nrow-1) / blksz + 1; // number of block rows in child node
      int cncontrib = cnr-csa;
      // Source block
      Block<T, PoolAlloc> &blk = cnode.contrib_blocks[(ii-csa)+(jj-csa)*cncontrib];
      int blk_lda = blk.lda;
      int blk_m = blk.m;
      int blk_n = blk.n;

      // printf("[assemble_block] csa: %d\n", csa);
      // printf("[assemble_block] blk_m: %d, blk_n: %d, blk_lda: %d\n", 
      //        blk_m, blk_n, blk_lda);

      // index of first column in CB
      int col_sa = (csnode.ncol > jj*blksz) ? 0 : (jj*blksz-csnode.ncol);
      // Global index of last column
      // int c_en = std::min((jj+1)*blksz-csnode.ncol, cm); 
      // int r_sa = (csnode.ncol > ii*blksz) ? 0 : (ii*blksz-csnode.ncol); // first col in block
      // Global index of last row
      // int r_en = std::min((ii+1)*blksz-csnode.ncol, cm);
      
      // Index of first row in CB
      int row_sa = (csnode.ncol > ii*blksz) ? 0 : (ii*blksz-csnode.ncol);

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
   void assemble_contrib_block(NumericFront<T,PoolAlloc>& node, 
                               NumericFront<T,PoolAlloc>& cnode, 
                               int ii, int jj, int *cmap, int blksz) {

      // printf("[assemble_contrib_block]\n");

      SymbolicFront const& csnode = cnode.symb;
      
      int cm = csnode.nrow - csnode.ncol;
      int ncol = node.symb.ncol; // no delays!
      int nrow = node.symb.nrow;

      // Source block
      int csa = csnode.ncol / blksz; // Index of first block in contrib
      int cnr = (csnode.nrow-1) / blksz + 1; // number of block rows in child node
      int cncontrib = cnr-csa;
      Block<T, PoolAlloc> &src_blk = cnode.contrib_blocks[(ii-csa)+(jj-csa)*cncontrib];
      int src_blk_lda = src_blk.lda;
      int src_blk_m = src_blk.m;
      int src_blk_n = src_blk.n;

      // Destination block
      int sa = ncol / blksz; // Index of first block in contrib
      int nr = (nrow-1) / blksz + 1;
      int ncontrib = nr-sa;
      
      // Colum indexes
      int col_sa = (csnode.ncol > jj*blksz) ? 0 : (jj*blksz-csnode.ncol); // first col in block
      // int c_en = std::min((jj+1)*blksz-csnode.ncol, cm); // last col in block
      // row indexes
      // int r_sa = (csnode.ncol > ii*blksz) ? 0 : (ii*blksz-csnode.ncol); // first col in block
      // int r_en = std::min((ii+1)*blksz-csnode.ncol, cm); // last col in block

      // First row index in CB of source block
      int row_sa = (csnode.ncol > ii*blksz) ? 0 : (ii*blksz-csnode.ncol);

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
               // printf("[assemble_contrib_block] csa: %d\n", csa);
               // printf("[assemble_contrib_block] rr: %d, cc: %d\n", rr, cc);
               // printf("[assemble_contrib_block] r: %d, c: %d\n", r, c);
               // printf("[assemble_contrib_block] dest_row_sa: %d, dest_col_sa: %d\n", dest_row_sa, dest_col_sa);
               Block<T, PoolAlloc> &dest_blk = node.contrib_blocks[(rr-sa)+(cc-sa)*ncontrib];
               int dest_blk_lda = dest_blk.lda;
               T *dest = &dest_blk.a[ (c - ncol - dest_col_sa)*dest_blk_lda ];

               dest[ r - ncol - dest_row_sa ] += src[i];
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
   template <typename T, typename PoolAlloc>
   void assemble_subtree (
         NumericFront<T,PoolAlloc>& node,
         SymbolicFront const& csnode,
         void** child_contrib, 
         int contrib_idx // Index of subtree to assemble
         ) {

      printf("[assemble_subtree]\n");

      SymbolicFront const& snode = node.symb;

      // Retreive contribution block from subtrees
      int cn, ldcontrib, ndelay, lddelay;
      double const *cval, *delay_val;
      int const *crlist, *delay_perm;
      spral_ssids_contrib_get_data(
            child_contrib[contrib_idx], &cn, &cval, &ldcontrib, &crlist,
            &ndelay, &delay_perm, &delay_val, &lddelay
            );

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


   ////////////////////////////////////////////////////////////////////////////////   
   // Assemble contrib subtree
   template <typename T, typename PoolAlloc>
   void assemble_contrib_subtree(
         NumericFront<T,PoolAlloc>& node,
         SymbolicFront const& csnode,
         void** child_contrib, 
         int contrib_idx, // Index of subtree to assemble
         int blksz) {

      SymbolicFront const& snode = node.symb;

      /* Initialise variables */
      int ncol = snode.ncol + node.ndelay_in;
      int nrow = snode.nrow + node.ndelay_in;

      // Retreive contribution block from subtrees
      int cn, ldcontrib, ndelay, lddelay;
      double const *cval, *delay_val;
      int const *crlist, *delay_perm;
      spral_ssids_contrib_get_data(
            child_contrib[contrib_idx], &cn, &cval, &ldcontrib, &crlist,
            &ndelay, &delay_perm, &delay_val, &lddelay
            );
      printf("[assemble_contrib_subtree] ndelay = %d, cval = %p\n", ndelay, cval);
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
               Block<T, PoolAlloc> &dest_blk = node.contrib_blocks[(rr-sa)+(cc-sa)*ncontrib];
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

} /* end of namespace spldlt */

