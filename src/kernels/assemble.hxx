#pragma once

#include "ssids/cpu/NumericNode.hxx"
#include "ssids/cpu/SymbolicNode.hxx"
// #include "ssids/cpu/Workspace.hxx"
#include "ssids/cpu/kernels/assemble.hxx"

using namespace spral::ssids::cpu;

namespace spldlt {

   /* alloc node */
   template <typename T,
             typename FactorAlloc,
             typename PoolAlloc,
             int posdef = true>
   void alloc_node(
         SymbolicNode const& snode,
         NumericNode<T,PoolAlloc>& node,
         FactorAlloc& factor_alloc,
         PoolAlloc& pool_alloc
         ) {

      T *scaling = NULL;

      /* Rebind allocators */
      typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<double> FADoubleTraits;
      typename FADoubleTraits::allocator_type factor_alloc_double(factor_alloc);
      typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<int> FAIntTraits;
      typename FAIntTraits::allocator_type factor_alloc_int(factor_alloc);
      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;

      /* Count incoming delays and determine size of node */
      node.ndelay_in = 0;
      
      int nrow = snode.nrow + node.ndelay_in;
      int ncol = snode.ncol + node.ndelay_in;

      /* Get space for node now we know it size using Fortran allocator + zero it*/
      // NB L is  nrow x ncol and D is 2 x ncol (but no D if posdef)
      size_t ldl = align_lda<double>(nrow);
      size_t len = posdef ?  ldl    * ncol  // posdef
         : (ldl+2) * ncol; // indef (includes D)
      node.lcol = FADoubleTraits::allocate(factor_alloc_double, len);
   }
   
   /* alloc node in MF context
      FIXME bool mf as input parameter?
    */
   template<typename T, 
            typename FactorAlloc, 
            typename PoolAlloc,
            bool posdef = true>
   void alloc_node_mf(
         SymbolicNode const& snode,
         NumericNode<T,PoolAlloc>& node,
         FactorAlloc& factor_alloc,
         PoolAlloc& pool_alloc
         ) {

      T *scaling = NULL;

      /* Rebind allocators */
      typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<double> FADoubleTraits;
      typename FADoubleTraits::allocator_type factor_alloc_double(factor_alloc);
      typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<int> FAIntTraits;
      typename FAIntTraits::allocator_type factor_alloc_int(factor_alloc);
      typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;

      /* Count incoming delays and determine size of node */
      node.ndelay_in = 0;
      
      int nrow = snode.nrow + node.ndelay_in;
      int ncol = snode.ncol + node.ndelay_in;

      /* Get space for node now we know it size using Fortran allocator + zero it*/
      // NB L is  nrow x ncol and D is 2 x ncol (but no D if posdef)
      size_t ldl = align_lda<double>(nrow);
      size_t len = posdef ?  ldl    * ncol  // posdef
         : (ldl+2) * ncol; // indef (includes D)
      node.lcol = FADoubleTraits::allocate(factor_alloc_double, len);

      /* Get space for contribution block + (explicitly do not zero it!) */
      node.alloc_contrib();

      // TODO: pivoting

      /* Alloc + set perm for expected eliminations at this node (delays are set
       * when they are imported from children) */
      // node.perm = FAIntTraits::allocate(factor_alloc_int, ncol); // ncol fully summed variables
      // for(int i=0; i<snode.ncol; i++)
      //    node.perm[i] = snode.rlist[i];

      /* If we have no children, we're done. */
      // if(node.first_child == nullptr && snode.contrib.size() == 0) return;
   }

   template <typename T,
             // typename FactorAlloc,
             typename PoolAlloc>
   void init_node(
         SymbolicNode const& snode,
         NumericNode<T,PoolAlloc>& node,
         // FactorAlloc& factor_alloc,
         // PoolAlloc& pool_alloc,
         //         Workspace *work,
         T const* aval
         ) {

      // printf("[kernels] init node\n");
      bool posdef = true;
      T *scaling = NULL;

      // /* Rebind allocators */
      // typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<double> FADoubleTraits;
      // typename FADoubleTraits::allocator_type factor_alloc_double(factor_alloc);
      // typedef typename std::allocator_traits<FactorAlloc>::template rebind_traits<int> FAIntTraits;
      // typename FAIntTraits::allocator_type factor_alloc_int(factor_alloc);
      // typedef typename std::allocator_traits<PoolAlloc>::template rebind_alloc<int> PoolAllocInt;

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
      add_a_block(0, snode.num_a, node, aval, scaling);
   }

   /*
    * r block row
    * c block col
    */ 
   template <typename T, typename PoolAlloc, typename MapVector>
   void assemble_block(NumericNode<T,PoolAlloc>& node, NumericNode<T,PoolAlloc> const& cnode, 
                       int ii, int jj, MapVector const& map, int blksz) {
      
      SymbolicNode const& csnode = cnode.symb;
      
      int cm = csnode.nrow - csnode.ncol;

      // colum indexes
      int c_sa = (csnode.ncol > jj*blksz) ? 0 : (jj*blksz-csnode.ncol); // first col in block
      int c_en = std::min((jj+1)*blksz-csnode.ncol, cm); // last col in block
      // row indexes
      // int r_sa = (csnode.ncol > ii*blksz) ? 0 : (ii*blksz-csnode.ncol); // first col in block
      int r_en = std::min((ii+1)*blksz-csnode.ncol, cm); // last col in block

      // loop over column in block
      for (int j=c_sa; j<c_en; j++) {

         // printf("[factor_mf] col size: %d\n", cm-j);
         // printf("jj: %d, col: %d\n", jj, col);
         int c = map[ csnode.rlist[csnode.ncol+j] ];
         T *src = &(cnode.contrib[j*cm]);
                        
         int ncol = node.symb.ncol; // no delays!

         // printf("[factor_mf] c: %d, ncol: %d\n", c, ncol);

         if (c < ncol) {
            int ldd = node.get_ldl();
            T *dest = &node.lcol[c*ldd];

            // int const* idx = &cache[j];                           
            // loop over rows in block

            int r_sa = (ii==jj) ? j : (ii*blksz-csnode.ncol); // first col in block

            for (int i=r_sa; i<r_en; i++) {              

               // printf("[factor_mf] i: %d\n", i);

               // int ii = map[ csnode.rlist[csnode.ncol+col+row] ];
               // dest[ idx[i] ] += src[i];
               dest[ map[ csnode.rlist[csnode.ncol+i] ] ] += src[i];
            }
         }
      }

   }
   
   template <typename T, typename PoolAlloc, typename MapVector>
   void assemble_contrib_block(NumericNode<T,PoolAlloc>& node, NumericNode<T,PoolAlloc> const& cnode, 
                               int ii, int jj, MapVector const& map, int blksz) {


      SymbolicNode const& csnode = cnode.symb;
      
      int cm = csnode.nrow - csnode.ncol;
      int ncol = node.symb.ncol; // no delays!

      // colum indexes
      int c_sa = (csnode.ncol > jj*blksz) ? 0 : (jj*blksz-csnode.ncol); // first col in block
      int c_en = std::min((jj+1)*blksz-csnode.ncol, cm); // last col in block
      // row indexes
      // int r_sa = (csnode.ncol > ii*blksz) ? 0 : (ii*blksz-csnode.ncol); // first col in block
      int r_en = std::min((ii+1)*blksz-csnode.ncol, cm); // last col in block

      // loop over columns in block jj
      for (int j=c_sa; j<c_en; j++) {

         // printf("[factor_mf] col size: %d\n", cm-j);
         // printf("jj: %d, col: %d\n", jj, col);
         int c = map[ csnode.rlist[csnode.ncol+j] ];
         T *src = &(cnode.contrib[j*cm]);
                        
         // printf("[factor_mf] c: %d, ncol: %d\n", c, ncol);
         
         if (c >= ncol) {
            int ldd = node.symb.nrow - node.symb.ncol;
            T *dest = &node.contrib[(c-ncol)*ldd];

            // int const* idx = &cache[j];               
            // loop over rows in block

            // int r_sa = (csnode.ncol > ii*blksz) ? 0 : (ii*blksz-csnode.ncol); // first col in block
            int r_sa = (ii==jj) ? j : (ii*blksz-csnode.ncol); // first col in block

            for (int i=r_sa; i<r_en; i++) {              
            // for (int i=j; i<cm; i++) {          

               // printf("[factor_mf] i: %d\n", i);

               // int ii = map[ csnode.rlist[csnode.ncol+col+row] ];
               // dest[ idx[i] ] += src[i];
               dest[ map[ csnode.rlist[csnode.ncol+i] ] - ncol ] += src[i];
            }
         }
      }

   }

} /* end of namespace spldlt */

