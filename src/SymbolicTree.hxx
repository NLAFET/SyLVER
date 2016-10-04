#pragma once

#include <cstdio>
#include <vector>

#include "ssids/cpu/SymbolicNode.hxx"

using namespace spral::ssids::cpu;

namespace spldlt {

   class SymbolicTree {
   public:
      SymbolicTree(int nnodes, int const* sptr, int const* sparent, long const* rptr, int const* rlist, int const* nptr, int const* nlist) 
         : nnodes_(nnodes), nodes_(nnodes_+1)
      {
         
         printf("[SymbolicAtree] create symbolic atree, nnodes: %d\n", nnodes_);
         
         for(int ni=0; ni<nnodes_; ++ni) {
            nodes_[ni].idx = ni;
            nodes_[ni].nrow = static_cast<int>(rptr[ni+1] - rptr[ni]);
            printf("[SymbolicAtree] nodes: %d, nrow: %d\n", ni, nodes_[ni].nrow);
            nodes_[ni].ncol = sptr[ni+1] - sptr[ni];
            printf("[SymbolicAtree] nodes: %d, ncol: %d\n", ni, nodes_[ni].ncol);
            nodes_[ni].first_child = nullptr;
            nodes_[ni].next_child = nullptr;
            nodes_[ni].rlist = &rlist[rptr[ni]-1]; // rptr is Fortran indexed
            nodes_[ni].num_a = nptr[ni+1] - nptr[ni];
            nodes_[ni].amap = &nlist[2*(nptr[ni]-1)]; // nptr is Fortran indexed
            nodes_[ni].parent = sparent[ni]-1; // sparent is Fortran indexed
            printf("[SymbolicAtree] nodes: %d, parent: %d\n", ni, nodes_[ni].parent);
         }

         /* Count size of factors */
         nfactor_ = 0;
         for(int ni=0; ni<nnodes_; ++ni)
            nfactor_ += static_cast<size_t>(nodes_[ni].nrow)*nodes_[ni].ncol;

      }
   
   private:
      int nnodes_;
      size_t nfactor_;
      std::vector<SymbolicNode> nodes_;
   };
} /* end of namespace spldlt */
