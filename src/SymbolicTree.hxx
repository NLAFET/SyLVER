#pragma once

#include <cstdio>
#include <vector>

#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/SymbolicNode.hxx"

using namespace spral::ssids::cpu;

namespace spldlt {

   class SymbolicTree {
   public:
      SymbolicTree(int n, int nnodes, int const* sptr, int const* sparent, long const* rptr, int const* rlist, int const* nptr, int const* nlist) 
         : n(n), nnodes_(nnodes), nodes_(nnodes_+1)
      {
         
         // printf("[SymbolicTree] create symbolic atree, nnodes: %d\n", nnodes_);
         maxfront_ = 0;         
         for(int ni=0; ni<nnodes_; ++ni) {
            nodes_[ni].idx = ni;
            nodes_[ni].nrow = static_cast<int>(rptr[ni+1] - rptr[ni]);
            // printf("[SymbolicTree] nodes: %d, nrow: %d\n", ni, nodes_[ni].nrow);
            nodes_[ni].ncol = sptr[ni+1] - sptr[ni];
            // printf("[SymbolicTree] nodes: %d, ncol: %d\n", ni, nodes_[ni].ncol);
            nodes_[ni].first_child = nullptr;
            nodes_[ni].next_child = nullptr;
            nodes_[ni].rlist = &rlist[rptr[ni]-1]; // rptr is Fortran indexed
            nodes_[ni].num_a = nptr[ni+1] - nptr[ni];
            nodes_[ni].amap = &nlist[2*(nptr[ni]-1)]; // nptr is Fortran indexed
            nodes_[ni].parent = sparent[ni]-1; // sparent is Fortran indexed
            // printf("[SymbolicTree] nodes: %d, parent: %d\n", ni, nodes_[ni].parent);
            maxfront_ = std::max(maxfront_, (size_t) nodes_[ni].nrow);
         }

         /* Count size of factors */
         nfactor_ = 0;
         for(int ni=0; ni<nnodes_; ++ni)
            nfactor_ += static_cast<size_t>(nodes_[ni].nrow)*nodes_[ni].ncol;

      }

      size_t get_factor_mem_est(double multiplier) const {
         size_t mem = n*sizeof(int) + (2*n+nfactor_)*sizeof(double);
         return std::max(mem, static_cast<size_t>(mem*multiplier));
      }
      template <typename T>
      size_t get_pool_size() const {
         return maxfront_*align_lda<double>(maxfront_);
      }

      SymbolicNode const& operator[](int idx) const {
         return nodes_[idx];
      }
   public:
      int const n; //< Maximum row index
   private:
      int nnodes_;
      size_t nfactor_;
      size_t maxfront_;
      std::vector<SymbolicNode> nodes_;

      template <typename T, size_t PAGE_SIZE, typename FactorAlloc, bool posdef>
      friend class NumericTree;
   };
} /* end of namespace spldlt */
