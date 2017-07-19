#pragma once

#include <cstdio>
#include <vector>

#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/SymbolicNode.hxx"

#include "SymbolicSNode.hxx"

using namespace spral::ssids::cpu;

namespace spldlt {

   class SymbolicTree {
   public:
      SymbolicTree(int n, int nnodes, int const* sptr, int const* sparent, long const* rptr, int const* rlist, long const* nptr, long const* nlist, int nparts, int const* part, int const* exec_loc, int const* contrib_idx)
         : n(n), nnodes_(nnodes), nodes_(nnodes_+1)
      {

         // FIXME multifrontal mode: use template paramter?
         // bool mf = true;

         for(int ni=0; ni<nnodes_; ++ni) 
            nodes_[ni].least_desc = ni;
         
         // printf("[SymbolicTree] create symbolic atree, nnodes: %d\n", nnodes_);
         maxfront_ = 0;
         for(int ni=0; ni<nnodes_; ++ni) {
            
            // SymbolicNode info
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

            // SymbolicSNode info
            nodes_[ni].sa = sptr[ni];
            nodes_[ni].en = sptr[ni+1]-1;
            
            // Setup least_desc for easily traverse subtrees
            nodes_[nodes_[ni].parent].least_desc = std::min(
                  nodes_[nodes_[ni].parent].least_desc,
                  nodes_[ni].least_desc);
         }
         // Useful in a multifrontal mode 
         nodes_[nnodes_].first_child = nullptr; // List of roots
         /* Build child linked lists */
         for(int ni=0; ni<nnodes_; ++ni) {
            SymbolicNode *parent = &nodes_[ std::min(nodes_[ni].parent, nnodes_) ];
            nodes_[ni].next_child = parent->first_child;
            parent->first_child = &nodes_[ni];
         }
         
         /* Record contribution block inputs */
         for(int ci = 0; ci < nparts; ++ci) {
            int idx = contrib_idx[ci]-1; // contrib_idx is Fortran indexed
            nodes_[idx].contrib.push_back(ci);
            printf("[SymbolicTree] %d -> %d, exec_loc = %d\n", ci+1, idx+1, exec_loc[ci]);
         }
         // Count size of factors
         nfactor_ = 0;
         for(int ni=0; ni<nnodes_; ++ni)
            nfactor_ += static_cast<size_t>(nodes_[ni].nrow)*nodes_[ni].ncol;
         
         // Setup node partition and execution location information
         for(int p = 0; p < nparts; ++p) {
            for (int ni = part[p]-1; ni < part[p+1]-1; ++ni) {
               nodes_[ni].part = p;
               nodes_[ni].exec_loc = exec_loc[p];
               // printf("[SymbolicTree] node: %d, part: %d, exec_loc: %d\n", ni+1, p+1, exec_loc[p]);
            }
         }

      }

      size_t get_factor_mem_est(double multiplier) const {
         size_t mem = n*sizeof(int) + (2*n+nfactor_)*sizeof(double);
         return std::max(mem, static_cast<size_t>(mem*multiplier));
      }
      template <typename T>
      size_t get_pool_size() const {
         return maxfront_*align_lda<double>(maxfront_);
      }

      SymbolicSNode& operator[](int idx) {
         return nodes_[idx];
      }
      
      SymbolicSNode const& operator[](int idx) const {
         return nodes_[idx];
      }

   public:
      int const n; //< Maximum row index
   private:
      int nnodes_;
      size_t nfactor_;
      size_t maxfront_;
      std::vector<SymbolicSNode> nodes_;

      template <typename T, size_t PAGE_SIZE, typename FactorAlloc, bool posdef>
      friend class NumericTree;
   };
} /* end of namespace spldlt */
