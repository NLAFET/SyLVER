#pragma once

#include <cstdio>
#include <vector>

#include "ssids/cpu/cpu_iface.hxx"
// #include "ssids/cpu/SymbolicNode.hxx"

#include "SymbolicFront.hxx"

// using namespace spral::ssids::cpu;

namespace spldlt {

   class SymbolicTree {
   public:
      SymbolicTree(
            void* akeep, int n, int nnodes, int const* sptr, int const* sparent,
            long const* rptr, int const* rlist, long const* nptr, long const* nlist, 
            int nsubtrees, int const* subtrees, int const* small, int const* contrib_dest)
            // int nparts, int const* part, int const* contrib_idx, int const* exec_loc, 
            // int const* contrib_dest)
         : akeep_(akeep), n(n), nnodes_(nnodes), fronts_(nnodes_+1),
           nsubtrees_(nsubtrees), subtrees_(subtrees)
           // nparts_(nparts), part_(part)
      {
         printf("[SymbolicTree] nsubtrees = %d\n", nsubtrees);

         // for(int ni=0; ni<nnodes_; ++ni) 
         //    fronts_[ni].least_desc = ni;

         maxfront_ = 0;
         for(int ni=0; ni<nnodes_; ++ni) {
            // SymbolicNode info
            fronts_[ni].idx = ni; // Node index
            fronts_[ni].nrow = static_cast<int>(rptr[ni+1] - rptr[ni]); // Number of rows
            fronts_[ni].ncol = sptr[ni+1] - sptr[ni];
            fronts_[ni].first_child = nullptr;
            fronts_[ni].next_child = nullptr;
            fronts_[ni].rlist = &rlist[rptr[ni]-1]; // rptr is Fortran indexed
            fronts_[ni].num_a = nptr[ni+1] - nptr[ni];
            fronts_[ni].amap = &nlist[2*(nptr[ni]-1)]; // nptr is Fortran indexed
            fronts_[ni].parent = sparent[ni]-1; // sparent is Fortran indexed
            
            maxfront_ = std::max(maxfront_, (size_t) fronts_[ni].nrow); // Number of columns

            // // Setup useful info in supernodal mode 
            // // SymbolicSNode info
            // nodes_[ni].sa = sptr[ni];
            // nodes_[ni].en = sptr[ni+1]-1;
            
            // // Setup least_desc for easily traverse subtrees
            // nodes_[nodes_[ni].parent].least_desc = std::min(
            //       nodes_[nodes_[ni].parent].least_desc,
            //       nodes_[ni].least_desc);

         }

         /* Build child linked lists */
         for(int ni=0; ni<nnodes_; ++ni) {
            SymbolicFront *parent = &fronts_[ std::min(fronts_[ni].parent, nnodes_) ];
            fronts_[ni].next_child = parent->first_child;
            parent->first_child = &fronts_[ni];
         }
         
         /* Record contribution block inputs */
         // for(int ci = 0; ci < nparts; ++ci) {
         for(int ci = 0; ci < nsubtrees; ++ci) {
            int idx = contrib_dest[ci]-1; // contrib_dest is Fortran indexed
            // printf("[SymbolicTree] %d -> %d, exec_loc = %d\n", ci+1, idx+1, exec_loc[ci]);
            if (idx > 0) // idx equal to 0 means that ci is a root subtree 
               fronts_[idx].contrib.push_back(ci);
         }

         // count size of factors
         nfactor_ = 0;
         for(int ni=0; ni<nnodes_; ++ni)
            nfactor_ += static_cast<size_t>(fronts_[ni].nrow)*fronts_[ni].ncol;

         printf("[SymbolicTree] num factors: %zu\n", nfactor_);

         // Setup node partition and execution location information
         // for(int p = 0; p < nparts; ++p) {
         //    printf("[SybolicSubtree] part = %d, contrib_idx = %d\n", p, contrib_idx[p]);
         //    for (int ni = part[p]-1; ni < part[p+1]-1; ++ni) {
         //       fronts_[ni].part = p;
         //       fronts_[ni].exec_loc = exec_loc[p];
         //       fronts_[ni].contrib_idx = contrib_idx[p]-1; // contrib_idx is Fortran indexed
         //    }
         // }

         // Setup node partition and execution location information
         for(int ni=0; ni<nnodes_; ++ni) {
            fronts_[ni].part = -1;
            (small[ni] == 0) ? fronts_[ni].exec_loc = -1 : fronts_[ni].exec_loc = 1;   
         }
         for(int p = 0; p < nsubtrees; ++p) {
            int idx = subtrees[p]-1; // subtrees is Fortran
            fronts_[idx].part = p;
            fronts_[idx].contrib_idx = p;
         }
      }

      SymbolicFront& operator[](int idx) {
         return fronts_[idx];
      }
      
      SymbolicFront const& operator[](int idx) const {
         return fronts_[idx];
      }

      size_t get_factor_mem_est(double multiplier) const {
         size_t mem = n*sizeof(int) + (2*n+nfactor_)*sizeof(double);
         return std::max(mem, static_cast<size_t>(mem*multiplier));
      }

      template <typename T>
      size_t get_pool_size() const {
         return maxfront_*spral::ssids::cpu::align_lda<double>(maxfront_);
      }

   public:
      int const n; //< Maximum row index
   private:
      void* akeep_;
      int nnodes_;
      size_t nfactor_;
      size_t maxfront_;
      // int nparts_;
      // int const* part_;
      int nsubtrees_;
      int const* subtrees_;
      std::vector<SymbolicFront> fronts_;

      template<typename T, size_t PAGE_SIZE, typename FactorAllocator, bool posdef>
      friend class NumericTree;
   };
   
} /* end of namespace spldlt */
