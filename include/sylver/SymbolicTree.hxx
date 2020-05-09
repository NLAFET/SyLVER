#pragma once

#include <cstdio>
#include <vector>

#include "ssids/cpu/cpu_iface.hxx"
// #include "ssids/cpu/SymbolicNode.hxx"

#include "sylver/SymbolicFront.hxx"

// using namespace spral::ssids::cpu;

namespace sylver {

class SymbolicTree {
public:
   SymbolicTree(
         void* akeep, int n, int nnodes, int const* sptr, int const* sparent,
         long const* rptr, int const* rlist, long const* nptr, long const* nlist, 
         int nsubtrees, int const* subtrees, int const* small, int const* contrib_dest,
         int const* exec_loc); // FIXME exec_loc and small are redundent
   // int nparts, int const* part, int const* contrib_idx, int const* exec_loc, 
   // int const* contrib_dest)

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
      return maxfront_*spral::ssids::cpu::align_lda<T>(maxfront_);
   }
      
   /// @brief Return the number of nodes in the assembly tree
   inline int nnodes() const {
      return nnodes_;
   }

   /// @brief Return the number of subtrees in the assembly tree
   inline int nsubtrees() const {
      return nsubtrees_;
   }

   /// @brief Return akeep structure
   inline void* akeep() const {
      return akeep_;
   }

   inline int const* subtrees() const {
      return subtrees_;
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
};
   
} /* end of namespace sylver */
