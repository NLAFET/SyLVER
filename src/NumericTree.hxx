#include "SymbolicTree.hxx"

#include "ssids/cpu/BuddyAllocator.hxx"

namespace spldlt {

   template<typename T,
            size_t PAGE_SIZE,
            typename FactorAllocator>
   class NumericTree {
      typedef BuddyAllocator<T,std::allocator<T>> PoolAllocator;
   public:
      
      NumericTree(SymbolicTree const& symb, T const* aval)
         : symb_(symb)
      {}

   private:
      SymbolicTree const& symb_;
   };

} /* end of namespace spldlt */
