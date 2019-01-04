// SyLVER
#include "SymbolicTree.hxx"

namespace spldlt {

   template<typename T>
   class NumericTreeUnsym {
   
   public:
      // Delete copy constructors for safety re allocated memory
      NumericTreeUnsym(const NumericTreeUnsym&) =delete;
      NumericTreeUnsym& operator=(const NumericTreeUnsym&) =delete;

      NumericTreeUnsym(
      SymbolicTree& symbolic_tree, T *lval)
         : symb_(symbolic_tree)
      {
         printf("[NumericTreeUnsym]\n");
      }

   private:
      SymbolicTree& symb_;

   };

}
