/** \file
 *  \copyright 2016- The Science and Technology Facilities Council (STFC)
 *  \author    Florent Lopez
 */
// SyLVER
#include "SymbolicTree.hxx"
#include "sylver_ciface.hxx"

namespace sylver { 
   namespace splu {

      template<typename T>
      class NumericTreeUnsym {
   
      public:
         // Delete copy constructors for safety re allocated memory
         NumericTreeUnsym(const NumericTreeUnsym&) =delete;
         NumericTreeUnsym& operator=(const NumericTreeUnsym&) =delete;

         NumericTreeUnsym(
               spldlt::SymbolicTree& symbolic_tree, 
               T *val, 
               struct sylver_options_c &options)
            : symb_(symbolic_tree)
         {
         
            printf("[NumericTreeUnsym] u = %e, small = %e\n", options.u, options.small);
            
            

         }

      private:
         spldlt::SymbolicTree& symb_; // Structure holding symbolic factorization data 
      };

   } // End of namespace splu
} // End of namespace sylver
