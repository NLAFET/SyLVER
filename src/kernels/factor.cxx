/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

#include "factor.hxx"

namespace spldlt {

   // @brief Factor subtree kernel in double precision
   extern "C" void spldlt_factor_subtree_c(
         void *akeep, 
         void *fkeep,
         int p,
         double *aval, 
         double *scaling, 
         void **child_contrib, 
         struct spral::ssids::cpu::cpu_factor_options *options,
         spral::ssids::cpu::ThreadStats *stats);


   template <typename T>
   void factor_subtree(
         void *akeep,
         void *fkeep,
         int p,
         T *aval, 
         T *scaling, 
         void **child_contrib, 
         struct spral::ssids::cpu::cpu_factor_options *options,
         spral::ssids::cpu::ThreadStats *stats) {
      
      std::cout << "size of T = " << sizeof(T) << std::endl;
      throw std::runtime_error("[factor_subtree] factor_subtree NOT implemented for working precision");
   }

   template void factor_subtree<float>(
         void *akeep, void *fkeep, int p, float *aval, float *scaling, 
         void **child_contrib, 
         struct spral::ssids::cpu::cpu_factor_options *options,
         spral::ssids::cpu::ThreadStats *stats);
   
   // template<>
   // void factor_subtree<float>(
   //       void *akeep,
   //       void *fkeep,
   //       int p,
   //       float *aval, 
   //       void **child_contrib, 
   //       struct spral::ssids::cpu::cpu_factor_options *options,
   //       spral::ssids::cpu::ThreadStats *stats) {
   //    throw std::runtime_error("[factor_subtree] factor_subtree NOT implemented for working precision");
   // }

   template<>
   void factor_subtree<double>(
         void *akeep, void *fkeep, int p,
         double *aval,
         double *scaling, 
         void **child_contrib, 
         struct spral::ssids::cpu::cpu_factor_options *options,
         spral::ssids::cpu::ThreadStats *stats) {

      spldlt_factor_subtree_c(akeep, fkeep, p, aval, scaling, child_contrib, options, stats);
   }
}
