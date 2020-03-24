/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

namespace sylver {
namespace spldlt {

   // TODO Use NumericFront as base class
   // TODO Init nelim to 0
   template<typename T, typename TFac = T>
   class NumericFrontGPU {
   public:
      NumericFrontGPU(::spldlt::SymbolicFront& symb)
         : symb_(symb), lcol(nullptr), dev_lcol(nullptr), ndelay_in(0), ldl(0),
           nelim(symb.ncol)
      {}

      /// @brief Return the number of rows in the node
      inline int get_nrow() const {
         return symb_.nrow + ndelay_in;
      }

      /// @brief Return the number of columns in the node
      inline int get_ncol() const {
         return symb_.ncol + ndelay_in;
      }

      /// @brief Return leading dimension of lcol array
      inline int get_ldl() const { return ldl; }

      /// @brief Return leading dimension of lcol array on device
      inline int get_dev_ldl() const { return dev_ldl; }

      /// @brief Return number of cols in cb passed to L part of parent node
      inline int get_npassl() const { return symb_.npassl; }

   public:
      // Factors
      T *lcol; // Pointer to factors on host side
      int ldl; // lcol leading dimension
      TFac *dev_lcol; // Pointer to factors on device
      int dev_ldl; // lcol leading dimension on device
      int ndelay_in; // Number of incoming delays
      int nelim; // Number of eliminated columns
   private:
      /* Symbolic node associate with this one */
      ::spldlt::SymbolicFront& symb_;
   };
   
}} // End of namespace sylver::spldlt
