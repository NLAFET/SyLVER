/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

#include "ssids/cpu/ThreadStats.hxx"
#include "ssids/cpu/cpu_iface.hxx"

namespace sylver {
   
   enum struct PivotMethod : int {
      app_aggressive = 1, 
      app_block      = 2, 
      tpp            = 3
   };

   enum struct FailedPivotMethod : int {
      tpp  = 1,
      pass = 2
   };

   enum struct CPUTopology : int
      {
       automatic = 1,
       flat = 2,
       numa = 3
      };
   
   /// @brief Interoperable subset of sylver_options
   struct options_c {
      int print_level;
      bool action;
      double small;
      double u;
      double multiplier;
      long small_subtree_threshold;
      int nb; // Block size
      sylver::PivotMethod pivot_method;
      sylver::FailedPivotMethod failed_pivot_method;
      sylver::CPUTopology cpu_topology;
      
      /// @brief Copy structure into a
      /// spral::ssids::cpu::cpu_factor_options structure
      void copy(spral::ssids::cpu::cpu_factor_options& other);
   };
   using options_t = struct options_c;
   
   // @brief SyLVER error/warning flags.
   //
   // Must match Fortran definitions in src/sylver_datatypes_mod.F90
   enum Flag : int {
      SUCCESS                 = 0,

      ERROR_SINGULAR          = -5,
      ERROR_NOT_POS_DEF       = -6,
      ERROR_ALLOCATION        = -50,
      ERROR_CUDA_UNKNOWN      = -51,
      ERROR_CUBLAS_UNKNOWN    = -52,
      ERROR_UNKNOWN           = -99,
      
      WARNING_FACT_SINGULAR   = 7
   };

   struct inform_c {
      Flag flag = Flag::SUCCESS; ///< Error flag for thread
      int num_delay = 0;   ///< Number of delays
      int num_neg = 0;     ///< Number of negative pivots
      int num_two = 0;     ///< Number of 2x2 pivots
      int num_zero = 0;    ///< Number of zero pivots
      int maxfront = 0;    ///< Maximum front size
      int not_first_pass = 0;    ///< Number of pivots not eliminated in APP
      int not_second_pass = 0;   ///< Number of pivots not eliminated in APP or TPP

      inform_c& operator+=(inform_c const& other);
      inform_c& operator+=(spral::ssids::cpu::ThreadStats const& other);

   };
   using inform_t = struct inform_c;

   // ///
   // /// @brief Exception class for options.action = false and singular matrix.
   // ///
   // class SingularError: public std::runtime_error {
   // public:
   //    SingularError(int col)
   //       : std::runtime_error("Matrix is singular"), col(col)
   //    {}
   
   //    int const col;
   // };
}
