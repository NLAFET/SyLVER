/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

#include "sylver_ciface.hxx"
#include "ssids/cpu/ThreadStats.hxx"
#include "ssids/cpu/cpu_iface.hxx"

#include <algorithm>

namespace sylver {

   void sylver::options_t::copy(spral::ssids::cpu::cpu_factor_options& other) {
      other.print_level = print_level;
      other.action = action;
      other.small = small;
      other.u = u;
      other.multiplier = multiplier;
      other.small_subtree_threshold = small_subtree_threshold;
      other.cpu_block_size = nb;

      // Pivot method
      switch(pivot_method) {
      case PivotMethod::app_aggressive:
         other.pivot_method = spral::ssids::cpu::PivotMethod::app_aggressive;
         break;
      case PivotMethod::app_block:
         other.pivot_method = spral::ssids::cpu::PivotMethod::app_block;
         break;
      case PivotMethod::tpp:
         other.pivot_method = spral::ssids::cpu::PivotMethod::tpp;
         break;
      default:
         other.pivot_method = spral::ssids::cpu::PivotMethod::app_block;
         break;
      }

      // Failed pivot method
      switch(failed_pivot_method) {
      case FailedPivotMethod::tpp:
         other.failed_pivot_method = spral::ssids::cpu::FailedPivotMethod::tpp;
         break;
      case FailedPivotMethod::pass:
         other.failed_pivot_method = spral::ssids::cpu::FailedPivotMethod::pass;
         break;
      default:
         other.failed_pivot_method = spral::ssids::cpu::FailedPivotMethod::tpp;
         break;
      }
   }
   
   inform_t& inform_t::operator+=(inform_t const& other) {
      flag = (flag<0 || other.flag<0) ? std::min(flag, other.flag) // error
                   : std::max(flag, other.flag);// warning/pass
      num_delay += other.num_delay;
      num_neg += other.num_neg;
      num_two += other.num_two;
      num_zero += other.num_zero;
      maxfront = std::max(maxfront, other.maxfront);
      not_first_pass += other.not_first_pass;
      not_second_pass += other.not_second_pass;

      return *this;
   }

   inform_t& inform_t::operator+=(spral::ssids::cpu::ThreadStats const& other) {
      flag = (flag<0 || other.flag<0) ? 
                   static_cast<sylver::Flag>(
                         std::min( // Error
                               static_cast<typename std::underlying_type<sylver::Flag>::type>(flag), 
                               static_cast<typename std::underlying_type<spral::ssids::cpu::Flag>::type>(other.flag)))
                   :
         static_cast<sylver::Flag>(
               std::max( // Warning/pass
                     static_cast<typename std::underlying_type<sylver::Flag>::type >(flag),
                     static_cast<typename std::underlying_type<spral::ssids::cpu::Flag>::type>(other.flag)));
      num_delay += other.num_delay;
      num_neg += other.num_neg;
      num_two += other.num_two;
      num_zero += other.num_zero;
      maxfront = std::max(maxfront, other.maxfront);
      not_first_pass += other.not_first_pass;
      not_second_pass += other.not_second_pass;

      return *this;
   }

} // End of namespace sylver 
