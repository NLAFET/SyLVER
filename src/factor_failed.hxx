/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "sylver_ciface.hxx"
#include "kernels/factor_indef.hxx"
#include "tasks/form_contrib.hxx"

#if defined(SPLDLT_USE_STARPU)
#include <starpu.h>
#endif

#include <assert.h>

// SSIDS
#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/Workspace.hxx"
#include "ssids/cpu/ThreadStats.hxx"

namespace spldlt {

   /// @brief Factor the failed pivots in a frontal matrix
   template <typename T, typename PoolAlloc>
   void factor_front_indef_failed(
         NumericFront<T, PoolAlloc>& node,
         std::vector<spral::ssids::cpu::Workspace>& workspaces, 
         // spral::ssids::cpu::Workspace& work,
         sylver::options_t& options,
         sylver::inform_t& stats) {

      int m = node.get_nrow();
      int n = node.get_ncol();
      size_t ldl = node.get_ldl();
      T *lcol = node.lcol;
      T *d = &lcol[n*ldl];
      int *perm = node.perm;
         
      int nelim = 0;
      // bool formcb = false;

      // Setup workspace
      int workerid = 0;
#if defined(SPLDLT_USE_STARPU)
      workerid = starpu_worker_get_id();
      if (workerid < 0) // current context is not a StarPU worker
         workerid = 0;
#endif
      spral::ssids::cpu::Workspace& work = workspaces[workerid];
      
      // Record the number of columns eliminated during the first pass
      node.nelim1 = node.nelim; 

      // Try to eliminate the columns uneliminated at first pass
      if (node.nelim < n) {
         // printf("[factor_front_indef_failed] nelim = %d\n", node.nelim);
         nelim = node.nelim;
         if(options.pivot_method!=sylver::PivotMethod::tpp)
            stats.not_first_pass += n-nelim;

         // Use TPP factor to eliminate the remaining columns in the following cases:
         // 1) options.pivot_method is set to tpp;
         // 2) We are at a root node;
         // 3) options.failed_pivot_method is set to tpp.
         if(m==n ||
            options.pivot_method==sylver::PivotMethod::tpp ||
            options.failed_pivot_method==sylver::FailedPivotMethod::tpp
               ) {

            T *ld = work.get_ptr<T>(m-nelim);
            // auto start = std::chrono::high_resolution_clock::now();
            try {
               node.nelim += ldlt_tpp_factor(
                     m-nelim, n-nelim, &perm[nelim], &lcol[nelim*(ldl+1)], ldl, 
                     &d[2*nelim], ld, m-nelim, options.action, options.u, options.small, 
                     nelim, &lcol[nelim], ldl);
            } catch (spral::ssids::cpu::SingularError const&) {
               stats.flag = sylver::Flag::ERROR_SINGULAR;
            }
            // auto end = std::chrono::high_resolution_clock::now();
            // long t_factor = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
            // long t_form_contrib = 0;
            if(
                  (m-n>0) && // We're not at a root node
                  (node.nelim > nelim) // We've eliminated columns at second pass
                  ) {

#if defined(SPLDLT_USE_GPU)
               // printf("[factor_front_indef_failed] workerid = %d\n", workerid);
               // printf("[factor_front_indef_failed] form contrib, nodeidx = %d\n", node.symb.idx);
               // formcb = true;
               // Compute contribution blocks
               // auto start = std::chrono::high_resolution_clock::now();
               form_contrib_notask(node, work, nelim, node.nelim-1);
               // form_contrib_task(node, workspaces, nelim, node.nelim-1);
#if defined(SYLVER_HAVE_STARPU)
               starpu_fxt_trace_user_event(node.symb().idx); // DEBUG
#endif
               // auto end = std::chrono::high_resolution_clock::now();
               // t_form_contrib = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();

               // int nodeidx = node.symb.idx;
               // starpu_tag_t tag_factor_failed = (starpu_tag_t) (3*nodeidx+2);
               // // starpu_tag_notify_from_apps(tag_factor_failed);
               // starpu_tag_remove(tag_factor_failed);

#else
               form_contrib_notask(node, work, nelim, node.nelim-1);
#endif
            }

            // printf("[factor_front_indef_failed] m = %d, n = %d, LDLT TPP = %e, form contrib = %e\n", m-nelim, n-nelim, 1e-9*t_factor, 1e-9*t_form_contrib);

            if(options.pivot_method==sylver::PivotMethod::tpp) {
               stats.not_first_pass += n - node.nelim;
            } else {
               // printf("[factor_front_indef_failed] Not second pass = %d\n", n-node.nelim);
               stats.not_second_pass += n - node.nelim;
            }
         }

      }
      // Update number of delayed columns
      node.ndelay_out(n - node.nelim);
      stats.num_delay += node.ndelay_out();
      // if (node.ndelay_out() > 0)
         // printf("[factor_front_indef_failed] nodeidx = %d, ndelay_out = %d\n",
                // node.symb.idx, node.ndelay_out);
         
      if (node.nelim == 0) {
// #if defined(SPLDLT_USE_GPU)
         // printf("[factor_front_indef_failed] zero contrib blocks\n");
// #endif
         node.zero_contrib_blocks();
         // zero_contrib_blocks_task(node);
      }

#if defined(SPLDLT_USE_STARPU)
// #if defined(SPLDLT_USE_GPU)
      // if(!formcb) {
      int nodeidx = node.symb().idx;
      // printf("[factor_front_indef_failed] nodeidx = %d\n", nodeidx);
      starpu_tag_t tag_factor_failed = (starpu_tag_t) (3*nodeidx+2);
      starpu_tag_notify_from_apps(tag_factor_failed);
      // starpu_tag_remove(tag_factor_failed);
      // }
// #endif
#endif
  
   }

} // end of namespace spldlt
