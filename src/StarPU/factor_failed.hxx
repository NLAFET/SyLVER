#pragma once

// SpLDLT
#include "NumericFront.hxx"
#include "factor_failed.hxx"

// StarPU
#include <starpu.h>

// SSIDS
#include "ssids/cpu/Workspace.hxx"
#include "ssids/cpu/ThreadStats.hxx"

namespace spldlt { namespace starpu {

      ////////////////////////////////////////
      // factor_front_indef_failed

      // CPU kernel
      template <typename T, typename PoolAlloc>
      void factor_front_indef_failed_cpu_func(void *buffers[], void *cl_arg) {
         
         NumericFront<T, PoolAlloc> *node = nullptr;
         std::vector<spral::ssids::cpu::Workspace> *workspaces = nullptr;
         struct cpu_factor_options *options = nullptr;
         std::vector<ThreadStats> *worker_stats = nullptr;

         starpu_codelet_unpack_args(
               cl_arg, &node, &workspaces, &options, &worker_stats);

         int workerid = starpu_worker_get_id();
         // spral::ssids::cpu::Workspace& work = (*workspaces)[workerid];
         spral::ssids::cpu::ThreadStats& stats = (*worker_stats)[workerid];

         factor_front_indef_failed(*node, *workspaces, *options, stats);

         // int nodeidx = node->symb.idx;
         // starpu_tag_t tag_factor_failed = (starpu_tag_t) (3*nodeidx+2);
         // starpu_tag_notify_from_apps(tag_factor_failed);
         // starpu_tag_remove(tag_factor_failed);            
      }

      // SarPU codelet
      extern struct starpu_codelet cl_factor_front_indef_failed;

      template <typename T, typename PoolAlloc>
      void insert_factor_front_indef_failed(
            starpu_data_handle_t col_hdl,
            starpu_data_handle_t contrib_hdl,
            starpu_data_handle_t *hdls, int nhdl,
            NumericFront<T, PoolAlloc> *node,
            std::vector<spral::ssids::cpu::Workspace> *workspaces,
            struct cpu_factor_options *options,
            std::vector<ThreadStats> *worker_stats
            ) {

         int ret;

         struct starpu_data_descr *descrs = new starpu_data_descr[nhdl+2];

         int nh = 0;
         for (int i=0; i<nhdl; i++) {
            descrs[nh].handle = hdls[i]; descrs[nh].mode = STARPU_R;
            nh++;
         }

         descrs[nh].handle = col_hdl; descrs[nh].mode = STARPU_RW;
         nh++;

         descrs[nh].handle = contrib_hdl; descrs[nh].mode = STARPU_RW;
         nh++;
         
         ret = starpu_insert_task(
               &cl_factor_front_indef_failed,
               // STARPU_RW, col_hdl,
               // STARPU_RW, contrib_hdl,
               STARPU_DATA_MODE_ARRAY, descrs, nh,
               STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
               STARPU_VALUE, &workspaces, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
               STARPU_VALUE, &options, sizeof(struct cpu_factor_options*),
               STARPU_VALUE, &worker_stats, sizeof(std::vector<ThreadStats>*),
               0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
               
         delete[] descrs;

      }

      template<typename T, typename Allocator>
      void codelet_init_factor_failed() {

         ////////////////////////////////////////////////////////////
         // factor_front_indef_failed StarPU codelet

         starpu_codelet_init(&cl_factor_front_indef_failed);
         cl_factor_front_indef_failed.where = STARPU_CPU;
         cl_factor_front_indef_failed.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_factor_front_indef_failed.name = "FactorFrontFailed";
         cl_factor_front_indef_failed.cpu_funcs[0] =
            factor_front_indef_failed_cpu_func<T, Allocator>;

      }

   } // end of namespaces starpu
} // end of namespaces spldlt
