/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "sylver_ciface.hxx"
#include "factor_indef.hxx"

// StarPU
#include <starpu.h>

namespace sylver {
namespace spldlt {
namespace starpu {

   ////////////////////////////////////////////////////////////////////////////////
   // factor_front_indef

   // CPU kernel
   template <typename NumericFrontType, typename PoolAlloc>
   void factor_front_indef_cpu_func(void *buffers[], void *cl_arg) {
      
      NumericFrontType *node;
      std::vector<spral::ssids::cpu::Workspace> *workspaces;
      PoolAlloc *pool_alloc;
      sylver::options_t *options;
      std::vector<sylver::inform_t> *worker_stats;
      
      starpu_codelet_unpack_args (
            cl_arg, &node, &workspaces, &pool_alloc, &options, 
            &worker_stats);

      factor_front_indef(
            *node, *workspaces, *pool_alloc, *options, *worker_stats);
         
      // int workerid = starpu_worker_get_id();
      // // printf("[factor_front_indef_cpu_func] workerid = %d\n", workerid);
      // factor_front_indef_notask(
      //       *options, *pool_alloc, *node,
      //       (*workspaces)[workerid], (*worker_stats)[workerid]);
   }

   // SarPU codelet
   extern struct starpu_codelet cl_factor_front_indef;
      
   template <typename NumericFrontType, typename PoolAlloc>
   void insert_factor_front_indef(
         starpu_data_handle_t node_hdl,
         NumericFrontType *node,
         std::vector<spral::ssids::cpu::Workspace> *workspaces,
         PoolAlloc *pool_alloc,
         sylver::options_t *options,
         std::vector<sylver::inform_t> *worker_stats
         ) {

      int ret;

      ret = starpu_task_insert(
            &cl_factor_front_indef,
            STARPU_RW, node_hdl,
            STARPU_VALUE, &node, sizeof(NumericFrontType*),
            STARPU_VALUE, &workspaces, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
            STARPU_VALUE, &pool_alloc, sizeof(PoolAlloc*),
            STARPU_VALUE, &options, sizeof(sylver::options_t*),
            STARPU_VALUE, &worker_stats, sizeof(std::vector<sylver::inform_t> *),
            0);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
   }
      
   ////////////////////////////////////////////////////////////////////////////////

   template <typename NumericFrontType, typename PoolAlloc>
   void codelet_init_factor_indef() {

      // Initialize factor_front_indef StarPU codelet
      starpu_codelet_init(&cl_factor_front_indef);
      cl_factor_front_indef.where = STARPU_CPU;
      cl_factor_front_indef.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_factor_front_indef.name = "FactorFront";
      cl_factor_front_indef.cpu_funcs[0] = factor_front_indef_cpu_func<NumericFrontType, PoolAlloc>;

   }

}}} // End namespaces sylver::spldlt::starpu
