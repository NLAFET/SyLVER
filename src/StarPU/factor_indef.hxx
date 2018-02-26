#pragma once

#include "factor_indef.hxx"

#include <starpu.h>

namespace spldlt { namespace starpu {

      ////////////////////////////////////////////////////////////////////////////////
      // factor_front_indef

      // CPU kernel
      template <typename T, typename PoolAlloc>
      void factor_front_indef_cpu_func(void *buffers[], void *cl_arg) {
      
         NumericFront<T, PoolAlloc> *node;
         std::vector<spral::ssids::cpu::Workspace> *workspaces;
         PoolAlloc *pool_alloc;
         struct cpu_factor_options *options;
      
         starpu_codelet_unpack_args (
               cl_arg, &node, &workspaces, &pool_alloc, &options);

         factor_front_indef(
               node->symb, *node, *workspaces, *pool_alloc, *options);
      }      

      // SarPU codelet
      extern struct starpu_codelet cl_factor_front_indef;
      
      template <typename T, typename PoolAlloc>
      void insert_factor_front_indef(
            starpu_data_handle_t node_hdl,
            NumericFront<T, PoolAlloc> *node,
            std::vector<spral::ssids::cpu::Workspace> *workspaces,
            PoolAlloc *pool_alloc,
            struct cpu_factor_options *options
            ) {

         int ret;

         ret = starpu_task_insert(
               &cl_factor_front_indef,
               STARPU_RW, node_hdl,
               STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
               STARPU_VALUE, &workspaces, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
               STARPU_VALUE, &pool_alloc, sizeof(PoolAlloc*),
               STARPU_VALUE, &options, sizeof(struct cpu_factor_options *),
               0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");         
         
      }      

      ////////////////////////////////////////////////////////////////////////////////
      // factor_front_indef_nocontrib

      // CPU kernel
      template <typename T, typename PoolAlloc>
      void factor_front_indef_nocontrib_cpu_func(void *buffers[], void *cl_arg) {
      
         NumericFront<T, PoolAlloc> *node;
         std::vector<spral::ssids::cpu::Workspace> *workspaces;
         PoolAlloc *pool_alloc;
         struct cpu_factor_options *options;
      
         starpu_codelet_unpack_args (
               cl_arg, &node, &workspaces, &pool_alloc, &options);

         factor_front_indef_nocontrib(
               node->symb, *node, *workspaces, *pool_alloc, *options);

      }

      // SarPU codelet
      extern struct starpu_codelet cl_factor_front_indef_nocontrib;

      template <typename T, typename PoolAlloc>
      void insert_factor_front_indef_nocontrib(
            starpu_data_handle_t node_hdl,
            NumericFront<T, PoolAlloc> *node,
            std::vector<spral::ssids::cpu::Workspace> *workspaces,
            PoolAlloc *pool_alloc,
            struct cpu_factor_options *options
            ) {

         int ret;

         ret = starpu_task_insert(
               &cl_factor_front_indef_nocontrib,
               STARPU_RW, node_hdl,
               STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
               STARPU_VALUE, &workspaces, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
               STARPU_VALUE, &pool_alloc, sizeof(PoolAlloc*),
               STARPU_VALUE, &options, sizeof(struct cpu_factor_options *),
               0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");         
         
      }

      ////////////////////////////////////////////////////////////////////////////////

      template <typename T, typename PoolAlloc>
      void form_contrib_front_cpu_func(void *buffers[], void *cl_arg) {

         NumericFront<T, PoolAlloc> *node;
         int blksz;

         printf("[form_contrib_front_cpu_func]\n");
         
         starpu_codelet_unpack_args (
               cl_arg, &node, &blksz);

         form_contrib_front(
               node->symb, *node, blksz);
         
      }
      
      // SarPU codelet
      extern struct starpu_codelet cl_form_contrib_front;

      template <typename T, typename PoolAlloc>
      void insert_form_contrib_front(
            starpu_data_handle_t node_hdl,
            // starpu_data_handle_t col_hdl,
            NumericFront<T, PoolAlloc> *node,
            int blksz) {

         int ret;

         ret = starpu_task_insert(
               &cl_form_contrib_front,
               STARPU_R, node_hdl,
               // STARPU_R, col_hdl,
               STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
               STARPU_VALUE, &blksz, sizeof(int),
               0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");         

      }
      
      ////////////////////////////////////////////////////////////////////////////////

      template <typename T, typename PoolAlloc>
      void codelet_init_factor_indef() {

         // Initialize factor_front_indef_nocontrib StarPU codelet
         starpu_codelet_init(&cl_factor_front_indef_nocontrib);
         cl_factor_front_indef_nocontrib.where = STARPU_CPU;
         cl_factor_front_indef_nocontrib.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_factor_front_indef_nocontrib.name = "FACTOR_FRONT_NOCONTRIB";
         cl_factor_front_indef_nocontrib.cpu_funcs[0] = factor_front_indef_nocontrib_cpu_func<T, PoolAlloc>;

         // Initialize factor_front_indef StarPU codelet
         starpu_codelet_init(&cl_factor_front_indef);
         cl_factor_front_indef.where = STARPU_CPU;
         cl_factor_front_indef.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_factor_front_indef.name = "FACTOR_FRONT_NOCONTRIB";
         cl_factor_front_indef.cpu_funcs[0] = factor_front_indef_cpu_func<T, PoolAlloc>;

         // Initialize form_contrib StarPU codelet
         starpu_codelet_init(&cl_form_contrib_front);
         cl_form_contrib_front.where = STARPU_CPU;
         cl_form_contrib_front.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_form_contrib_front.name = "FORM_CONTRIB_FRONT";
         cl_form_contrib_front.cpu_funcs[0] = form_contrib_front_cpu_func<T, PoolAlloc>;

      }

}} /* namespaces spldlt::starpu  */
