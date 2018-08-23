#pragma once

#include "assemble.hxx"

#include <starpu.h>

namespace spldlt { namespace starpu {
      
      ////////////////////////////////////////////////////////////////////////////////
      // fini child nodes
      template <typename T, typename PoolAlloc>
      void fini_cnodes_cpu_func(void *buffers[], void *cl_arg) {

         NumericFront<T, PoolAlloc> *node = nullptr;

         starpu_codelet_unpack_args(cl_arg, &node);
      
         // printf("[fini_cnodes_cpu_func]\n");

         fini_cnodes(*node);
      }

      // fini_cnodes StarPU codelet
      extern struct starpu_codelet cl_fini_cnodes;

      template <typename T, typename PoolAlloc>
      void insert_fini_cnodes(
            starpu_data_handle_t node_hdl,
            NumericFront<T, PoolAlloc> *node) {

         int ret;
         ret = starpu_task_insert(&cl_fini_cnodes,
                                  STARPU_RW, node_hdl,
                                  STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
                                  0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

      }

      ////////////////////////////////////////////////////////////////////////////////
      // Assemble contribution blocks

      template <typename T, typename PoolAlloc>
      void assemble_contrib_cpu_func(void *buffers[], void *cl_arg) {

         NumericFront<T, PoolAlloc> *node = nullptr;
         void** child_contrib;
         std::vector<spral::ssids::cpu::Workspace> *workspaces;
      
         // printf("[assemble_contrib_cpu_func]\n");
      
         starpu_codelet_unpack_args(
               cl_arg, &node, &child_contrib, &workspaces);
      
         assemble_contrib(*node, child_contrib, *workspaces);
      }

      // assemble_contrib StarPU codelet
      extern struct starpu_codelet cl_assemble_contrib;
      
      template <typename T, typename PoolAlloc>
      void insert_assemble_contrib(
            starpu_data_handle_t node_hdl, // Node's symbolic handle
            // starpu_data_handle_t contrib_hdl, // Node's contribution blocks symbolic handle
            starpu_data_handle_t *cnode_hdls, int nhdl, // Children node's symbolic handles
            NumericFront<T, PoolAlloc> *node,
            void** child_contrib,
            std::vector<spral::ssids::cpu::Workspace> *workspaces
            ) {

         // printf("[insert_assemble_contrib]\n");

         struct starpu_data_descr *descrs = new starpu_data_descr[nhdl+1];

         int nh = 0;
         descrs[nh].handle = node_hdl; descrs[nh].mode = STARPU_RW;
         nh++;

         for (int i=0; i<nhdl; i++) {
            descrs[nh].handle = cnode_hdls[i]; descrs[nh].mode = STARPU_R;
            nh++;
         }
        
         // starpu_tag_t tag1 = (starpu_tag_t) (2*node->symb.idx);
         // starpu_tag_t tag2 = (starpu_tag_t) (2*node->symb.idx+1);
         // starpu_tag_declare_deps(tag2, 1, tag1);

         int ret;
         ret = starpu_task_insert(
               &cl_assemble_contrib,
               STARPU_DATA_MODE_ARRAY, descrs, nh,
               // STARPU_RW, node_hdl,
               // STARPU_RW, contrib_hdl,
               // STARPU_TAG, tag2, 
               STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
               STARPU_VALUE, &child_contrib, sizeof(void**),
               STARPU_VALUE, &workspaces, sizeof(std::vector<spral::ssids::cpu::Workspace>*),
               0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

         // test/debug
         // struct starpu_task *taskA = starpu_task_create();
         // taskA->cl = NULL;
         // taskA->use_tag = 1;
         // taskA->tag_id = tagA;
         // ret = starpu_task_submit(taskA); 
         // STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
         // end test/debug

         delete[] descrs;
      }

      ////////////////////////////////////////////////////////////
      // Assemble fully summed

      template <typename T, typename PoolAlloc>
      void assemble_cpu_func(void *buffers[], void *cl_arg) {

         int n;
         NumericFront<T, PoolAlloc> *node;
         void** child_contrib;
         PoolAlloc *pool_alloc;

         starpu_codelet_unpack_args(
               cl_arg, &n, &node, &child_contrib, &pool_alloc);

         assemble(n, *node, child_contrib, *pool_alloc);         
      }      

      // assemble StarPU codelet
      extern struct starpu_codelet cl_assemble;

      template <typename T, typename PoolAlloc>
      void insert_assemble(
            starpu_data_handle_t node_hdl, // Node's symbolic handle
            starpu_data_handle_t *cnode_hdls, int nhdl, // Children node's symbolic handles
            int n,
            NumericFront<T, PoolAlloc> *node,
            void** child_contrib, 
            PoolAlloc *pool_alloc
            ) {

         struct starpu_data_descr *descrs = new starpu_data_descr[nhdl+1];

         int nh = 0;
         descrs[nh].handle = node_hdl; descrs[nh].mode = STARPU_RW;
         nh++;

         for (int i=0; i<nhdl; i++) {
            descrs[nh].handle = cnode_hdls[i]; descrs[nh].mode = STARPU_R;
            nh++;
         }
         // printf("[insert_assemble] node = %d, nh = %d\n", node->symb.idx+1, nh);

         int ret;
         ret = starpu_task_insert(&cl_assemble,
                                  STARPU_DATA_MODE_ARRAY, descrs, nh,
                                  STARPU_VALUE, &n, sizeof(int),
                                  STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
                                  STARPU_VALUE, &child_contrib, sizeof(void**),
                                  STARPU_VALUE, &pool_alloc, sizeof(PoolAlloc*),
                                  0);
         STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

         delete[] descrs;
      }

      template <typename T, typename PoolAlloc>
      void codelet_init_assemble() {

         // assemble StarPU codelet
         starpu_codelet_init(&cl_assemble);
         cl_assemble.where = STARPU_CPU;
         cl_assemble.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_assemble.name = "Assemble";
         cl_assemble.cpu_funcs[0] = assemble_cpu_func<T, PoolAlloc>;

         // assemble_contrib StarPU codelet
         starpu_codelet_init(&cl_assemble_contrib);
         cl_assemble_contrib.where = STARPU_CPU;
         cl_assemble_contrib.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_assemble_contrib.name = "ASSEMBLE_CONTRIB";
         cl_assemble_contrib.cpu_funcs[0] = assemble_contrib_cpu_func<T, PoolAlloc>;

         // fini_cnodes StarPU codelet
         starpu_codelet_init(&cl_fini_cnodes);
         cl_fini_cnodes.where = STARPU_CPU;
         cl_fini_cnodes.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_fini_cnodes.name = "FINI_CNODES";
         cl_fini_cnodes.cpu_funcs[0] = fini_cnodes_cpu_func<T, PoolAlloc>;

      }

}} // end of namespaces spldlt::starpu
