#pragma once

#include "assemble.hxx"

#include <starpu.h>

namespace spldlt { namespace starpu {

   ////////////////////////////////////////////////////////////////////////////////
   // Assemble node contribution blocks

   template <typename T, typename PoolAlloc>
   void assemble_contrib_cpu_func(void *buffers[], void *cl_arg) {

      NumericFront<T, PoolAlloc> *node;
      void** child_contrib;
      
      // printf("[assemble_contrib_cpu_func]\n");
      
      starpu_codelet_unpack_args(
            cl_arg, &node, &child_contrib);
      
      assemble_contrib(*node, child_contrib);         
   }

   // assemble_contrib StarPU codelet
   extern struct starpu_codelet cl_assemble_contrib;

   template <typename T, typename PoolAlloc>
   void insert_assemble_contrib(
         starpu_data_handle_t node_hdl, // Node's symbolic handle
         // starpu_data_handle_t contrib_hdl, // Node's contribution blocks symbolic handle
         starpu_data_handle_t *cnode_hdls, int nhdl, // Children node's symbolic handles
         NumericFront<T, PoolAlloc> *node,
         void** child_contrib
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
      ret = starpu_task_insert(&cl_assemble_contrib,
                               STARPU_DATA_MODE_ARRAY, descrs, nh,
                               // STARPU_RW, node_hdl,
                               // STARPU_RW, contrib_hdl,
                               // STARPU_TAG, tag2, 
                               STARPU_VALUE, &node, sizeof(NumericFront<T, PoolAlloc>*),
                               STARPU_VALUE, &child_contrib, sizeof(void**),
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

   template <typename T, typename PoolAlloc>
   void codelet_init_assemble() {

      // assemble_contrib StarPU codelet
      starpu_codelet_init(&cl_assemble_contrib);
      cl_assemble_contrib.where = STARPU_CPU;
      cl_assemble_contrib.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_assemble_contrib.name = "ASSEMBLE_CONTRIB";
      cl_assemble_contrib.cpu_funcs[0] = assemble_contrib_cpu_func<T, PoolAlloc>;
   }

}} // end of namespaces spldlt::starpu
