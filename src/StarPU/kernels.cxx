/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez

// SyLVER
#include "StarPU/kernels.hxx"

// STD
#include <vector>
// StarPU
#include <starpu.h>

namespace sylver {
namespace spldlt {
namespace starpu {

   // activate_node StarPU codelet
   struct starpu_codelet cl_activate_node;

   // activate_init_node StarPU codelet
   struct starpu_codelet cl_activate_init_node;

   // init_node StarPU codelet
   struct starpu_codelet cl_init_node;

   // fini_node StarPU codelet
   struct starpu_codelet cl_fini_node;

   ////////////////////////////////////////////////////////////////////////////////   
   // factorize_contrib_block
      
   // factorize_contrib_block StarPU codelet
   struct starpu_codelet cl_factorize_contrib_block;

   void insert_factor_block(
         int k,
         starpu_data_handle_t bc_hdl,
         starpu_data_handle_t contrib_hdl,
         starpu_data_handle_t node_hdl, // Symbolic node handle
         int prio,
         std::vector<sylver::inform_t> *worker_stats) {

      int ret;

      if (node_hdl) {
         ret = starpu_insert_task(
               &cl_factorize_contrib_block,
               STARPU_RW, bc_hdl,
               STARPU_RW, contrib_hdl,
               STARPU_R, node_hdl,
               STARPU_VALUE, &k, sizeof(int),
               STARPU_VALUE, &worker_stats, sizeof(std::vector<sylver::inform_t> *),
               STARPU_PRIORITY, prio,
               0);
      }
      else {
         ret = starpu_insert_task(
               &cl_factorize_contrib_block,
               STARPU_RW, bc_hdl,
               STARPU_RW, contrib_hdl,
               STARPU_VALUE, &k, sizeof(int),
               STARPU_VALUE, &worker_stats, sizeof(std::vector<sylver::inform_t> *),
               STARPU_PRIORITY, prio,
               0);
      }

      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
   }

   ////////////////////////////////////////////////////////////////////////////////   
   // factorize_block

   // factorize_block StarPU codelet
   struct starpu_codelet cl_factorize_block;      

   void insert_factor_block(
         starpu_data_handle_t bc_hdl,
         starpu_data_handle_t node_hdl, // Symbolic node handle
         int prio,
         std::vector<sylver::inform_t> *worker_stats) {

      int ret;

      if (node_hdl) {
         ret = starpu_insert_task(
               &cl_factorize_block,
               STARPU_RW, bc_hdl,
               STARPU_R, node_hdl,
               STARPU_VALUE, &worker_stats, sizeof(std::vector<sylver::inform_t> *),
               STARPU_PRIORITY, prio,
               0);
      }
      else {
         ret = starpu_insert_task(
               &cl_factorize_block,
               STARPU_RW, bc_hdl,
               STARPU_VALUE, &worker_stats, sizeof(std::vector<sylver::inform_t> *),
               STARPU_PRIORITY, prio,
               0);
      }

      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
   }

   ////////////////////////////////////////////////////////////////////////////////   
   // solve_block

   // solve_block StarPU codelet
   struct starpu_codelet cl_solve_block;

   void insert_solve_block(
         starpu_data_handle_t bc_kk_hdl, /* diag block handle */
         starpu_data_handle_t bc_ik_hdl, /* sub diag block handle */
         starpu_data_handle_t node_hdl,
         int prio
         ) {

      int ret;

      ret = starpu_insert_task(
            &cl_solve_block,
            STARPU_R, bc_kk_hdl,
            STARPU_RW, bc_ik_hdl,
            STARPU_R, node_hdl,
            STARPU_PRIORITY, prio,
            0);

      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
   }

   ////////////////////////////////////////////////////////////////////////////////   
   // solve_contrib_block

   // solve_contrib_block StarPU codelet
   struct starpu_codelet cl_solve_contrib_block;

   void insert_solve_block(
         int k, int nb,
         starpu_data_handle_t bc_kk_hdl, // Diag block handle
         starpu_data_handle_t bc_ik_hdl, // Sub diag block handle
         starpu_data_handle_t contrib_hdl, // Contrib block handle
         starpu_data_handle_t node_hdl,
         int prio) {

      int ret;

      ret = starpu_insert_task(
            &cl_solve_contrib_block,
            STARPU_R, bc_kk_hdl,
            STARPU_RW, bc_ik_hdl,
            STARPU_RW, contrib_hdl,
            STARPU_R, node_hdl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &k, sizeof(int),
            STARPU_VALUE, &nb, sizeof(int),
            0);

      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
   }      

   ////////////////////////////////////////////////////////////////////////////////   
   // update_block

   // update_block codelet
   struct starpu_codelet cl_update_block;      

   void insert_update_block(
         starpu_data_handle_t bc_ij_hdl, /* A_ij block handle */
         starpu_data_handle_t bc_ik_hdl, /* A_ik block handle */
         starpu_data_handle_t bc_jk_hdl, /* A_jk block handle */
         starpu_data_handle_t node_hdl,
         int prio) {

      int ret;

      ret = starpu_insert_task(
            &cl_update_block,
            STARPU_RW, bc_ij_hdl,
            STARPU_R, bc_ik_hdl,
            STARPU_R, bc_jk_hdl,
            STARPU_R, node_hdl,
            STARPU_PRIORITY, prio,
            0);

      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");         
   }

   ////////////////////////////////////////////////////////////////////////////////   
   // update_contrib_block

   // update_contrib_block StarPU codelet
   struct starpu_codelet cl_update_contrib_block;

   void insert_update_block(
         int k, int nb,
         starpu_data_handle_t bc_ij_hdl, /* A_ij block handle */
         starpu_data_handle_t bc_ik_hdl, /* A_ik block handle */
         starpu_data_handle_t bc_jk_hdl, /* A_jk block handle */
         starpu_data_handle_t contrib_hdl, /* A_ij block handle */
         starpu_data_handle_t node_hdl,
         int prio) {

      int ret;

      ret = starpu_insert_task(
            &cl_update_contrib_block,
            STARPU_RW, bc_ij_hdl,
            STARPU_R, bc_ik_hdl,
            STARPU_R, bc_jk_hdl,
            STARPU_RW, contrib_hdl,
            STARPU_R, node_hdl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &k, sizeof(int),
            STARPU_VALUE, &nb, sizeof(int),
            0);

      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");         
   }

   // update_contrib StarPU codelet
   struct starpu_codelet cl_update_contrib;      

   void insert_update_contrib(
         int k,
         starpu_data_handle_t bc_ij_hdl, /* A_ij block handle */
         starpu_data_handle_t bc_ik_hdl, /* A_ik block handle */
         starpu_data_handle_t bc_jk_hdl, /* A_jk block handle */
         starpu_data_handle_t node_hdl,
         int prio) {

      int ret;

      ret = starpu_insert_task(
            &cl_update_contrib,
            STARPU_RW, bc_ij_hdl,
            STARPU_R, bc_ik_hdl,
            STARPU_R, bc_jk_hdl,
            STARPU_R, node_hdl,
            STARPU_PRIORITY, prio,
            STARPU_VALUE, &k, sizeof(int),
            0);

      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");         
   }

   // factor_subtree StarPU codelet
   struct starpu_codelet cl_factor_subtree;

   // factor_subtree_gpu StarPU codelet
   struct starpu_codelet cl_factor_subtree_gpu;

   // subtree_assemble StarPU codelet
   struct starpu_codelet cl_subtree_assemble;

   // subtree_assemble_block StarPU codelet
   struct starpu_codelet cl_subtree_assemble_block;

   // subtree_assemble_contrib StarPU codelet
   struct starpu_codelet cl_subtree_assemble_contrib;

   // subtree_assemble_contrib_block StarPU codelet
   struct starpu_codelet cl_subtree_assemble_contrib_block;

   // assemble_block StarPU codelet
   struct starpu_codelet cl_assemble_block;

   // assemble_contrib_block StarPU codelet
   struct starpu_codelet cl_assemble_contrib_block;

}}}  // End of namespaces sylver::spldlt::starpu
