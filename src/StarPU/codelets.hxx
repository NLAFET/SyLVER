/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SpLDLT
#include "StarPU/codelets_posdef.hxx"
#include "StarPU/kernels_indef.hxx"
#include "StarPU/factor_indef.hxx"
#include "StarPU/factor_failed.hxx"

namespace sylver {
namespace spldlt {
namespace starpu {

   template <typename T, int iblksz, typename Backup, typename Allocator>
   void codelet_init_indef() {
         
      // printf("[codelet_init_indef]\n");

      typedef typename std::allocator_traits<Allocator>::template rebind_alloc<int> IntAlloc;
      
      ////////////////////////////////////////////////////////////
      // factor_block_app StarPU codelet

      starpu_codelet_init(&cl_factor_block_app);
      cl_factor_block_app.where = STARPU_CPU;
      cl_factor_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_factor_block_app.name = "FactorBlockAPP";
      cl_factor_block_app.cpu_funcs[0] = 
         factor_block_app_cpu_func<T, Backup, IntAlloc, Allocator>;

      // printf("[codelet_init_indef] %s\n", cl_factor_block_app.name);

      ////////////////////////////////////////////////////////////
      // applyN_block_app StarPU codelet

      starpu_codelet_init(&cl_applyN_block_app);
      cl_applyN_block_app.where = STARPU_CPU;
      cl_applyN_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_applyN_block_app.name = "ApplyNBlockAPP";
      cl_applyN_block_app.cpu_funcs[0] = 
         applyN_block_app_cpu_func<T, iblksz, Backup, IntAlloc>;

      ////////////////////////////////////////////////////////////
      // applyT_block_app StarPU codelet

      starpu_codelet_init(&cl_applyT_block_app);
      cl_applyT_block_app.where = STARPU_CPU;
      cl_applyT_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_applyT_block_app.name = "ApplyTBlockAPP";
      cl_applyT_block_app.cpu_funcs[0] = 
         applyT_block_app_cpu_func<T, iblksz, Backup, IntAlloc>;

      ////////////////////////////////////////////////////////////
      // updateN_block_app StarPU codelet

      starpu_codelet_init(&cl_updateN_block_app);
#if defined(SPLDLT_USE_GPU)
      // cl_updateN_block_app.where = STARPU_CPU; // DEBUG
      // cl_updateN_block_app.where = STARPU_CUDA; // DEBUG
      cl_updateN_block_app.where = STARPU_CPU | STARPU_CUDA;
#else
      cl_updateN_block_app.where = STARPU_CPU;
#endif
      cl_updateN_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_updateN_block_app.name = "UpdateNBlockAPP";
      cl_updateN_block_app.cpu_funcs[0] = updateN_block_app_cpu_func<T, iblksz, Backup, IntAlloc>;
#if defined(SPLDLT_USE_GPU)
      cl_updateN_block_app.cuda_funcs[0] =
         sylver::spldlt::starpu::updateN_block_app_cuda_func
         <T, iblksz, Backup, IntAlloc>;
      cl_updateN_block_app.cuda_flags[0] = STARPU_CUDA_ASYNC;
#endif

      ////////////////////////////////////////////////////////////
      // updateT_block_app StarPU codelet

      starpu_codelet_init(&cl_updateT_block_app);
      cl_updateT_block_app.where = STARPU_CPU;
      cl_updateT_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_updateT_block_app.name = "UpdateTBlockAPP";
      cl_updateT_block_app.cpu_funcs[0] = updateT_block_app_cpu_func<T, iblksz, Backup, IntAlloc>;

      // Initialize adjust StarPU codelet
      starpu_codelet_init(&cl_adjust);
      cl_adjust.where = STARPU_CPU;
      cl_adjust.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_adjust.name = "Adjust";
      cl_adjust.cpu_funcs[0] = adjust_cpu_func<T, IntAlloc>;

#if defined(SPLDLT_USE_PROFILING)

      ////////////////////////////////////////////////////////////
      // update_contrib_block_indef StarPU perfmodel

      starpu_perfmodel_init(&update_contrib_block_app_perfmodel);
      update_contrib_block_app_perfmodel.type = STARPU_HISTORY_BASED;
      update_contrib_block_app_perfmodel.symbol = "UpdateContribBlockAPPModel";
      update_contrib_block_app_perfmodel.size_base = update_contrib_block_app_size_base<T, IntAlloc, Allocator>;
      update_contrib_block_app_perfmodel.footprint = update_contrib_block_app_footprint<T, IntAlloc, Allocator>;

#endif

      ////////////////////////////////////////////////////////////
      // update_contrib_block_indef StarPU codelet

      starpu_codelet_init(&cl_update_contrib_block_app);
#if defined(SPLDLT_USE_GPU)
      // cl_update_contrib_block_app.where = STARPU_CPU;
      // cl_update_contrib_block_app.where = STARPU_CUDA; // Debug
      cl_update_contrib_block_app.where = STARPU_CPU | STARPU_CUDA;
#else
      cl_update_contrib_block_app.where = STARPU_CPU;
#endif
      cl_update_contrib_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_update_contrib_block_app.name = "UpdateContribBlockAPP";
#if defined(SPLDLT_USE_GPU)
      cl_update_contrib_block_app.cuda_funcs[0] =
         sylver::spldlt::starpu::update_contrib_block_app_cuda_func
         <T, IntAlloc, Allocator>;
      cl_update_contrib_block_app.cuda_flags[0] = STARPU_CUDA_ASYNC;
#endif
      cl_update_contrib_block_app.cpu_funcs[0]  = update_contrib_block_app_cpu_func<T, IntAlloc, Allocator>;
#if defined(SPLDLT_USE_PROFILING)
      cl_update_contrib_block_app.model = &update_contrib_block_app_perfmodel;
#endif


      ////////////////////////////////////////////////////////////
      // permute failed

      starpu_codelet_init(&cl_permute_failed);
      cl_permute_failed.where = STARPU_CPU;
      cl_permute_failed.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_permute_failed.name = "PermuteFailed";
      cl_permute_failed.cpu_funcs[0] =
         permute_failed_cpu_func<T, IntAlloc, Allocator>;

      ////////////////////////////////////////////////////////////
      // form_contrib StarPU codelet

      starpu_codelet_init(&cl_form_contrib);

      cl_form_contrib.where = STARPU_CPU;
      cl_form_contrib.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_form_contrib.name = "FormContrib";
      cl_form_contrib.cpu_funcs[0] = form_contrib_cpu_func<T, Allocator>;

      ////////////////////////////////////////////////////////////
      // zero_contrib_blocks StarPU codelet

      starpu_codelet_init(&cl_zero_contrib_blocks);
      cl_zero_contrib_blocks.where = STARPU_CPU;
      cl_zero_contrib_blocks.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_zero_contrib_blocks.name = "ZeroContrib";
      cl_zero_contrib_blocks.cpu_funcs[0] = zero_contrib_blocks_cpu_func<T, Allocator>;
         
      // // Initialize factor_sync StarPU codelet
      // starpu_codelet_init(&cl_factor_sync);
      // // cl_factor_sync.where = STARPU_NOWHERE;
      // cl_factor_sync.where = STARPU_CPU;
      // cl_factor_sync.nbuffers = 1;// STARPU_VARIABLE_NBUFFERS;
      // cl_factor_sync.modes[0] = STARPU_RW;
      // // cl_factor_sync.modes[0] = STARPU_R;
      // cl_factor_sync.name = "FACTOR_SYNC";
      // cl_factor_sync.cpu_funcs[0] = factor_sync_cpu_func<T, Allocator>;

      ////////////////////////////////////////////////////////////
      // assemble_contrib_sync StarPU codelet

      starpu_codelet_init(&cl_assemble_contrib_sync);
      // cl_factor_sync.where = STARPU_NOWHERE;
      cl_assemble_contrib_sync.where = STARPU_CPU;
      cl_assemble_contrib_sync.nbuffers = 1;
      cl_assemble_contrib_sync.modes[0] = STARPU_RW;
      // cl_assemble_contrib_sync.modes[0] = STARPU_R;
      cl_assemble_contrib_sync.name = "AssembleContribSync";
      // cl_assemble_contrib_sync.cpu_funcs[0] = assemble_contrib_sync_cpu_func<T, Allocator>;
      cl_assemble_contrib_sync.cpu_funcs[0] = assemble_contrib_sync_cpu_func;

      ////////////////////////////////////////////////////////////
      // factor_sync StarPU codelet

      starpu_codelet_init(&cl_nelim_sync);
      // cl_nelim_sync.where = STARPU_NOWHERE;
      cl_nelim_sync.where = STARPU_CPU;
      cl_nelim_sync.nbuffers = 1;
      // cl_nelim_sync.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_nelim_sync.modes[0] = STARPU_RW;
      // cl_nelim_sync.modes[0] = STARPU_R;
      cl_nelim_sync.name = "NelimSync";
      // cl_nelim_sync.cpu_funcs[0] = nelim_sync_cpu_func<T, Allocator>;
      cl_nelim_sync.cpu_funcs[0] = nelim_sync_cpu_func;

      ////////////////////////////////////////////////////////////
      // Restore failed

      starpu_codelet_init(&cl_restore_failed_block_app);
      cl_restore_failed_block_app.where = STARPU_CPU;
      cl_restore_failed_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_restore_failed_block_app.name = "RestoreFailedBlock";
      cl_restore_failed_block_app.cpu_funcs[0] =
         restore_failed_block_app_cpu_func<T, iblksz, Backup, IntAlloc>;

      ////////////////////////////////////////////////////////////
      // assemble_delays StarPU codelet
      starpu_codelet_init(&cl_assemble_delays);
      cl_assemble_delays.where = STARPU_CPU;
      cl_assemble_delays.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_assemble_delays.name = "AssembleDelays";
      cl_assemble_delays.cpu_funcs[0] = assemble_delays_cpu_func<T, Allocator>;

      ////////////////////////////////////////////////////////////
      // assemble_delays_subtree StarPU codelet
      starpu_codelet_init(&cl_assemble_delays_subtree);
      cl_assemble_delays_subtree.where = STARPU_CPU;
      cl_assemble_delays_subtree.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_assemble_delays_subtree.name = "AssembleDelaysSubtree";
      cl_assemble_delays_subtree.cpu_funcs[0] =
         assemble_delays_subtree_cpu_func<T, Allocator>;
   }

   
   ////////////////////////////////////////////////////////////
      
   template <typename T,
             int iblksz,
             typename Backup,
             typename FactorAlloc,
             typename PoolAlloc>
   void codelets_init(bool posdef) {
         
      sylver::spldlt::starpu::codelets_init_posdef<T, FactorAlloc, PoolAlloc>();

      if (!posdef) {
         codelet_init_indef<T, iblksz, Backup, PoolAlloc>();            
         codelet_init_factor_indef<T, PoolAlloc>();
         sylver::spldlt::starpu::codelet_init_factor_failed<T, PoolAlloc>();
      }

   }
      
}}} // End of namespaces sylver::spldlt::starpu
