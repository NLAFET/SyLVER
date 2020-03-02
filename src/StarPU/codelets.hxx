/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SpLDLT
#include "StarPU/kernels.hxx"
#include "StarPU/kernels_indef.hxx"
#include "StarPU/factor_indef.hxx"
#include "StarPU/assemble.hxx"
#include "StarPU/factor_failed.hxx"

namespace spldlt {
namespace starpu {

   ////////////////////////////////////////////////////////////

   // As it is not possible to statically intialize codelet in C++,
   // we do it via this function
   template <typename T, typename FactorAlloc, typename PoolAlloc>
   void codelet_init() {

      // activate node StarPU codelet
      starpu_codelet_init(&cl_activate_node);
      cl_activate_node.where = STARPU_CPU;
      cl_activate_node.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_activate_node.name = "ActivateNode";
      cl_activate_node.cpu_funcs[0] = activate_node_cpu_func<T, FactorAlloc,PoolAlloc>;

      // init_node StarPU codelet
      starpu_codelet_init(&cl_init_node);
      cl_init_node.where = STARPU_CPU;
      cl_init_node.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_init_node.name = "InitNode";
      cl_init_node.cpu_funcs[0] = init_node_cpu_func<T, PoolAlloc>;

      // activate_init node StarPU codelet
      starpu_codelet_init(&cl_activate_init_node);
      cl_activate_init_node.where = STARPU_CPU;
      cl_activate_init_node.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_activate_init_node.name = "ActivateInitNode";
      cl_activate_init_node.cpu_funcs[0] = activate_init_node_cpu_func<T, FactorAlloc,PoolAlloc>;

      // fini_node StarPU codelet
      starpu_codelet_init(&cl_fini_node);
      cl_fini_node.where = STARPU_CPU;
      cl_fini_node.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_fini_node.name = "FiniNode";
      cl_fini_node.cpu_funcs[0] = fini_node_cpu_func<T, PoolAlloc>;

      // factorize_block StarPU codelet
      starpu_codelet_init(&cl_factorize_block);
      cl_factorize_block.where = STARPU_CPU;
      cl_factorize_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_factorize_block.name = "FactoBlk";
      cl_factorize_block.cpu_funcs[0] = factorize_block_cpu_func<T>;

      // factorize_contrib_block StarPU codelet
      starpu_codelet_init(&cl_factorize_contrib_block);
      cl_factorize_contrib_block.where = STARPU_CPU;
      cl_factorize_contrib_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_factorize_contrib_block.name = "FactoContribBlk";
      cl_factorize_contrib_block.cpu_funcs[0] = factorize_contrib_block_cpu_func<T>;

      // solve_block StarPU codelet
      starpu_codelet_init(&cl_solve_block);
      cl_solve_block.where = STARPU_CPU;
      cl_solve_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_solve_block.name = "SolveBlk";
      cl_solve_block.cpu_funcs[0] = solve_block_cpu_func<T>;

      // solve_contrib_block StarPU codelet
      starpu_codelet_init(&cl_solve_contrib_block);
      cl_solve_contrib_block.where = STARPU_CPU;
      cl_solve_contrib_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_solve_contrib_block.name = "SolveContribBlock";
      cl_solve_contrib_block.cpu_funcs[0] = solve_contrib_block_cpu_func<T>;

      // update_block StarPU codelet
      starpu_codelet_init(&cl_update_block);
#if defined(SPLDLT_USE_GPU)
      cl_update_block.where = STARPU_CPU | STARPU_CUDA;
      // cl_update_block.where = STARPU_CPU; // Debug
#else
      cl_update_block.where = STARPU_CPU;
#endif
      cl_update_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_update_block.name = "UpdateBlock";
#if defined(SPLDLT_USE_GPU)
      cl_update_block.cuda_funcs[0] =
         sylver::spldlt::starpu::update_block_cuda_func<T>;
      cl_update_block.cuda_flags[0] = STARPU_CUDA_ASYNC;
#endif
      cl_update_block.cpu_funcs[0] = update_block_cpu_func<T>;

      // update_contrib_block StarPU codelet
      starpu_codelet_init(&cl_update_contrib_block);
#if defined(SPLDLT_USE_GPU)
      cl_update_contrib_block.where = STARPU_CPU | STARPU_CUDA;
#else
      cl_update_contrib_block.where = STARPU_CPU;
#endif
      cl_update_contrib_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_update_contrib_block.name = "UpdateContribBlock";
#if defined(SPLDLT_USE_GPU)
      cl_update_contrib_block.cuda_funcs[0] =
         sylver::spldlt::starpu::update_contrib_block_cuda_func<T>;
      cl_update_contrib_block.cuda_flags[0] = STARPU_CUDA_ASYNC;
#endif
      cl_update_contrib_block.cpu_funcs[0] = update_contrib_block_cpu_func<T>;

      // // update_diag_block StarPU codelet
      // starpu_codelet_init(&cl_update_diag_block);
      // cl_update_diag_block.where = STARPU_CPU;
      // cl_update_diag_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
      // cl_update_diag_block.name = "UPDATE_BLK";
      // cl_update_diag_block.cpu_funcs[0] = update_diag_block_cpu_func;

      // update_contrib StarPU codelet
      starpu_codelet_init(&cl_update_contrib);
#if defined(SPLDLT_USE_GPU)
      cl_update_contrib.where = STARPU_CPU | STARPU_CUDA;
      // cl_update_contrib.where = STARPU_CPU;
#else
      cl_update_contrib.where = STARPU_CPU;
#endif
      cl_update_contrib.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_update_contrib.name = "UpdateContrib";
#if defined(SPLDLT_USE_GPU)
      cl_update_contrib.cuda_funcs[0] =
         sylver::spldlt::starpu::update_contrib_cuda_func<T>;
      cl_update_contrib.cuda_flags[0] = STARPU_CUDA_ASYNC;
#endif
      cl_update_contrib.cpu_funcs[0] = update_contrib_cpu_func<T>;

      // // update_between StarPU codelet
      // starpu_codelet_init(&cl_update_between);
      // cl_update_between.where = STARPU_CPU;
      // cl_update_between.nbuffers = STARPU_VARIABLE_NBUFFERS;
      // cl_update_between.name = "UPDATE_BETWEEN_BLK";
      // cl_update_between.cpu_funcs[0] = update_between_cpu_func<T, PoolAlloc>;

      // assemble_block StarPU codelet
      starpu_codelet_init(&cl_assemble_block);
      cl_assemble_block.where = STARPU_CPU;
      cl_assemble_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_assemble_block.name = "AssembleBlk";
      cl_assemble_block.cpu_funcs[0] = assemble_block_cpu_func<T, PoolAlloc>;

      // assemble_contrib_block StarPU codelet
      starpu_codelet_init(&cl_assemble_contrib_block);
      cl_assemble_contrib_block.where = STARPU_CPU;
      cl_assemble_contrib_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_assemble_contrib_block.name = "AssembleContribBlk";
      cl_assemble_contrib_block.cpu_funcs[0] = assemble_contrib_block_cpu_func<T, PoolAlloc>;

      // subtree_assemble StarPU codelet
      starpu_codelet_init(&cl_subtree_assemble);
      cl_subtree_assemble.where = STARPU_CPU;
      cl_subtree_assemble.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_subtree_assemble.name = "SubtreeAssemble";
      cl_subtree_assemble.cpu_funcs[0] = subtree_assemble_cpu_func<T, PoolAlloc>;

      // subtree_assemble_block StarPU codelet
      starpu_codelet_init(&cl_subtree_assemble_block);
      cl_subtree_assemble_block.where = STARPU_CPU;
      cl_subtree_assemble_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_subtree_assemble_block.name = "SubtreeAssembleBlk";
      cl_subtree_assemble_block.cpu_funcs[0] = subtree_assemble_block_cpu_func<T, PoolAlloc>;
      
      // subtree_assemble_contrib StarPU codelet
      starpu_codelet_init(&cl_subtree_assemble_contrib);
      cl_subtree_assemble_contrib.where = STARPU_CPU;
      cl_subtree_assemble_contrib.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_subtree_assemble_contrib.name = "SubtreeAssembleContrib";
      cl_subtree_assemble_contrib.cpu_funcs[0] = subtree_assemble_contrib_cpu_func<T, PoolAlloc>;

      // facto_subtree StarPU codelet
      starpu_codelet_init(&cl_factor_subtree);
      cl_factor_subtree.where = STARPU_CPU;
      cl_factor_subtree.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_factor_subtree.name = "FactorSubtree";
      cl_factor_subtree.cpu_funcs[0] = factor_subtree_cpu_func<T>;

#if defined(SPLDLT_USE_GPU)
      // facto_subtree_gpu StarPU codelet
      starpu_codelet_init(&cl_factor_subtree_gpu);
      cl_factor_subtree_gpu.where = STARPU_CUDA;
      cl_factor_subtree_gpu.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_factor_subtree_gpu.name = "FactorSubtreeGPU";
      // This function can handle subtrees on the GPU
      cl_factor_subtree_gpu.cuda_funcs[0] = factor_subtree_cpu_func<T>; 
#endif
   }

   ////////////////////////////////////////////////////////////

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
         
      codelet_init<T, FactorAlloc, PoolAlloc>();

      if (!posdef) {
         codelet_init_indef<T, iblksz, Backup, PoolAlloc>();            
         codelet_init_factor_indef<T, PoolAlloc>();
         codelet_init_assemble<T, PoolAlloc>();
         sylver::spldlt::starpu::codelet_init_factor_failed<T, PoolAlloc>();
      }

   }
      
}} // end of namespaces spldlt::starpu
