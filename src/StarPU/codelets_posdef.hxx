/// @file
/// @copyright 2020- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "StarPU/assemble.hxx"
#include "StarPU/kernels.hxx"
#include "StarPU/subtree.hxx"

namespace sylver {
namespace spldlt {
namespace starpu {

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

      // subtree_assemble_contrib_block StarPU codelet
      starpu_codelet_init(&cl_subtree_assemble_contrib_block);
      cl_subtree_assemble_contrib_block.where = STARPU_CPU;
      cl_subtree_assemble_contrib_block.nbuffers = STARPU_VARIABLE_NBUFFERS;
      cl_subtree_assemble_contrib_block.name = "SubtreeAssembleContribBlk";
      cl_subtree_assemble_contrib_block.cpu_funcs[0] = subtree_assemble_contrib_block_cpu_func<T, PoolAlloc>;

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

   template <typename T,
             typename FactorAlloc,
             typename PoolAlloc>
   void codelets_init_posdef() {
      codelet_init<T, FactorAlloc, PoolAlloc>();
      codelet_init_assemble<T, PoolAlloc>();
   }
   
}}} // End of namespace sylver::spldlt::starpu
