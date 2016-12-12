#pragma once

// #include "StarPU/kernels.hxx"

#include <starpu.h>

namespace spldlt { namespace starpu {

      /* factor_block_app StarPU task
         
       */
      
      /* factor_block_app StarPU codelet */
      static
      struct starpu_codelet cl_factor_block_app;      

      /* factor_block_app CPU kernel */
      template<typename T>
      void
      factor_block_app_cpu_func(void *buffers[], void *cl_arg) {
         
      }

      /* applyN_block_app StarPU codelet */
      static
      struct starpu_codelet cl_applyN_block_app;      

      /*  applyN_block_app StarPU task
         
       */
      template<typename T>
      void
      applyN_block_app_cpu_func(void *buffers[], void *cl_arg) {
         
      }

      /* applyT_block_app StarPU codelet */
      static
      struct starpu_codelet cl_applyT_block_app;      

      /*  applyT_block_app StarPU task
         
       */
      template<typename T>
      void
      applyT_block_app_cpu_func(void *buffers[], void *cl_arg) {
         
      }

      /* updateN_block_app StarPU codelet */
      static
      struct starpu_codelet cl_updateN_block_app;      

      /*  updateN_block_app StarPU task
         
       */
      template<typename T>
      void 
      updateN_block_app_cpu_func(void *buffers[], void *cl_arg) {
         
      }

      /* updateT_block_app StarPU codelet */
      static
      struct starpu_codelet cl_updateT_block_app;      

      /*  updateT_block_app StarPU task
         
       */
      template<typename T>
      void 
      updateT_block_app_cpu_func(void *buffers[], void *cl_arg) {
         
      }

      /* As it is not possible to statically intialize codelet in C++,
         we do it via this function */
      template <typename T, typename PoolAlloc>
      void codelet_init_indef() {

         // Init codelet for posdef tasks 
         // codelet_init<T, PoolAlloc>();

         // Initialize factor_block_app StarPU codelet
         starpu_codelet_init(&cl_factor_block_app);
         cl_factor_block_app.where = STARPU_CPU;
         cl_factor_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_factor_block_app.name = "FACOTR_BLK_APP";
         cl_factor_block_app.cpu_funcs[0] = factor_block_app_cpu_func<T>;

         // Initialize applyN_block_app StarPU codelet
         starpu_codelet_init(&cl_applyN_block_app);
         cl_applyN_block_app.where = STARPU_CPU;
         cl_applyN_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_applyN_block_app.name = "FACOTR_BLK_APP";
         cl_applyN_block_app.cpu_funcs[0] = applyN_block_app_cpu_func<T>;
         
         // Initialize applyT_block_app StarPU codelet
         starpu_codelet_init(&cl_applyT_block_app);
         cl_applyT_block_app.where = STARPU_CPU;
         cl_applyT_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_applyT_block_app.name = "FACOTR_BLK_APP";
         cl_applyT_block_app.cpu_funcs[0] = applyT_block_app_cpu_func<T>;

         // Initialize updateN_block_app StarPU codelet
         starpu_codelet_init(&cl_updateN_block_app);
         cl_updateN_block_app.where = STARPU_CPU;
         cl_updateN_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_updateN_block_app.name = "FACOTR_BLK_APP";
         cl_updateN_block_app.cpu_funcs[0] = updateN_block_app_cpu_func<T>;

         // Initialize updateT_block_app StarPU codelet
         starpu_codelet_init(&cl_updateT_block_app);
         cl_updateT_block_app.where = STARPU_CPU;
         cl_updateT_block_app.nbuffers = STARPU_VARIABLE_NBUFFERS;
         cl_updateT_block_app.name = "FACOTR_BLK_APP";
         cl_updateT_block_app.cpu_funcs[0] = updateT_block_app_cpu_func<T>;
      }
}} /* namespaces spldlt::starpu  */
