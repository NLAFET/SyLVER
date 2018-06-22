#include <starpu_heteroprio.h>

namespace spldlt { namespace starpu {

      // static const int FACTOR_APP_PRIO   = 0;
      // static const int ADJUST_APP_PRIO   = 0;
      // static const int APPLYN_APP_PRIO   = 1;
      // static const int RESTORE_APP_PRIO  = 1;
      // static const int UPDATEN_APP_PRIO  = 2;
      // static const int UPDATEC_APP_PRIO  = 2;
      // static const int APPLYT_APP_PRIO   = 3;
      // static const int UPDATET_APP_PRIO  = 3;

      void init_heteroprio(unsigned sched_ctx) {

         int nprio = 4;
         
         // Create queues for CPU tasks
         starpu_heteroprio_set_nb_prios(sched_ctx, STARPU_CPU_IDX, nprio);

         // Set lookup order for CPU workers
         // 0 => 3
         // 1 => 2
         // ..
         // 3 => 0
         // Use simple mapping
         for (int bucketid=0; bucketid<nprio; bucketid++) {

            // int prio = nprio-bucketid-1;
            // int prio = bucketid;
            starpu_heteroprio_set_mapping(sched_ctx, STARPU_CPU_IDX, bucketid, bucketid);
            starpu_heteroprio_set_faster_arch(sched_ctx, STARPU_CPU_IDX, bucketid);
         }

         // Number of tasks (prio) running on CUDA
         int nprio_cuda = 1; 
         
         // Create queues for CUDA tasks
         starpu_heteroprio_set_nb_prios(sched_ctx, STARPU_CUDA_IDX, nprio_cuda);

         int updaten_prio     = 2;
         // int updaten_bucketid = 2; // nprio-p-1 with p=1
         // Set lookup order for CUDA workers
         starpu_heteroprio_set_mapping(sched_ctx, STARPU_CUDA_IDX, 0, updaten_prio);

         starpu_heteroprio_set_faster_arch(sched_ctx, STARPU_CUDA_IDX, updaten_prio);
         starpu_heteroprio_set_arch_slow_factor(sched_ctx, STARPU_CPU_IDX, updaten_prio, 40.0f);

      }

}}
