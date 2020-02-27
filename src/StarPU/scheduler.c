#include <starpu_heteroprio.h>
#if defined(HAVE_LAHP)
#include <starpu_laheteroprio.h>
#endif

// static const int FACTOR_APP_PRIO   = 0;
// static const int ADJUST_APP_PRIO   = 0;
// static const int APPLYN_APP_PRIO   = 1;
// static const int RESTORE_APP_PRIO  = 1;
// static const int UPDATEN_APP_PRIO  = 2;
// static const int UPDATEC_APP_PRIO  = 2;
// static const int APPLYT_APP_PRIO   = 3;
// static const int UPDATET_APP_PRIO  = 3;

#ifdef __cplusplus
extern "C" {
#endif


#if defined(HAVE_LAHP)
void init_laheteroprio(unsigned sched_ctx) {
    /* printf("[init_laheteroprio]\n"); */

    int nprio = 4;

    // Create queues for CPU tasks
    starpu_laheteroprio_set_nb_prios(sched_ctx, STARPU_CPU_IDX, nprio);

    // Set lookup order for CPU workers
    // 0 => 3
    // 1 => 2
    // ..
    // 3 => 0
    // Use simple mapping
    for (int bucketid=0; bucketid<nprio; bucketid++) {

       // int prio = nprio-bucketid-1;
       // int prio = bucketid;
       starpu_laheteroprio_set_mapping(sched_ctx, STARPU_CPU_IDX, bucketid, bucketid);
       starpu_laheteroprio_set_faster_arch(sched_ctx, STARPU_CPU_IDX, bucketid);
    }

    // Number of tasks (prio) running on CUDA
    int nprio_cuda = 1;

    // Create queues for CUDA tasks
    starpu_laheteroprio_set_nb_prios(sched_ctx, STARPU_CUDA_IDX, nprio_cuda);

    int updaten_prio     = 2;
    // int updaten_bucketid = 2; // nprio-p-1 with p=1
    // Set lookup order for CUDA workers
    starpu_laheteroprio_set_mapping(sched_ctx, STARPU_CUDA_IDX, 0, updaten_prio);

    starpu_laheteroprio_set_faster_arch(sched_ctx, STARPU_CUDA_IDX, updaten_prio);
    starpu_laheteroprio_set_arch_slow_factor(sched_ctx, STARPU_CPU_IDX, updaten_prio, 30.0f);


    // Init specific to la
    starpu_laheteroprio_map_wgroup_memory_nodes(sched_ctx);
    // Can be print out for debug :
    starpu_laheteroprio_print_wgroups(sched_ctx);
}
#endif
   
void init_heteroprio(unsigned sched_ctx) {
   /* printf("[init_heteroprio]\n"); */

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
   /* starpu_heteroprio_set_arch_slow_factor(sched_ctx, STARPU_CPU_IDX, updaten_prio, 30.0f); */
   starpu_heteroprio_set_arch_slow_factor(sched_ctx, STARPU_CPU_IDX, updaten_prio, 40.0f);
}

#ifdef __cplusplus
} // extern "C"
#endif
