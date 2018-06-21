#include <starpu_heteroprio.h>

namespace spldlt { namespace starpu {

      // static const int FACTOR_APP_PRIO   = 3;
      // static const int APPLYN_APP_PRIO   = 2;
      // static const int APPLYT_APP_PRIO   = 0;
      // static const int ADJUST_APP_PRIO   = 3;
      // static const int RESTORE_APP_PRIO  = 2;
      // static const int UPDATEN_APP_PRIO  = 1;
      // static const int UPDATET_APP_PRIO  = 0;
      // static const int UPDATEC_APP_PRIO  = 1;

      void init_heteroprio(unsigned sched_ctx) {
         
         // Create 4 queues for CPU tasks
         starpu_heteroprio_set_nb_prios(sched_ctx, STARPU_CPU_IDX, 4);
         
         
         
      }

}}
