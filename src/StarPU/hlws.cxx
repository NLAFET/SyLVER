#include <starpu.h>

#include <string>

#include "sylver/StarPU/hlws.hxx"

namespace sylver {
namespace starpu {

std::string const HeteroLwsScheduler::name = "hlws";

std::string const HeteroLwsScheduler::description = "Heterogeneous locality work stealing";

void HeteroLwsScheduler::add_workers(unsigned sched_ctx_id, int *workerids,unsigned nworkers) {

   using SchedulerData = HeteroLwsScheduler::Data;

   auto *sched_data = reinterpret_cast<SchedulerData*>(starpu_sched_ctx_get_policy_data(sched_ctx_id));

   for (unsigned i = 0; i < nworkers; i++) {
      int workerid = workerids[i];
      starpu_sched_ctx_worker_shares_tasks_lists(workerid, sched_ctx_id);
      auto& worker_data = sched_data->worker_data[workerid];
      worker_data.running = true;
      worker_data.busy = false;
   }
}

void HeteroLwsScheduler::remove_workers(unsigned sched_ctx_id, int *workerids,unsigned nworkers) {

   using SchedulerData = HeteroLwsScheduler::Data;

   auto *sched_data = reinterpret_cast<SchedulerData*>(starpu_sched_ctx_get_policy_data(sched_ctx_id));

   for (unsigned i = 0; i < nworkers; i++) {
      
      int workerid = workerids[i];

      auto& worker_data = sched_data->worker_data[workerid];
      worker_data.running = false;
      // free(ws->per_worker[workerid].proxlist);
      // ws->per_worker[workerid].proxlist = NULL;
   }

}
   
int HeteroLwsScheduler::select_victim(
      HeteroLwsScheduler::Data *sched_data, unsigned sched_ctx_id,
      int workerid) {

   // Round robin strategy
   
   unsigned worker = sched_data->last_pop_worker;
   unsigned nworkers;
   int *workerids = NULL;
   nworkers = starpu_sched_ctx_get_workers_list_raw(sched_ctx_id, &workerids);
   unsigned ntasks = 0;

   /* If the worker's queue is empty, let's try
    * the next ones */
   while (1)
      {
         /* Here helgrind would shout that this is unprotected, but we
          * are fine with getting outdated values, this is just an
          * estimation */
         // ntasks = ws->per_worker[workerids[worker]].queue.ntasks;

         auto& worker_data = sched_data->worker_data[workerids[worker]];
         ntasks = worker_data.task_queue.size();

         if (
               (ntasks > 0) &&
               (worker_data.busy || starpu_worker_is_blocked_in_parallel(workerids[worker]))) {
            
            break;
         }

         worker = (worker + 1) % nworkers;
         if (worker == sched_data->last_pop_worker)
            {
               /* We got back to the first worker,
                * don't go in infinite loop */
               ntasks = 0;
               break;
            }
      }

   sched_data->last_pop_worker = (worker + 1) % nworkers;

   worker = workerids[worker];

   if (ntasks)
      return worker;
   else
      return -1;

   // int nworkers = starpu_sched_ctx_get_nworkers(sched_ctx_id);
   // int i;
   // for (i = 0; i < nworkers; i++)
   //    {
   //       int neighbor = ws->per_worker[workerid].proxlist[i];
   //       /* FIXME: do not keep looking again and again at some worker
   //        * which has tasks, but that can't execute on me */
   //       int ntasks = ws->per_worker[neighbor].queue.ntasks;
   //       if (ntasks && (ws->per_worker[neighbor].busy
   //                      || starpu_worker_is_blocked_in_parallel(neighbor)))
   //          return neighbor;
   //    }
   // return -1;
   
}

   
void HeteroLwsScheduler::initialize(unsigned sched_ctx_id) {

   using SchedulerData =  HeteroLwsScheduler::Data;
   
   auto* sched_data = new SchedulerData;

   starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)sched_data);

   unsigned const nw = starpu_worker_get_count();
   
   sched_data->worker_data.resize(nw);

   /* The application may use any integer */
   if (starpu_sched_ctx_min_priority_is_set(sched_ctx_id) == 0)
      starpu_sched_ctx_set_min_priority(sched_ctx_id, INT_MIN);
   if (starpu_sched_ctx_max_priority_is_set(sched_ctx_id) == 0)
      starpu_sched_ctx_set_max_priority(sched_ctx_id, INT_MAX);

}

void HeteroLwsScheduler::finalize(unsigned sched_ctx_id) {

   using SchedulerData =  HeteroLwsScheduler::Data;

   SchedulerData *ws = reinterpret_cast<SchedulerData*>(starpu_sched_ctx_get_policy_data(sched_ctx_id));

   delete ws;
}
   
}} // End of namespace sylver::starpu
