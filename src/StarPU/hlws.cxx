#include "sylver/StarPU/hlws.hxx"

#include <algorithm>
#include <iostream>
#include <string>

#include <starpu.h>
#include <starpu_scheduler.h>

// extern "C" {
// hlws_sched_policy.init_sched = HeteroLwsScheduler::initialize;
// hlws_sched_policy.pre_exec_hook = NULL;
   // hlws_sched_policy.policy_name = "hlws";
// }

namespace sylver {
namespace starpu {

struct starpu_sched_policy HeteroLwsScheduler::hlws_sched_policy_;

struct starpu_sched_policy& HeteroLwsScheduler::starpu_sched_policy() {

   struct starpu_sched_policy& sched = HeteroLwsScheduler::hlws_sched_policy_;
   
   sched.init_sched = HeteroLwsScheduler::initialize;
   sched.deinit_sched = HeteroLwsScheduler::finalize;
   sched.add_workers = HeteroLwsScheduler::add_workers;
   sched.remove_workers = HeteroLwsScheduler::remove_workers;
   sched.push_task = HeteroLwsScheduler::push_task;
   sched.pop_task = HeteroLwsScheduler::pop_task;
   sched.pre_exec_hook = NULL;
   sched.post_exec_hook = NULL;
   sched.pop_every_task = NULL;
   sched.policy_name = "hlws";
   sched.policy_description = "locality work stealing";
// #ifdef STARPU_HAVE_HWLOC
   // sched.worker_type = STARPU_WORKER_TREE;
// #else
   sched.worker_type = STARPU_WORKER_LIST;
// #endif

   return sched;
}
   
// struct starpu_sched_policy
   
   // {
   //    policy_name : "hlws"                                         
   //    };

// {
//  .init_sched = HeteroLwsScheduler::initialize,
//  .deinit_sched = HeteroLwsScheduler::finalize,
//  .add_workers = HeteroLwsScheduler::add_workers,
//  .remove_workers = HeteroLwsScheduler::remove_workers,
//  .push_task = HeteroLwsScheduler::push_task,
//  .pop_task = HeteroLwsScheduler::pop_task,
//  .pre_exec_hook = NULL,
//  .post_exec_hook = NULL,
//  .pop_every_task = NULL,
//  .policy_name = "hlws",
//  .policy_description = "heterogeneous locality work stealing",
//  // #ifdef STARPU_HAVE_HWLOC
//  // .worker_type = STARPU_WORKER_TREE,
//  // #else
//  .worker_type = STARPU_WORKER_LIST,
//  // #endif

// };

   
std::string const HeteroLwsScheduler::name = "hlws";

std::string const HeteroLwsScheduler::description = "Heterogeneous locality work stealing";

bool can_execute(starpu_task* task, int workerid) {

   // // Debug
   // if (starpu_worker_get_type(workerid) == STARPU_CUDA_WORKER) {
   //    return false;
   // }
   
   for (unsigned i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++) {
      if (starpu_worker_can_execute_task(workerid, task, i)) {
         
            starpu_task_set_implementation(task, i);
            return true;
         }
   }
   return false;
   
}
   
struct starpu_task* HeteroLwsScheduler::pick_task(
      HeteroLwsScheduler::Data *sched_data, int source, int target) {

   starpu_task* task = nullptr;
   
   auto& source_list = sched_data->worker_data[source].task_queue;
   auto task_iterator = source_list.begin();

   while (task_iterator != source_list.end()) {

      task = *task_iterator;

      if (can_execute(task, target)) {
         // Task can be executed on target worker, pick it and remove
         // it from the source worker task queue
         source_list.erase(task_iterator);

         // Debug
         if (starpu_worker_get_type(target) == STARPU_CUDA_WORKER) {
            std::cout << "[HeteroLwsScheduler::pick_task] source = " << source
                      << ", task = " << task
                      << ", task name = " << task->cl->name
                      << std::endl;
         }
         
         return task;
      }

      task_iterator++;
   }

   return nullptr;
}
   
struct starpu_task* HeteroLwsScheduler::pop_task(unsigned sched_ctx_id) {

   // std::cout << "[HeteroLwsScheduler::pop_task]" << std::endl;

   using SchedulerData = HeteroLwsScheduler::Data;
   
   auto *sched_data = reinterpret_cast<SchedulerData*>(starpu_sched_ctx_get_policy_data(sched_ctx_id));
   
   struct starpu_task *task = NULL;
   // Note: starpu_worker_get_id_check is similar to
   // `starpu_worker_get_id` exepct that it aborts if not called from
   // a worker
   unsigned workerid = starpu_worker_get_id_check();
   auto& worker_data = sched_data->worker_data[workerid];

   // StarPU Context list
   // unsigned *sched_ctxs = NULL;
   // int nsched_ctxs = -1;
   // starpu_worker_get_sched_ctx_list(workerid, sched_ctxs);
   // assert(NULL != sched_ctxs);
   // assert(nsched_ctxs > 0);
   
   worker_data.busy = false;

#ifdef STARPU_NON_BLOCKING_DRIVERS
   // if (STARPU_RUNNING_ON_VALGRIND || !worker_data.task_queue.empty())
   if (!worker_data.task_queue.empty())
#endif
      {
         task = HeteroLwsScheduler::pick_task(sched_data, workerid, workerid);
         // if (task) {
         //    locality_popped_task(ws, task, workerid, sched_ctx_id);
         // }
      }

   if (task) {
      /* there was a local task */
      worker_data.busy = true;
      // if (_starpu_get_nsched_ctxs() > 1)
      //    {
      //       starpu_worker_relax_on();
      //       _starpu_sched_ctx_lock_write(sched_ctx_id);
      //       starpu_worker_relax_off();
      //       starpu_sched_ctx_list_task_counters_decrement(sched_ctx_id, workerid);
      //       if (_starpu_sched_ctx_worker_is_master_for_child_ctx(sched_ctx_id, workerid, task))
      //          task = NULL;
      //       _starpu_sched_ctx_unlock_write(sched_ctx_id);
      //    }
      return task;
   }

   // Debug
   // worker_data.busy = !!task;
   // return task;

   // Debug
   // if (starpu_worker_get_type(workerid) == STARPU_CUDA_WORKER) {
   //    worker_data.busy = !!task;
   //    return task;
   // }
   
   // std::cout << "[HeteroLwsScheduler::pop_task] Stealing task " << std::endl;

   /* we need to steal someone's job */
   starpu_worker_relax_on();
   // Select a suitable worker to steal task from 
   int victim = HeteroLwsScheduler::select_victim(sched_data, sched_ctx_id, workerid);
   starpu_worker_relax_off();

   // std::cout << "[HeteroLwsScheduler::pop_task] victim = " << victim << std::endl;

   if (victim == -1) {
      // Could not find a worker to steal task from. Return no task.
      
      return NULL;
   }

   // Try locking `victim` worker. Note: `starpu_worker_trylock`
   // return 0 if successful
   if (starpu_worker_trylock(victim)) {
      
      /* victim is busy, don't bother it, come back later */
      return NULL;
   }

   auto& victim_data = sched_data->worker_data[victim];

   if (victim_data.running && !victim_data.task_queue.empty()) {
      // Victim is running and has ready tasks available in its task
      // queue

      task = HeteroLwsScheduler::pick_task(sched_data, victim, workerid);
       
      // if (starpu_worker_get_type(workerid) != STARPU_CUDA_WORKER) {
      //    task = HeteroLwsScheduler::pick_task(sched_data, victim, workerid);
      // }
      // Debug
      // if (starpu_worker_get_type(workerid) == STARPU_CUDA_WORKER) {
      //    task = nullptr;
      // }
   }

   if (task) {
      // _STARPU_TRACE_WORK_STEALING(workerid, victim);
      starpu_sched_task_break(task);
      starpu_sched_ctx_list_task_counters_decrement(sched_ctx_id, victim);
      // record_data_locality(task, workerid);
      // record_worker_locality(ws, task, workerid, sched_ctx_id);
      // locality_popped_task(ws, task, victim, sched_ctx_id);
   }
   starpu_worker_unlock(victim);

#ifndef STARPU_NON_BLOCKING_DRIVERS

   /* While stealing, perhaps somebody actually give us a task, don't miss
    * the opportunity to take it before going to sleep. */
   {
      struct _starpu_worker *worker = _starpu_get_worker_struct(starpu_worker_get_id());
      if (!task && worker->state_keep_awake)
         {
            task = HeteroLwsScheduler::pick_task(sched_data, workerid, workerid);
            if (task)
               {
                  /* keep_awake notice taken into account here, clear flag */
                  worker->state_keep_awake = 0;
                  // locality_popped_task(ws, task, workerid, sched_ctx_id);
               }
         }
   }
#endif

   // if (task && _starpu_get_nsched_ctxs() > 1) {
      
   //    starpu_worker_relax_on();
   //    _starpu_sched_ctx_lock_write(sched_ctx_id);
   //    starpu_worker_relax_off();
   //    if (_starpu_sched_ctx_worker_is_master_for_child_ctx(sched_ctx_id, workerid, task))
   //       task = NULL;
   //    _starpu_sched_ctx_unlock_write(sched_ctx_id);
   //    if (!task)
   //       return NULL;
   // }

   // Record weather we are busy or not 
   worker_data.busy = !!task;
   return task;
}

   
unsigned HeteroLwsScheduler::select_worker(
      HeteroLwsScheduler::Data *sched_data, struct starpu_task *task,
      unsigned sched_ctx_id) {


   // Round robin
   
   unsigned worker;
   unsigned nworkers;
   int *workerids;
   nworkers = starpu_sched_ctx_get_workers_list_raw(sched_ctx_id, &workerids);

   worker = sched_data->last_push_worker;
   auto& worker_data = sched_data->worker_data[workerids[worker]];
   do {
      worker = (worker + 1) % nworkers;
      worker_data = sched_data->worker_data[workerids[worker]];
   }
   while (!worker_data.running || !starpu_worker_can_execute_task_first_impl(workerids[worker], task, NULL));

   sched_data->last_push_worker = worker;

   return workerids[worker];

}
   
int HeteroLwsScheduler::push_task(struct starpu_task *task) {

   // std::cout << "[HeteroLwsScheduler::push_task]" << std::endl;

   using SchedulerData = HeteroLwsScheduler::Data;

   unsigned sched_ctx_id = task->sched_ctx;
   auto *sched_data = reinterpret_cast<SchedulerData*>(starpu_sched_ctx_get_policy_data(sched_ctx_id));
   int workerid;

// #ifdef USE_LOCALITY
//    workerid = select_worker_locality(ws, task, sched_ctx_id);
// #else
   workerid = -1;
// #endif
   if (workerid == -1)
      workerid = starpu_worker_get_id();

   /* If the current thread is not a worker but the main thread (-1)
    * or the current worker is not in the target context, we find the
    * better one to put task on its queue */
   if (workerid == -1 || !starpu_sched_ctx_contains_worker(workerid, sched_ctx_id) ||
       !starpu_worker_can_execute_task_first_impl(workerid, task, NULL)) {
      workerid = select_worker(sched_data, task, sched_ctx_id);
   }

   assert(workerid != -1);
   
   starpu_worker_lock(workerid);
   // STARPU_AYU_ADDTOTASKQUEUE(starpu_task_get_job_id(task), workerid);
   starpu_sched_task_break(task);
   // record_data_locality(task, workerid);
   // STARPU_ASSERT_MSG(worker_data.running, "workerid=%d, ws=%p\n", workerid, sched_data);
   // _starpu_prio_deque_push_back_task(&ws->per_worker[workerid].queue, task);

   // std::cout << "[HeteroLwsScheduler::push_task] workerid = " << workerid << std::endl;

   auto& worker_data = sched_data->worker_data[workerid];

   // assert(std::find(worker_data.task_queue.begin(),
   //                  worker_data.task_queue.end(), task) ==
   //        worker_data.task_queue.end());
      
   worker_data.task_queue.push_back(task);

   // std::cout << "[HeteroLwsScheduler::push_task] task_queue::size() = " << worker_data.task_queue.size() << std::endl;

   // locality_pushed_task(ws, task, workerid, sched_ctx_id);

   starpu_push_task_end(task);
   starpu_worker_unlock(workerid);
   starpu_sched_ctx_list_task_counters_increment(sched_ctx_id, workerid);

#if !defined(STARPU_NON_BLOCKING_DRIVERS) || defined(STARPU_SIMGRID)
   /* TODO: implement fine-grain signaling, similar to what eager does */
   struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
   struct starpu_sched_ctx_iterator it;

   workers->init_iterator(workers, &it);
   while(workers->has_next(workers, &it))
      starpu_wake_worker_relax_light(workers->get_next(workers, &it));
#endif
   return 0;

}
   
void HeteroLwsScheduler::add_workers(unsigned sched_ctx_id, int *workerids,unsigned nworkers) {

   std::cout << "[HeteroLwsScheduler::add_workers]" << std::endl;

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

   // std::cout << "[HeteroLwsScheduler::select_victim] last_pop_worker = " << sched_data->last_pop_worker << std::endl;

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

   std::cout << "[HeteroLwsScheduler::initialize]" << std::endl;
   
   using SchedulerData =  HeteroLwsScheduler::Data;
   
   auto* sched_data = new SchedulerData;

   starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)sched_data);

   sched_data->last_pop_worker = 0;
   sched_data->last_push_worker = 0;

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
