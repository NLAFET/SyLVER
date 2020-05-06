#pragma once

#include <starpu.h>

#include <string>
#include <vector>
#include <list>

namespace sylver {
namespace starpu {

class HeteroLwsScheduler {
public:

   struct WorkerData {
      std::list<struct starpu_task *> task_queue;
      bool running;
      bool busy;
   };

   struct Data {
      std::vector<WorkerData> worker_data;

      unsigned last_pop_worker;
      unsigned last_push_worker;
   };
         
   static void initialize(unsigned sched_ctx_id);

   static void finalize(unsigned sched_ctx_id);

   static void add_workers(
         unsigned sched_ctx_id, int *workerids, unsigned nworkers);

   static void remove_workers(
         unsigned sched_ctx_id, int *workerids, unsigned nworkers);

   static int push_task(struct starpu_task *task);

   static struct starpu_task *pop_task(unsigned sched_ctx_id);

   static std::string const name;

   static std::string const description;

   static struct starpu_sched_policy& starpu_sched_policy();
   
private:

   static struct starpu_sched_policy hlws_sched_policy_;

   static struct starpu_task* pick_task(
         HeteroLwsScheduler::Data *sched_data, int source, int target);
   
   static int select_victim(
         HeteroLwsScheduler::Data *sched_data, unsigned sched_ctx_id,
         int workerid);

   static unsigned select_worker(
         HeteroLwsScheduler::Data *sched_data, struct starpu_task *task,
         unsigned sched_ctx_id);
};
   
}} // End of namespace sylver::starpu
