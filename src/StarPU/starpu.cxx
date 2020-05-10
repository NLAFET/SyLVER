#include <starpu.h>

#include "sylver/StarPU/hlws.hxx"
#include "sylver/StarPU/starpu.hxx"
#include "StarPU/scheduler.h"

namespace sylver {
namespace starpu {

int StarPU::ncpu = 0;
int StarPU::ncuda = 0;

StarPU::Scheduler StarPU::sched = StarPU::Scheduler::LWS;
   
struct starpu_conf StarPU::conf;

void StarPU::initialize() {

   int ret;

   starpu_conf_init(&conf);

   // if(ncpu >= 0) {
   conf.ncpus = ncpu;
   // }

#if defined(SPLDLT_USE_GPU)
   // if(ngpu >= 0) {
   conf.ncuda = ncuda;
   // }
#else
   conf.ncuda = 0;
#endif

   switch (StarPU::sched) {
   case Scheduler::HLWS:
      std::cout << "[StarPU::initialize] Init StarPU witth HLWS scheduler" << std::endl;
      conf.sched_policy_name = NULL;
      sylver::starpu::StarPU::conf.sched_policy =
         &sylver::starpu::HeteroLwsScheduler::starpu_sched_policy();
      break;
   case Scheduler::HP:
      std::cout << "[StarPU::initialize] Init StarPU witth HP scheduler" << std::endl;
      conf.sched_policy_name = "heteroprio";
      conf.sched_policy_init = &init_heteroprio;
      break;
   case Scheduler::LWS:
      std::cout << "[StarPU::initialize] Init StarPU witth LWS scheduler" << std::endl;
      conf.sched_policy_name = "lws";
      break;
   case Scheduler::WS:
      std::cout << "[StarPU::initialize] Init StarPU witth WS scheduler" << std::endl;
      conf.sched_policy_name = "ws";      
      break;
   default:
      std::cout << "[StarPU::initialize] Init StarPU witth LWS scheduler" << std::endl;
      conf.sched_policy_name = "lws";
   }
   
   
   ret = starpu_init(&conf);
   STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#if defined(SPLDLT_USE_GPU)
  
   /* auto t_starpu_cublas_init_start = std::chrono::high_resolution_clock::now(); */
   starpu_cublas_init();
   /* auto t_starpu_cublas_init_end = std::chrono::high_resolution_clock::now(); */
   /* long t_starpu_cublas_init = std::chrono::duration_cast */
   /* <std::chrono::nanoseconds>(t_starpu_cublas_init_end-t_starpu_cublas_init_start).count(); */
   /* printf("[NumericTree] StarPU cuBLAS init = %e\n", 1e-9*t_starpu_cublas_init); */
#endif

}

}} // End of namespace sylver::starpu

extern "C"
void sylver_starpu_init(int ncpu, int ngpu) {

   sylver::starpu::StarPU::ncpu = ncpu;
   sylver::starpu::StarPU::ncuda = ngpu;
      
   sylver::starpu::StarPU::initialize();
}
