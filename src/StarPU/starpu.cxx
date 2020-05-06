#include <starpu.h>

#include "sylver/StarPU/hlws.hxx"
#include "sylver/StarPU/starpu.hxx"

namespace sylver {
namespace starpu {

int StarPU::ncpu = 0;
int StarPU::ncuda = 0;

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
   
   // conf.sched_policy_name = "lws";

   // conf.sched_policy_name = "ws";

   conf.sched_policy_name = NULL;
   sylver::starpu::StarPU::conf.sched_policy =
      &sylver::starpu::HeteroLwsScheduler::starpu_sched_policy();

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
