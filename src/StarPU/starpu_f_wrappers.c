#include "scheduler.h"

#include <starpu.h>
#include <starpu_cuda.h>
#include <starpu_profiling.h>
#include <limits.h>

int starpu_f_init_c(
      int ncpu,
      int ngpu) {

  int info;
  struct starpu_conf conf;

  /* printf("[starpu_f_init_c], ncpu = %d, ngpu = %d\n", ncpu, ngpu); */
  
  starpu_conf_init(&conf);

  if(ncpu >= 0)
    conf.ncpus = ncpu;
#if defined(SPLDLT_USE_GPU)
  if(ngpu >= 0)
     conf.ncuda = ngpu;
#else
     conf.ncuda = 0;
#endif
  printf("[starpu_f_init_c], conf.ncpus = %d, conf.ncuda = %d\n", conf.ncpus, conf.ncuda);

#if defined(SPLDLT_USE_GPU)
  /* conf->sched_policy_name = "dmdas"; */
  /* conf.sched_policy_name = "eager"; */
  /* conf->sched_policy_name = "lws"; */


  if(getenv("USE_LAHETEROPRIO") != NULL
          && (strcmp(getenv("USE_LAHETEROPRIO"),"TRUE")==0||strcmp(getenv("USE_LAHETEROPRIO"),"true")==0)){
      printf("[starpu_f_init_c] use laheteroprio\n");
      conf.sched_policy_name = "laheteroprio";
      conf.sched_policy_init = &init_laheteroprio;
  }
  else{
      printf("[starpu_f_init_c] use heteroprio\n");
      conf.sched_policy_name = "heteroprio";
      conf.sched_policy_init = &init_heteroprio;
  }

#else
  /* conf.sched_policy_name = "eager"; */
  conf.sched_policy_name = "ws"; // Use WS because LWS is currently buggy
  /* conf.sched_policy_name = "lws"; */
  /* conf->sched_policy_name = "prio"; */
#endif

  info = starpu_init(&conf);
  STARPU_CHECK_RETURN_VALUE(info, "starpu_init");

#if defined(SPLDLT_USE_GPU)
  
  /* auto t_starpu_cublas_init_start = std::chrono::high_resolution_clock::now(); */
  starpu_cublas_init();
  /* auto t_starpu_cublas_init_end = std::chrono::high_resolution_clock::now(); */
  /* long t_starpu_cublas_init = std::chrono::duration_cast */
     /* <std::chrono::nanoseconds>(t_starpu_cublas_init_end-t_starpu_cublas_init_start).count(); */
  /* printf("[NumericTree] StarPU cuBLAS init = %e\n", 1e-9*t_starpu_cublas_init); */
#endif


  return info;
}

void starpu_f_get_buffer(void *buffers[], int num, void **A, int *m, int *n, int *lda) {

  *A   = (void *)STARPU_MATRIX_GET_PTR(buffers[num]);
  *m   = (int)STARPU_MATRIX_GET_NX(buffers[num]);
  *n   = (int)STARPU_MATRIX_GET_NY(buffers[num]);
  *lda = (int)STARPU_MATRIX_GET_LD(buffers[num]);

  return;

}

void starpu_f_vector_get_buffer(void *buffers[], int num, void **A, int *m) {

  *A   = (void *)STARPU_VECTOR_GET_PTR(buffers[num]);
  *m   = (int)STARPU_VECTOR_GET_NX(buffers[num]);

  return;

}

void starpu_f_alloc_handle(void **ptr) {
 
   *ptr = (void *)malloc(sizeof(starpu_data_handle_t));
  
   return;
}
