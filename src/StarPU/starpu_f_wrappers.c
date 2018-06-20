#include <starpu.h>
#include <starpu_cuda.h>
#include <starpu_profiling.h>
#include <limits.h>

int starpu_f_init_c(int ncpus) {

  int info;
  struct starpu_conf *conf = malloc(sizeof(struct starpu_conf));
  
  starpu_conf_init(conf);

  if(ncpus > 0)
    conf->ncpus = ncpus;  
  printf("starpu_f_init_c, ncpu: %d\n", conf->ncpus);
#if defined(SPLDLT_USE_GPU)
  conf->ncuda = 1;
#endif

#if defined(SPLDLT_USE_GPU)
  /* conf->sched_policy_name = "dmdas"; */
  conf->sched_policy_name = "eager";
  /* conf->sched_policy_name = "lws"; */
#else
  conf->sched_policy_name = "ws"; // Use WS because LWS is currently buggy
  /* conf->sched_policy_name = "lws"; */
  /* conf->sched_policy_name = "prio"; */
#endif

  info = starpu_init(conf);
  STARPU_CHECK_RETURN_VALUE(info, "starpu_init");

  free(conf);
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
