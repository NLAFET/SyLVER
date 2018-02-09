#include <stdlib.h> 
#include <stdio.h> 
#include <mutex>
#include <starpu.h>

// namespace spldlt {

extern "C"
void* test_malloc(int p, void *ptr_in) {

   void *ptr;

   ptr = (void *)malloc(8*sizeof(double));
   // *ptr = NULL;
   // ptr = NULL;
   // ptr = (void *)(0x10+(void *)p);
   // array = NULL;
   // array = new double[8];
   // ptr = ptr_in;
   ptr = (void *)(0x10+(void *)p);
   printf("[test_malloc] ptr = %p, p = %d, workerid = %d\n", ptr, p, starpu_worker_get_id());

   return (void*)ptr;
}

// }
