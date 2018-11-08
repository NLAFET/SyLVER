# pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Initialize heteroprio scheduler
void init_heteroprio(unsigned sched_ctx);

#if defined(HAVE_LAHP)
// Initialize laheteroprio scheduler
void init_laheteroprio(unsigned sched_ctx);
#endif
   
#ifdef __cplusplus
} // extern "C"
#endif
