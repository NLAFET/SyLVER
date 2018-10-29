# pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Initialize heteroprio scheduler
void init_heteroprio(unsigned sched_ctx);

// Initialize laheteroprio scheduler
void init_laheteroprio(unsigned sched_ctx);

#ifdef __cplusplus
} // extern "C"
#endif
