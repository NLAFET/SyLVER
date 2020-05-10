#pragma once

namespace sylver {
namespace starpu {
   
   struct StarPU {
   // public:

      enum class Scheduler
         {
          // Heterogeneous Locality Work Stealing
          HLWS,
          HP,
          LWS,
          WS,
         };

      static void initialize();
      
      static int ncpu;
      static int ncuda;

      static Scheduler sched;
      
   // private:
      static struct starpu_conf conf;
   };  
   
}} // End of namespace sylver::starpu

extern "C"
void sylver_starpu_init(int ncpu, int ngpu);
