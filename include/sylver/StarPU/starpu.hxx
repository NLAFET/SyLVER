#pragma once

namespace sylver {
namespace starpu {
 
   struct StarPU {
   // public:

      static void initialize();
      
      static int ncpu;
      static int ncuda;

   private:
      static struct starpu_conf conf_;
   };  
   
}} // End of namespace sylver::starpu

extern "C"
void sylver_starpu_init(int ncpu, int ngpu);
