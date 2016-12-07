#pragma once

#include <starpu.h>

namespace spldlt { namespace starpu {

      /* As it is not possible to statically intialize codelet in C++,
         we do it via this function */
      template <typename T, typename PoolAlloc>
      void codelet_init_indef() {

         // Init codelet for posdef tasks 
         codelet_init();
      }
   
}} /* namespaces spldlt::starpu  */
