#pragma once

// SpLDLT
#include "StarPU/kernels.hxx"
#include "StarPU/kernels_indef.hxx"
#include "StarPU/factor_indef.hxx"
#include "StarPU/assemble.hxx"
#include "StarPU/factor_failed.hxx"

namespace spldlt { namespace starpu {

      template <typename T,
                int iblksz,
                typename Backup,
                typename FactorAlloc,
                typename PoolAlloc>
      void codelets_init(bool posdef) {
         
         codelet_init<T, FactorAlloc, PoolAlloc>();

         if (!posdef) {
            codelet_init_indef<T, iblksz, Backup, PoolAlloc>();            
            codelet_init_factor_indef<T, PoolAlloc>();
            codelet_init_assemble<T, PoolAlloc>();
            codelet_init_factor_failed<T, PoolAlloc>();
         }

      }
      
   } // end of namespaces starpu
} // end of namespaces spldlt
