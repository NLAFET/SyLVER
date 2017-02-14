#include "kernels/ldlt_app.hxx"

// namespace spral { namespace ssids { namespace cpu {
namespace spldlt {

using namespace spral::ssids::cpu;
using namespace spldlt::ldlt_app_internal;

template<typename T>
size_t ldlt_app_factor_mem_required(int m, int n, int block_size) {
   int const align = 32;
   return align_lda<T>(m) * n * sizeof(T) + align; // CopyBackup
}

// template<typename T, typename Allocator>
// int ldlt_app_factor(int m, int n, int* perm, T* a, int lda, T* d, T beta, T* upd, int ldupd, struct cpu_factor_options const& options, std::vector<Workspace>& work, Allocator const& alloc) {
//    // If we've got a tall and narrow node, adjust block size so each block
//    // has roughly blksz**2 entries
//    // FIXME: Decide if this reshape is actually useful, given it will generate
//    //        a lot more update tasks instead?
//    int outer_block_size = options.cpu_block_size;
//    /*if(n < outer_block_size) {
//        outer_block_size = int((long(outer_block_size)*outer_block_size) / n);
//    }*/

// #ifdef PROFILE
//    Profile::setState("TA_MISC1");
// #endif

//    // Template parameters and workspaces
//    bool const debug = false;
//    //PoolBackup<T, Allocator> backup(m, n, outer_block_size, alloc);
//    CopyBackup<T, Allocator> backup(m, n, outer_block_size, alloc);

//    // Actual call
//    bool const use_tasks = true;
//    return LDLT
//       <T, INNER_BLOCK_SIZE, CopyBackup<T,Allocator>, use_tasks, debug,
//        Allocator>
//       ::factor(
//             m, n, perm, a, lda, d, backup, options, options.pivot_method,
//             outer_block_size, beta, upd, ldupd, work, alloc
//             );
// }
// template int ldlt_app_factor<double, BuddyAllocator<double,std::allocator<double>>>(int, int, int*, double*, int, double*, double, double*, int, struct cpu_factor_options const&, std::vector<Workspace>&, BuddyAllocator<double,std::allocator<double>> const& alloc);

template <typename T>
void ldlt_app_solve_fwd(int m, int n, T const* l, int ldl, int nrhs, T* x, int ldx) {
   if(nrhs==1) {
      host_trsv(FILL_MODE_LWR, OP_N, DIAG_UNIT, n, l, ldl, x, 1);
      if(m > n)
         gemv(OP_N, m-n, n, -1.0, &l[n], ldl, x, 1, 1.0, &x[n], 1);
   } else {
      host_trsm(SIDE_LEFT, FILL_MODE_LWR, OP_N, DIAG_UNIT, n, nrhs, 1.0, l, ldl, x, ldx);
      if(m > n)
         host_gemm(OP_N, OP_N, m-n, nrhs, n, -1.0, &l[n], ldl, x, ldx, 1.0, &x[n], ldx);
   }
}
template void ldlt_app_solve_fwd<double>(int, int, double const*, int, int, double*, int);

template <typename T>
void ldlt_app_solve_diag(int n, T const* d, int nrhs, T* x, int ldx) {
   for(int i=0; i<n; ) {
      if(i+1==n || std::isfinite(d[2*i+2])) {
         // 1x1 pivot
         T d11 = d[2*i];
         for(int r=0; r<nrhs; ++r)
            x[r*ldx+i] *= d11;
         i++;
      } else {
         // 2x2 pivot
         T d11 = d[2*i];
         T d21 = d[2*i+1];
         T d22 = d[2*i+3];
         for(int r=0; r<nrhs; ++r) {
            T x1 = x[r*ldx+i];
            T x2 = x[r*ldx+i+1];
            x[r*ldx+i]   = d11*x1 + d21*x2;
            x[r*ldx+i+1] = d21*x1 + d22*x2;
         }
         i += 2;
      }
   }
}
template void ldlt_app_solve_diag<double>(int, double const*, int, double*, int);

template <typename T>
void ldlt_app_solve_bwd(int m, int n, T const* l, int ldl, int nrhs, T* x, int ldx) {
   if(nrhs==1) {
      if(m > n)
         gemv(OP_T, m-n, n, -1.0, &l[n], ldl, &x[n], 1, 1.0, x, 1);
      host_trsv(FILL_MODE_LWR, OP_T, DIAG_UNIT, n, l, ldl, x, 1);
   } else {
      if(m > n)
         host_gemm(OP_T, OP_N, n, nrhs, m-n, -1.0, &l[n], ldl, &x[n], ldx, 1.0, x, ldx);
      host_trsm(SIDE_LEFT, FILL_MODE_LWR, OP_T, DIAG_UNIT, n, nrhs, 1.0, l, ldl, x, ldx);
   }
}
template void ldlt_app_solve_bwd<double>(int, int, double const*, int, int, double*, int);

// }}} /* namespaces spral::ssids::cpu */

}
