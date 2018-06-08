
namespace spldlt { namespace tests {

      template<typename T, bool debug = false>
      int update_block_test(
            int m, int n, int k,
            int test=0, int seed=0) {

         bool failed = false;

         T u = 0.01;
         T small = 1e-20;
         
         printf("[update_block_test] m = %d, n = %d\n", m, n);

         // Create matrix for generating D
         T* c = new T[k*k];
         T* d = new T[2*k];

         T *tmp = new T[2*k];
         int* perm = new int[k];
         for(int i=0; i<k; i++) perm[i] = i;

         int nelim = 0;

         nelim += ldlt_tpp_factor(
               k, k, perm, c, k, d, tmp, k, true, u, small, 0, c, k);

         delete[] tmp;
         delete[] perm;
         
         // Create matrix L_ij
         T* l_ij = new T[m*n];
         
         // Create matrix L_ij
         T* l_ik = new T[m*k];

         // Create matrix L_ij
         T* l_jk = new T[n*k];

         ////////////////////////////////////////
         // Cleanup memory
         delete[] c;
         delete[] l_ij;
         delete[] l_ik;
         delete[] l_jk;

         return failed ? -1 : 0;

      }

}} // namespace spldlt::tests
