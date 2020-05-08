namespace sylver {
namespace spldlt {
namespace cuda {

template<typename T>
void block_ldlt(
      cudaStream_t const stream, int nrows, int ncols, int p,
      T* a, int lda,
      T* f, int ldf,
      T* fd, int ldfd,
      T* d,
      T delta, T eps,
      int* index, int* stat
      );
   
}}} // End of namespace sylver::spldlt::cuda
