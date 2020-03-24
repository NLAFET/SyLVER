/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "gpu/SymbolicTreeGPU.hxx"
#include "sylver_ciface.hxx"
#include "kernels/gpu/common.hxx"
#include "gpu/NumericLevelGPU.hxx"
#include "gpu/NumericFrontGPU.hxx"
#include "kernels/llt.hxx"
#include "gpu/StackAllocGPU.hxx"
#include "kernels/gpu/convert.cuh"

// STD
#include <iostream>
#include <cassert>
#include <chrono>

// CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace sylver {
namespace spldlt {

   /// @brief Structure for handling the factors corresponding to a
   /// symmetric matrix and represented by an assembly tree.
   ///
   /// @tparam T Working precision i.e. precision used to store the
   /// coefficients from the original matrix.
   /// @tparam posdef True is the input matrix is definite positive.
   /// @tparam TDev Precision used for computing the factors. Same as
   /// T by default.
   template<typename T, bool posdef, typename TDev = T>
   class NumericTreeGPU {
   public:
      // Delete copy constructors for safety re allocated memory
      NumericTreeGPU(const NumericTreeGPU&) =delete;
      NumericTreeGPU& operator=(const NumericTreeGPU&) =delete;

      NumericTreeGPU(
            SymbolicTreeGPU& symbolic_tree, 
            T *aval,
            T* scaling,
            sylver::options_t& options,
            sylver::inform_t& inform)
         : symb_(symbolic_tree), dev_aval_(nullptr), dev_scaling_(nullptr),
           dev_stack_alloc_(0), cuhandle_(nullptr), custream_(nullptr)
      {

         std::string context = "NumericTreeGPU";
         // std::cout << "[" << context << "]" << std::endl;

         int nnodes = symbolic_tree.get_nnodes();
         
         // Setup numeric fronts
         fronts_.reserve(nnodes);
         for(int ni=0; ni<nnodes; ++ni) {
            fronts_.emplace_back(symbolic_tree[ni]);
         }

         // Setup numeric levels
         levels_.reserve(symb_.num_level);
         for (int l = 0; l<symb_.num_level; ++l) {
            levels_.emplace_back(l, symb_.lvlptr, symb_.lvllist, symb_.child_ptr, dev_stack_alloc_);
         }

         // CUDA/cuBLAS error managment
         cudaError_t cuerr;
         cublasStatus_t custat;
         // Select GPU device
         cuerr = cudaSetDevice(symbolic_tree.get_device());
         sylver::gpu::cuda_check_error(cuerr, context, "Failed to set GPU device");
  
         // Init CUDA stream
         cuerr = cudaStreamCreate(&custream_);
         // Init cuBLAS handle
         custat = cublasCreate(&cuhandle_);
         sylver::gpu::cublas_check_error(custat, context, "Failed to create cuBLAS handle");
         //std::cout << "custat = " << custat << ", cuhandle_ = " << cuhandle_ << std::endl; 
         //return;
         custat = cublasSetStream(cuhandle_, custream_);
         sylver::gpu::cublas_check_error(custat, context, "Failed to associate cuSream to cuBLAS handle");

         // Workspace size needed for factorization
         std::size_t worksz = get_workspace_size();

         // Copy A values to GPU
         std::size_t avalsz = symb_.get_nnz()*sizeof(TDev);
         std::size_t scalingsz = symb_.get_n()*sizeof(TDev);

         // Resize GPU stack allocator  
         dev_stack_alloc_.resize(worksz);
         
         cuerr = cudaMalloc((void**)&dev_aval_, avalsz);
         sylver::gpu::cuda_check_error(cuerr, context, "Failed to allocate memory for aval on GPU device");
         // cuerr = cudaMemcpy(dev_aval_, aval, avalsz, cudaMemcpyHostToDevice);
         
         // Copy scaling to GPU
         if (scaling) {
            cuerr = cudaMalloc((void**)&dev_scaling_, scalingsz);            
            sylver::gpu::cuda_check_error(cuerr, context, "Failed to allocate memory for scaling on GPU device");
            // cuerr = cudaMemcpy(dev_scaling_, scaling, scalingsz, cudaMemcpyHostToDevice);
         }

         // If necessary, create a copy of dev_aval on the GPU and
         // convert it to TDev precision
         // auto start = std::chrono::high_resolution_clock::now();
         if (std::is_same<T, TDev>::value) {
            cuerr = cudaMemcpy(dev_aval_, aval, avalsz, cudaMemcpyHostToDevice);
            sylver::gpu::cuda_check_error(cuerr, context, "Failed to send aval on the GPU device");

            if (scaling) {
               cuerr = cudaMemcpy(dev_scaling_, scaling, scalingsz, cudaMemcpyHostToDevice);
               sylver::gpu::cuda_check_error(cuerr, context, "Failed to send scaling on the GPU device");
            }
         }
         else {
            
            T *dev_aval_tmp = nullptr;
            std::size_t aval_tmp_sz = symb_.get_nnz()*sizeof(T);
            cuerr = cudaMalloc((void**)&dev_aval_tmp, aval_tmp_sz);
            sylver::gpu::cuda_check_error(cuerr, context, "Failed to allocate memory for dev_aval_tmp on GPU device");
            cuerr = cudaMemcpy(dev_aval_tmp, aval, aval_tmp_sz, cudaMemcpyHostToDevice);
            sylver::gpu::cuda_check_error(cuerr, context, "Failed to send aval on the GPU device");
            sylver::gpu::convert(custream_, symb_.get_nnz(), 1, dev_aval_tmp, symb_.get_nnz(), dev_aval_, symb_.get_nnz());
            cuerr = cudaStreamSynchronize(custream_);
            cuerr = cudaFree(dev_aval_tmp);
            sylver::gpu::cuda_check_error(cuerr, context, "Failed to deallocate aval_tmp memory on the GPU device");
 
            if (scaling) {

               T *dev_scaling_tmp = nullptr;
               std::size_t scaling_tmp_sz = symb_.get_n()*sizeof(T);
               cuerr = cudaMalloc((void**)&dev_scaling_tmp, scaling_tmp_sz);
               sylver::gpu::cuda_check_error(cuerr, context, "Failed to allocate memory for dev_scaling_tmp on GPU device");
               cuerr = cudaMemcpy(dev_scaling_tmp, scaling, scaling_tmp_sz, cudaMemcpyHostToDevice);
               sylver::gpu::cuda_check_error(cuerr, context, "Failed to send scaling on the GPU device");
               sylver::gpu::convert(custream_, symb_.get_n(), 1, dev_scaling_tmp, symb_.get_n(), dev_scaling_, symb_.get_n());
               cuerr = cudaStreamSynchronize(custream_);
               cuerr = cudaFree(dev_scaling_tmp);
               sylver::gpu::cuda_check_error(cuerr, context, "Failed to deallocate scaling_tmp memory on the GPU device");

            }
         }
         // auto end = std::chrono::high_resolution_clock::now();
         // long tsend = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
         // printf("send aval and scaling to GPU time (s) = %e\n", 1e-9*tsend);

         std::chrono::high_resolution_clock::time_point fact_sa, fact_en;
            
         // Compute factors
         // fact_sa = std::chrono::high_resolution_clock::now();
         factor(options, inform);
         // fact_en = std::chrono::high_resolution_clock::now();
         // long tfactor = std::chrono::duration_cast<std::chrono::nanoseconds>(fact_en-fact_sa).count();

         // printf("[NumericTreeGPU][Profile] factor time (s) = %e\n", 1e-9*tfactor);
         
         
      }

      ~NumericTreeGPU() {
         // Free A entries on GPU
         if (dev_aval_) cudaFree(dev_aval_);
         // Cleanup CUDA stream
         cudaStreamDestroy(custream_);
         // Cleanup cuBLAS handles 
         cublasDestroy(cuhandle_);
      }

      // Copy factors on the GPU to host memory 
      void setup_host_solve() {

         std::string context = "NumericTreeGPU::setup_solve_host";
         cudaError_t cuerr; // CUDA error

         // GPU buffer for converting factor entries from TDev to T
         // precision
         T *dev_buffer = nullptr; 

         if (!std::is_same<T, TDev>::value) {
            // Allocate buffer for data convertion between TDev and T
            // types
            auto max_front_nfactor = symb_.get_max_front_nfactor();

            cuerr = cudaMalloc(
                  (void**)&dev_buffer,
                  sylver::gpu::aligned_size(max_front_nfactor*sizeof(T)));            
            sylver::gpu::cuda_check_error(cuerr, context, "Failed to allocate buffer array on GPU");
         }

         int nnodes = symb_.get_nnodes();

         for(int ni=0; ni<nnodes; ++ni) {
            // If lcol is not NULL then assume that factors have
            // already been copied back to host
            if (fronts_[ni].lcol == nullptr) {
               if (fronts_[ni].dev_lcol == nullptr)
                  std::cout << "[Error] Factors not computed" << std::endl;
               int nrow = fronts_[ni].get_nrow();
               int ncol = fronts_[ni].get_ncol();
               std::size_t factorsz = (nrow+2)*ncol;
               // std::size_t factorsz = nrow*ncol;
               // TODO Use memory allocator for host (Stack allocator)
               fronts_[ni].lcol = new T[factorsz];
               fronts_[ni].ldl = nrow;
               
               // std::cout << "nrow=" << nrow << ", ncol=" << ncol << std::endl;
               // std::cout << "factorsz = " << factorsz << std::endl;
               // std::cout << "lcol = " << fronts_[ni].lcol << std::endl;
               // std::cout << "dev_lcol = " << fronts_[ni].dev_lcol << std::endl;
               
               if (std::is_same<T, TDev>::value) {
                  cuerr = cudaMemcpy(fronts_[ni].lcol, fronts_[ni].dev_lcol, factorsz*sizeof(T), cudaMemcpyDeviceToHost);
                  sylver::gpu::cuda_check_error(cuerr, context, "Failed to copy factors to host memory");
               }
               else {
                  // Convert factors on GPU from TDev to T before
                  // sending retrieving them on host
                  // std::cout << "converting factor entries " << std::endl;
                  sylver::gpu::convert(custream_, factorsz, 1, fronts_[ni].dev_lcol, factorsz, dev_buffer, factorsz);
                  cuerr = cudaStreamSynchronize(custream_);
                  sylver::gpu::cuda_check_error(cuerr, context, "Failed to synchronize stream");
                  cuerr = cudaMemcpy(fronts_[ni].lcol, dev_buffer, factorsz*sizeof(T), cudaMemcpyDeviceToHost);
                  sylver::gpu::cuda_check_error(cuerr, context, "Failed to copy factors to host memory");
               }
               // sylver::print_mat(nrow, ncol, fronts_[ni].lcol, fronts_[ni].get_ldl());
            }
         }

         if (!std::is_same<T, TDev>::value) {
            // Deallocate buffer used for data conversion
            cuerr = cudaFree(dev_buffer);
            sylver::gpu::cuda_check_error(cuerr, context, "Failed to deallocate buffer memory on the GPU device");

         }
      }

      // Note: single rhs, posdef only
      void host_solve_fwd(T* x) {

         std::vector<T> xlocal(symb_.get_n());
         
         // Bottom-up tree traversal 
         for(int ni=0; ni<symb_.get_nnodes(); ++ni) {

            // Front info
            int nrow = fronts_[ni].get_nrow();
            int ncol = fronts_[ni].get_ncol();
            int ldl = fronts_[ni].get_ldl();

            int const *map;
            map = symb_[ni].rlist;

            // Gather into xlocal
            for(int i=0; i<nrow; ++i)
               xlocal[i] = x[map[i]-1]; // Fortran indexed

            // Perform fwd solve
            sylver::spldlt::cholesky_solve_fwd(
                  nrow, ncol, fronts_[ni].lcol, ldl, 1, &xlocal[0], symb_.get_n());

            // Scatter result
            for(int i=0; i<nrow; ++i)
               x[map[i]-1] = xlocal[i];

         }
      }

      void host_solve_bwd(T* x) {

         std::vector<T> xlocal(symb_.get_n());

         // Top-down tree traversal 
         for(int ni=symb_.get_nnodes()-1; ni>=0; --ni) {

            int nrow = fronts_[ni].get_nrow();
            int ncol = fronts_[ni].get_ncol();
            int ldl = fronts_[ni].get_ldl();

            int const *map;
            map = symb_[ni].rlist;

            // Gather into xlocal
            for(int i=0; i<nrow; ++i)
               xlocal[i] = x[map[i]-1]; // Fortran indexed

            // Perform bwd solve
            sylver::spldlt::cholesky_solve_bwd(
                  nrow, ncol, fronts_[ni].lcol, ldl, 1, &xlocal[0], symb_.get_n());

            // Scatter result (only first nelim entries have changed)
            for(int i=0; i<ncol; ++i)
               x[map[i]-1] = xlocal[i];

         }
      }
      
   private:
      
      void factor(
            sylver::options_t& options,
            sylver::inform_t& inform) {

         if (symb_.num_level <= 0) return;

         std::string context = "NumericTreeGPU::factor";
         cudaError_t cuerr; // CUDA error

         // Number of nodes in the tree
         int nnodes = symb_.get_nnodes();
         int const* sptr = symb_.get_sptr();
         long const* rptr = symb_.get_rptr();
         long const* nptr = symb_.get_nptr();
         // Pointers to subtree contributions
         // std::vector<void*> gpu_contribs(nnodes);
         // std::fill(gpu_contribs.begin(), gpu_contribs.end(), nullptr);
         
         // Allocate workspace for forming contribution blocks
         std::size_t max_dev_cb_work_sz = 0;
         std::size_t dev_cb_work_sz = 0;
         for (int l=0; l<symb_.num_level; ++l) {
            // Compute workspace size (fully and non fully summed
            // coefficients) for current level
            dev_cb_work_sz = 0;
            for (int p = symb_.lvlptr[l]; p<symb_.lvlptr[l+1]; ++p) {
               int n = symb_.lvllist[p-1]-1; // lvlptr is 1-indexed
               int nrow = symb_[n].nrow;
               dev_cb_work_sz = dev_cb_work_sz + nrow*nrow; 
            }
            // Determine maximum workspace size needed for
            // factorization
            max_dev_cb_work_sz = std::max(max_dev_cb_work_sz,
                                              dev_cb_work_sz);
         }
         // std::cout << "max_dev_cb_work_sz = " << max_dev_cb_work_sz << std::endl;
         
         // Allocate twice the (maximum) storage needed for each level
         // (one for the current level and one for the previous level)
         TDev *dev_cb_work = nullptr;
         cuerr = cudaMalloc(
               (void**)&dev_cb_work,
               sylver::gpu::aligned_size(2*max_dev_cb_work_sz*sizeof(TDev)));            
         sylver::gpu::cuda_check_error(
               cuerr, context,
               "Failed to allocate memory for factor workspace on GPU device");
         // printf("dev_cb_work = %p\n", dev_cb_work);
         
         // Init nodes (Fortran) structure
         // void * nodes = nullptr;
         // spral_ssids_nodes_init(nnodes, &nodes);
         // assert(nodes != nullptr);
         
         // Init asminf (Fortran) structure
         // void *asminf = nullptr;
         // spral_ssids_asminf_init(
         //       nnodes, symb_.child_ptr, symb_.child_list,
         //       symb_.get_sptr(), symb_.get_rptr(), &(symb_.rlist_direct)[0],
         //       &asminf);
         // assert(asminf != nullptr);
         
         // Init GPU stack allocator for workspace
         // void *gpu_work_alloc = nullptr; // GPU stack allocator
         // spral_ssids_custack_init(&gpu_work_alloc);
         // assert(gpu_work_alloc != nullptr);

         // Pointers to GPU workspaces for factors and contribution
         // blocks for current and previous levels. It is necessary to
         // have both in order to perform assembly
         TDev *dev_cb_work_lvl = nullptr; // Workspace for current level
         TDev *dev_cb_work_pre = nullptr; // Workspace for previous level

         // Memory offset for each node's contribution block
         std::vector<long> lvlcbofs(nnodes);

         // Timings
         std::chrono::high_resolution_clock::time_point init_sa, init_en,
            asm_sa, asm_en, fact_sa, fact_en, formcontrib_sa, formcontrib_en,
            asmcontrib_sa, asmcontrib_en, sa, en;
         long t_init = 0, t_asm = 0, t_fact = 0, t_formcontrib = 0,
            t_asmcontrib = 0, time = 0;

         // long tfactor = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
         
         // Loop over level and perform operations: assembly, factor, etc
         for (int l=0; l<symb_.num_level; ++l) {

            NumericLevelGPU<T, TDev> &level = levels_[l];

            // Init workspace for current level that stores computed
            // factors and contribution blocks
            if ((level.get_lvl() % 2) == 0)
               dev_cb_work_lvl = dev_cb_work;
            else
               dev_cb_work_lvl = &dev_cb_work[max_dev_cb_work_sz];
               // dev_cb_work_lvl = (T*)sylver::gpu::aligned_ptr(dev_cb_work, max_dev_cb_work_sz*sizeof(T));

            // printf("dev_cb_work_lvl = %p\n", dev_cb_work_lvl);
            // printf("dev_cb_work_pre = %p\n", dev_cb_work_pre);
                     
            // Setup workspace memory for current level 
            // int info;
            // spral_ssids_level_custack_init(
            //       l, nnodes, &(symb_.lvlptr)[0], &(symb_.lvllist)[0],
            //       symb_.child_ptr, symb_.child_list,
            //       nodes, sptr, rptr, asminf, gpu_work_alloc,
            //       &info);

            // Compute number of entries in level
            level.set_lvlsz(fronts_);
            
            // std::cout << "l = " << l << ", lvlsz = " << lvlsz << std::endl; 
            // std::cout << "l = " << l << ", lvlnch = " << level.get_nch() << std::endl; 

            // Allocate GPU memory for factors in current level
            // Use stack alloc for factors
            cuerr = cudaMalloc((void**)&(level.dev_lcol),
                               sylver::gpu::aligned_size(level.get_lvlsz()*sizeof(TDev)));
            sylver::gpu::cuda_check_error(cuerr, context, "Failed to allocate memory for level factors");

            // TODO In indef case, allocate memory storage for
            // computing LD

            // Determine lcol pointer on GPU for each node in current level
            std::size_t ofs = 0; // Memory offset for factors
            long cbofs = 0; // Memory offset for cb
            for (int p = symb_.lvlptr[l]; p<symb_.lvlptr[l+1]; ++p) {
               // lvlptr is 1-indexed
               int n = symb_.lvllist[p-1]-1; // lvllist is 1-indexed
               int nrow = fronts_[n].get_nrow();
               int ncol = fronts_[n].get_ncol();
               TDev *dev_lcol = nullptr;
               // Setup lcol pointer for node n in current level
               dev_lcol = &level.dev_lcol[ofs];
               // dev_lcol = (T*)sylver::gpu::aligned_ptr(
               //       level.dev_lcol, ofs*sizeof(T));
               // printf("level.dev_lcol = %p, dev_lcol = %p \n", level.dev_lcol, dev_lcol);
               fronts_[n].dev_lcol = dev_lcol;
               fronts_[n].dev_ldl = nrow;
               // spral_ssids_node_set_gpu_lcol(nnodes, nodes, n, dev_lcol); // Update nodes structure
               // Compute ofset for factor entries within level memory storage
               ofs = ofs + (nrow+2)*ncol; // FIXME: diagonal unecessary in posdef case
               // Store ofset 
               lvlcbofs[n] = cbofs;
               int cbm = nrow - ncol;
               cbofs = cbofs + cbm*cbm; // This offset includes the contribution block
               // Init local permuation array
               // spral_ssids_node_init_lperm(nnodes, nodes, n, sptr);
            }

            // init_sa = std::chrono::high_resolution_clock::now();

            // Zero memory for storing factors and initialize it with
            // original matrix A
            // spral_ssids_init_l_with_a(
            //       &custream_, nnodes, level.get_lvl(), &(symb_.lvlptr)[0],
            //       &(symb_.lvllist)[0], nodes, level.get_nnodes(), level.get_lvlsz(),
            //       symb_.get_nptr(), symb_.get_rptr(), symb_.dev_nlist, 
            //       symb_.dev_rlist, dev_aval_, level.dev_lcol,
            //       gpu_work_alloc, &info, dev_scaling_);
            level.init_lcol_with_a(
                  custream_, fronts_, nptr, rptr,
                  symb_.dev_nlist, symb_.dev_rlist, dev_aval_,
                  dev_scaling_);
            // cudaStreamSynchronize(custream_);
            // init_en = std::chrono::high_resolution_clock::now();
            // t_init += std::chrono::duration_cast<std::chrono::nanoseconds>(init_en-init_sa).count();

            // Assemble fully-summed columns

            // asm_sa = std::chrono::high_resolution_clock::now();
            // spral_ssids_assemble_fully_summed(
            //       &custream_, nnodes, level.get_nch(), level.get_lvl(), &(symb_.lvlptr)[0],
            //       &(symb_.lvllist)[0], nodes, dev_cb_work_pre, &gpu_contribs[0],
            //       level.dev_lcol, symb_.dev_rlist_direct, symb_.child_ptr,
            //       symb_.child_list, &lvlcbofs[0], asminf, rptr, sptr,
            //       gpu_work_alloc, &info);
            // level.assemble_fully_summed_basic(
            //       custream_, fronts_,
            //       symb_.dev_rlist_direct, rptr,
            //       dev_cb_work_pre, lvlcbofs,
            //       symb_.child_ptr, symb_.child_list);

            // debug
            level.assemble_fully_summed(
                  custream_, fronts_,
                  symb_.dev_rlist_direct, rptr,
                  dev_cb_work_pre, lvlcbofs,
                  symb_.child_ptr, symb_.child_list);
            // cudaStreamSynchronize(custream_);
            // asm_en = std::chrono::high_resolution_clock::now();
            // t_asm += std::chrono::duration_cast<std::chrono::nanoseconds>(asm_en-asm_sa).count();

            // Factor fully-summed columns
            cudaStreamSynchronize(custream_);
            fact_sa = std::chrono::high_resolution_clock::now();

            // spral_ssids_factor_posdef(
            //       &custream_, nnodes, level.get_lvl(), &(symb_.lvlptr)[0],
            //       &(symb_.lvllist)[0], nodes, rptr, sptr, level.dev_lcol,
            //       &cuhandle_, gpu_work_alloc);

            level.factor(cuhandle_, fronts_);
            cudaStreamSynchronize(custream_);
            fact_en = std::chrono::high_resolution_clock::now();
            t_fact += std::chrono::duration_cast<std::chrono::nanoseconds>(fact_en-fact_sa).count();
            
            // Form contribution block

            // cudaStreamSynchronize(custream_);
            formcontrib_sa = std::chrono::high_resolution_clock::now();

            // if (cbofs>0)
            //    spral_ssids_form_contrib(
            //          &custream_, nnodes, level.get_lvl(), &(symb_.lvlptr)[0],
            //          &(symb_.lvllist)[0], nodes, rptr, sptr, &lvlcbofs[0],
            //          dev_cb_work_lvl, gpu_work_alloc);
            
            if (cbofs>0)
               level.form_contrib(
                     custream_, fronts_, lvlcbofs, dev_cb_work_lvl);
            cudaStreamSynchronize(custream_);
            formcontrib_en = std::chrono::high_resolution_clock::now();
            t_formcontrib += std::chrono::duration_cast<std::chrono::nanoseconds>(formcontrib_en-formcontrib_sa).count();


            // Assemble contribution block
            // asmcontrib_sa = std::chrono::high_resolution_clock::now();

            // spral_ssids_assemble_contrib(
            //       &custream_, nnodes, level.get_nch(), level.get_lvl(),
            //       &(symb_.lvlptr)[0], &(symb_.lvllist)[0], symb_.child_ptr,
            //       symb_.child_list, rptr, sptr, asminf, cbofs, &lvlcbofs[0],
            //       dev_cb_work_pre, &gpu_contribs[0], dev_cb_work_lvl,
            //       symb_.dev_rlist_direct, gpu_work_alloc);
            // level.assemble_contrib_basic(
            //       custream_, fronts_,
            //       symb_.dev_rlist_direct, rptr,
            //       dev_cb_work_pre, lvlcbofs,
            //       symb_.child_ptr, symb_.child_list,
            //       dev_cb_work_lvl);
            if(cbofs>0)
               level.assemble_contrib(
                     custream_, fronts_,
                     symb_.dev_rlist_direct, rptr,
                     dev_cb_work_pre, lvlcbofs,
                     symb_.child_ptr, symb_.child_list,
                     dev_cb_work_lvl);
            // asmcontrib_en = std::chrono::high_resolution_clock::now();
            // cudaStreamSynchronize(custream_);
            // t_asmcontrib += std::chrono::duration_cast<std::chrono::nanoseconds>(asmcontrib_en-asmcontrib_sa).count();
            
            // Set current GPU workspace as previous for future
            // assembly operations
            dev_cb_work_pre = dev_cb_work_lvl; 


         }

         // Wait for CUDA kernels
         cudaStreamSynchronize(custream_);

         // printf("[Profile] Time (s) = %e\n", 1e-9*time);
            
         // printf("[Profile] Init time (s) = %e\n", 1e-9*t_init);
         // printf("[Profile] Assemble time (s) = %e\n", 1e-9*t_asm);
         printf("[Profile] Factor time (s) = %e\n", 1e-9*t_fact);
         printf("[Profile] Form contrib time (s) = %e\n", 1e-9*t_formcontrib);
         // printf("[Profile] Assemble contrib time (s) = %e\n", 1e-9*t_asmcontrib);

         // TODO deallocate nodes structure
         // TODO deallocate asminf structure
      }

      /// @brief Estimate the size of the workspace (in bytes) on the
      /// device needed for the factorization of the tree
      std::size_t get_workspace_size() const {

         std::size_t worksz = 0;
         for (int l = 0; l<symb_.num_level; ++l) {
            worksz = std::max(
                  worksz,
                  levels_[l].get_workspace_size(fronts_, symb_.child_ptr, symb_.child_list));
         }

         return worksz;
      }
      
   private:
      SymbolicTreeGPU& symb_;
      // A entries on GPU
      TDev *dev_aval_;
      // Scaling array on GPU
      TDev *dev_scaling_;
      // CUDA
      cudaStream_t custream_;
      cublasHandle_t cuhandle_;
      std::vector<NumericLevelGPU<T, TDev>> levels_; // Level sets
      std::vector<sylver::spldlt::NumericFrontGPU<T, TDev>> fronts_; // Fronts and factor entries 
      // Memory stack allocator on the GPU
      sylver::gpu::StackAllocGPU<TDev> dev_stack_alloc_;
   };
   
}} // End of sylver::spldlt namespace
