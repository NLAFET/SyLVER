/// @file
/// @copyright 2016- The Science and Technology Facilities Council (STFC)
/// @author Florent Lopez
#pragma once

// SyLVER
#include "kernels/common.hxx"
#include "NumericFrontGPU.hxx"
#include "kernels/gpu/assemble.hxx"
#include "kernels/gpu/syrk.hxx"
#include "kernels/gpu/reorder.hxx"
#include "kernels/gpu/factor.hxx"
#include "gpu/StackAllocGPU.hxx"
#include "gpu/factor.hxx"

namespace sylver {
namespace spldlt {

   /// @tparam T Working precision i.e. precision used to store the
   /// coefficients.
   template<typename T, typename TDev = T>
   class NumericLevelGPU {
   public:

      /// @param lvl Level index in the assembly tree (0-indexed)
      NumericLevelGPU(
            int lvl, std::vector<int> const& lvlptr, std::vector<int> const& lvllist,
            int const* child_ptr, sylver::gpu::StackAllocGPU<TDev> const& dev_stack_alloc)
         : lvl_(lvl), lvlptr_(lvlptr), lvllist_(lvllist), dev_lcol(nullptr),
           lvlsz_(0), dev_stack_alloc_(dev_stack_alloc)
      {

         // Determine number of nodes in level
         lvl_nnodes_ = lvlptr[lvl+1] - lvlptr[lvl];
         
         // Count number of children nodes
         lvl_nch_ = 0;
         max_nch_ = 0;
         for (int p = lvlptr[lvl]; p<lvlptr[lvl+1]; ++p) {
            int ni = lvllist[p-1]-1; // Note that lvllist is 1-indexed
            int nch = child_ptr[ni+1] - child_ptr[ni];
            lvl_nch_ = lvl_nch_ + nch;
            max_nch_ = std::max(max_nch_, nch);
         }

      }

      ~NumericLevelGPU() {
         // Free level data on GPU
         if (dev_lcol) cudaFree(dev_lcol);
      }

      /// @brief Compute the number of entries in level
      void set_lvlsz(std::vector<sylver::spldlt::NumericFrontGPU<T, TDev>>& fronts) {

         lvlsz_ = 0;
         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
            // Note that lvlptr is 1-indexed
            int ni = lvllist_[p-1]-1; // Note that lvllist is 1-indexed
            int nrow = fronts[ni].get_nrow();
            int ncol = fronts[ni].get_ncol();
            lvlsz_ = lvlsz_ + (nrow+2)*ncol; // FIXME: diagonal unecessary in posdef case
         }         
      }

      void factor(
            cublasHandle_t& cuhandle,
            std::vector<sylver::spldlt::NumericFrontGPU<T, TDev>>& fronts) {

         std::string context = "NumericLevelGPU::factor";
         cudaError_t cuerr;
         cublasStatus_t custat; // CuBLAS status

         // std::cout << "[" << context << "]" << std::endl;

         cudaStream_t custream;
         custat = cublasGetStream(cuhandle, &custream);
         sylver::gpu::cublas_check_error(custat, context);
         
         //
         // Copy lower triangle into upper triangle so we can use
         // access (i,j) or (j,i) to get the same number while
         // pivoting.
         //

         std::vector<sylver::spldlt::gpu::multisymm_type<TDev>> msymmdata(get_nnodes());
         typename std::vector<sylver::spldlt::gpu::multisymm_type<TDev>>
            ::iterator msymm = msymmdata.begin();

         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {            
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
            int nrow = fronts[ni].get_nrow();
            int ncol = fronts[ni].get_ncol();
            TDev *dev_lcol = fronts[ni].dev_lcol;            

            msymm->nrows = nrow; 
            msymm->ncols = ncol; 
            msymm->lcol = dev_lcol; 
            
            ++msymm;
         }

         sylver::spldlt::gpu::multisymm_type<TDev> *dev_msymmdata = nullptr;
         // multisymm GPU allocator
         sylver::gpu::StackAllocGPU<sylver::spldlt::gpu::multisymm_type<TDev>>
            multisymm_type_alloc(dev_stack_alloc_);
         dev_msymmdata = multisymm_type_alloc.allocate(get_nnodes()); // Allocate memory on GPU for msymm

         cuerr = cudaMemcpyAsync(
               dev_msymmdata, &msymmdata[0],
               get_nnodes()*sizeof(sylver::spldlt::gpu::multisymm_type<TDev>),
               cudaMemcpyHostToDevice,
               custream);
         sylver::gpu::cuda_check_error(cuerr, context, "Failed to send msdata to the GPU");
                  
         sylver::spldlt::gpu::multisymm(custream, get_nnodes(), dev_msymmdata);

         multisymm_type_alloc.deallocate(dev_msymmdata, get_nnodes());

         // Allocate stat parameter
         sylver::gpu::StackAllocGPU<int> int_alloc(dev_stack_alloc_);
         int *dev_stat = int_alloc.allocate(1);
         
         //         
         // Factor several nodes simultaneously
         // 

         std::vector<int> multi_nrow(get_nnodes()); // Number of rows
         std::vector<int> multi_ncol(get_nnodes()); // Number of columns
         std::vector<TDev*> multi_lcol(get_nnodes());
         std::vector<int> multi_nelim(get_nnodes());

         TDev* dev_panel = nullptr; // Buffer for computing the panel
         sylver::gpu::StackAllocGPU<TDev> factor_type_alloc(dev_stack_alloc_);
         
         // Number of nodes to be processed simultaneously
         int nmultinode = 0;
         int tot_nrow = 0; // Total nrow
         int max_nrow = 0; // Max nrow
         
         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {            
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed

            int nrow = fronts[ni].get_nrow();
            int ncol = fronts[ni].get_ncol();
            TDev *dev_lcol = fronts[ni].dev_lcol;
            
            if (ncol < nrow) {
               multi_nrow[nmultinode] = nrow;
               multi_ncol[nmultinode] = ncol;
               multi_lcol[nmultinode] = dev_lcol;
               ++nmultinode;
               tot_nrow += nrow; 
            }
            else {
               // Root node
               std::max(max_nrow, nrow);
            }
         }

         // std::cout << context << ", lvl_nnodes: " << get_nnodes() << ", nmultinode: " << nmultinode << std::endl;
         
         // Determine biggest node
         max_nrow = std::max(max_nrow, tot_nrow);

         // Allocate panel buffer on GPU
         std::size_t panel_sz = max_nrow*BLOCK_SIZE;
         dev_panel = factor_type_alloc.allocate(panel_sz);
         
         if (nmultinode > 0) {
            // Launch batched panel factorization of multiple nodes
            // (i.e non-root)

            sylver::spldlt::gpu::multinode_factor(
                  cuhandle, nmultinode, multi_nrow, multi_ncol, multi_lcol,
                  this->dev_lcol, dev_panel, multi_nelim, dev_stack_alloc_);

            // TODO: Retrieve stats
            // ..
         }

         //
         // Factor root nodes
         // 

         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
            int nrow = fronts[ni].get_nrow();
            int ncol = fronts[ni].get_ncol();
            TDev *dev_lcol = fronts[ni].dev_lcol;
            int dev_ldl = fronts[ni].get_dev_ldl(); // Leading dimn

            if (nrow == ncol) {
               // Root node
               sylver::spldlt::gpu::factor(
                     cuhandle, 
                     nrow, ncol,
                     dev_lcol, dev_ldl,
                     dev_panel,
                     dev_stat);
            }
         }

         // Deallocate panel buffer on GPU
         factor_type_alloc.deallocate(dev_panel, panel_sz);
         // Deallocate stat
         int_alloc.deallocate(dev_stat, 1);

      }
      
      void form_contrib(
            cudaStream_t& custream,
            std::vector<sylver::spldlt::NumericFrontGPU<T, TDev>>& fronts,
            std::vector<long>& lvlcbofs,
            TDev *dev_cb_work_lvl // Contrib entries in current level
            ) {

         std::string context = "NumericLevelGPU::form_contrib";

         // Count the number of blocks
         int ncb = 0;
         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {            
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
            int m = fronts[ni].get_nrow();
            int n = fronts[ni].get_ncol();
            int cbm = m-n;
            int k = (cbm - 1)/32 + 1;
            ncb += (k*(k + 1))/2;
         }

         std::vector<sylver::spldlt::gpu::multisyrk_type<TDev>> msdata(ncb);

         // Fill-in msdata
         ncb = 0;
         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {            
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
            // Node info
            int m = fronts[ni].get_nrow();
            int n = fronts[ni].get_ncol();
            int nelim = fronts[ni].nelim;
            int cbm = m-n; // Front cb dimn
            int k = (cbm - 1)/32 + 1;
            k = (k*(k + 1))/2; // Number of 32x32 blocks in cb
            TDev *dev_lcol = fronts[ni].dev_lcol;
            long cbofs = lvlcbofs[ni];
            for (int j = 0; j < k; ++j) {
               msdata[ncb+j].first = ncb;
               msdata[ncb+j].lval = &dev_lcol[n];
               // If gpu_ldcol for LDLT case
               msdata[ncb+j].ldval = &dev_lcol[n]; // Posdef case
               msdata[ncb+j].offc = cbofs;
               msdata[ncb+j].n = cbm;
               msdata[ncb+j].k = nelim;
               msdata[ncb+j].lda = m;
               msdata[ncb+j].ldb = m;
            }
            ncb += k;
         }

         if (ncb > 0) {

            cudaError_t cuerr;

            sylver::spldlt::gpu::multisyrk_type<TDev> *dev_msdata = nullptr;
            sylver::gpu::StackAllocGPU<sylver::spldlt::gpu::multisyrk_type<TDev>> multisyrk_type_alloc(dev_stack_alloc_);         
            dev_msdata = multisyrk_type_alloc.allocate(ncb); // Allocate memory on GPU for msdata
            
            cuerr = cudaMemcpyAsync(
               dev_msdata, &msdata[0],
               ncb*sizeof(sylver::spldlt::gpu::multisyrk_type<TDev>),
               cudaMemcpyHostToDevice,
               custream);
            sylver::gpu::cuda_check_error(cuerr, context, "Failed to send msdata to the GPU");
            
            sylver::spldlt::gpu::multisyrk_low_col(
                  custream, ncb, dev_msdata, dev_cb_work_lvl);

            multisyrk_type_alloc.deallocate(dev_msdata, ncb);
            
         }
         
      }
      
      /// @brief Assemble entries in contribution block
      void assemble_contrib_basic(
            cudaStream_t& custream,
            std::vector<sylver::spldlt::NumericFrontGPU<T, TDev>>& fronts,
            int const* dev_rlist_direct, long const* rptr,
            TDev const* dev_cb_work_pre, // Contrib entries in previous level
            std::vector<long>& lvlcbofs,
            int const* child_ptr, int const* child_list,
            TDev *dev_cb_work // Contrib entries in current level
            ) {

         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
            
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
            int m = fronts[ni].get_nrow();
            int n = fronts[ni].get_ncol();
            int cbsz = m-n;
            long cbofs = lvlcbofs[ni];
            TDev *dev_cb = &dev_cb_work[cbofs];

            int nch = child_ptr[ni+1] - child_ptr[ni];
            if (nch == 0) continue; 

            for (int cp = child_ptr[ni]; cp<child_ptr[ni+1]; ++cp) {

               int ci = child_list[cp-1]-1;
               int cm = fronts[ci].get_nrow();
               int cn = fronts[ci].get_ncol();
               int ccbsz = cm-cn;

               long ccbofs = lvlcbofs[ci];
               TDev const* dev_cb_pre = &dev_cb_work_pre[ccbofs];
               int const* dev_crlist_direct = &dev_rlist_direct[rptr[ci]+cn-1];

               sylver::spldlt::gpu::assemble_contrib(
                     custream, cm, cn, dev_crlist_direct, dev_cb_pre, ccbsz,
                     m, n, dev_cb, cbsz);

            }
         }
            
      }

      /// @brief Setup assembly of contrib entries
      void assemble_contrib_setup(
            cudaStream_t& custream,
            std::vector<sylver::spldlt::NumericFrontGPU<T, TDev>>& fronts,
            int /*const*/ * dev_rlist_direct, long const* rptr,
            TDev /*const*/ *dev_cb_work_pre, std::vector<long>& lvlcbofs,
            int const* child_ptr, int const* child_list,
            TDev *dev_cb_work, // Contrib entries in current level
            sylver::spldlt::gpu::assemble_cp_type<TDev> *& dev_cpdata, int& ncp,
            sylver::spldlt::gpu::assemble_blk_type *& dev_blkdata, int& nblk,
            unsigned int *& dev_sync
            ) {

         std::string context = "NumericLevelGPU::assemble_contrib_setup";

         // Return if level has no children nodes
         if (get_nch() <= 0) return;

         // Determine the number of children node with contributions
         // into parent cb: ncp as well as the number of blocks: nblk
         ncp = 0;
         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
            int m = fronts[ni].get_nrow(); 
            int n = fronts[ni].get_ncol();
            if( (m-n) > 0) {

               for (int cptr = child_ptr[ni]; cptr<child_ptr[ni+1]; ++cptr) {
                  int ci = child_list[cptr-1]-1; // Note: child_list is 1-indexed
                  int cm = fronts[ci].get_nrow();
                  int cn = fronts[ci].get_ncol();
                  int cbm = cm-cn;
                  int npassl = fronts[ci].get_npassl();
                  // Number of cols assembled into parent contrib block
                  int npasscb = cbm-npassl;

                  if (npasscb > 0) {
                     ncp++;
                     int bx = (npasscb-1) / HOGG_ASSEMBLE_TX + 1;
                     int by = (npasscb-1) / HOGG_ASSEMBLE_TY + 1;
                     nblk += calc_blks_lwr(bx, by, HOGG_ASSEMBLE_TX, HOGG_ASSEMBLE_TY);
                  }

               }
            }
         }

         ////////////////////
         // cpdata

         std::vector<sylver::spldlt::gpu::assemble_cp_type<TDev>> cpdata(ncp);

         // Fill in child-parent information

         int cpi = 0; // cpdata index
         
         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
            int m = fronts[ni].get_nrow(); 
            int n = fronts[ni].get_ncol();
            long cbofs = lvlcbofs[ni]; // Ptr to contrib data for parent node

            int blk = 0;

            if ( (m-n) > 0) {
               // The current node has a contrib block
               
               for (int cptr = child_ptr[ni]; cptr<child_ptr[ni+1]; ++cptr) {
                  int ci = child_list[cptr-1]-1; // Note: child_list is 1-indexed
                  int cm = fronts[ci].get_nrow();
                  int cn = fronts[ci].get_ncol();
                  int cbm = cm-cn;
                  int npassl = fronts[ci].get_npassl();
                  // Number of cols assembled into parent contrib block
                  int npasscb = cbm-npassl;
                  long ccbofs = lvlcbofs[ci]; // Ptr to contrib data for child node

                  if (npasscb > 0) {
                     // Some cols in child node are assembled into
                     // parent contrib block
                     cpdata[cpi].cm = npasscb; // Dimension of contrib block
                     cpdata[cpi].cn = npasscb; // Number of cols in cb going to parent's contrib block 
                     cpdata[cpi].ldp = m-n; // Parent cb leading dimn
                     // Note: row rlist(i) of parent is row
                     // rlist(i)-n of contribution blk so we alter
                     // pval to account for this
                     cpdata[cpi].pval = dev_cb_work + cbofs - n*(1+cpdata[cpi].ldp);
                     // cpdata[cpi].pval = &dev_cb_work[cbofs]; // Debug
                     cpdata[cpi].ldc = cbm; // Child cb leading dimn
                     cpdata[cpi].cvoffset = ccbofs + npassl*(1+cpdata[cpi].ldc);
                     // If gpu_contrib ..
                     cpdata[cpi].cv = &dev_cb_work_pre[cpdata[cpi].cvoffset];
                     cpdata[cpi].rlist_direct = &dev_rlist_direct[rptr[ci]+cn-1+npassl];
                     
                     cpdata[cpi].sync_offset = cpi - 1;
                     // cpdata[cpi].sync_offset =  std::max(0, cpi-1);
                     cpdata[cpi].sync_wait_for = blk;

                     int bx = (cpdata[cpi].cm-1) / HOGG_ASSEMBLE_TX + 1;
                     int by = (cpdata[cpi].cn-1) / HOGG_ASSEMBLE_TY + 1;
                     blk = calc_blks_lwr(bx, by, HOGG_ASSEMBLE_TX, HOGG_ASSEMBLE_TY);
                     cpi++;
                  }          
               }
            }
         }

         ////////////////////
         // blkdata

         std::vector<sylver::spldlt::gpu::assemble_blk_type> blkdata(nblk);

         // int bi = 0; // Block index
         // for (int c = 0; c < get_max_nch(); ++c) {
         //    cpi = 0;
         //    for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
         //       int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
         //       int cptr = child_ptr[ni]+c;

         //       if ( cptr < child_ptr[ni+1] ) {

         //          int ci = child_list[cptr-1]-1; // Note: child_list is 1-indexed
         //          int cm = fronts[ci].get_nrow();
         //          int cn = fronts[ci].get_ncol();
         //          int cbm = cm-cn;
         //          int npassl = fronts[ci].get_npassl();
         //          // Number of cols assembled into parent contrib block
         //          int npasscb = cbm-npassl;

         //          if (npasscb>0) {
         //             // Compute tile indexes in contrib block
         //             int bx = (cpdata[cpi+c].cm-1) / HOGG_ASSEMBLE_TX + 1;
         //             int by = (cpdata[cpi+c].cn-1) / HOGG_ASSEMBLE_TY + 1;

         //             for (int blkj = 0; blkj < by; ++blkj) {
         //                for (int blki = 0; blki < bx; ++blki) {

         //                   if ( ((blki+1)*HOGG_ASSEMBLE_TX) < ((blkj+1)*HOGG_ASSEMBLE_TY) )
         //                      continue;  // Entirely in upper triangle

         //                   blkdata[bi].cp = cpi + c;
         //                   blkdata[bi].blk = blkj*bx + blki;
         //                   bi++;
         //                }
         //             }
         //          }
         //       }
               
         //       cpi += child_ptr[ni+1]-child_ptr[ni];
         //    }
         // }
         
         int bi = 0; // Block index
         for (int c = 0; c < get_max_nch(); ++c) {

            cpi = 0;
            for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
               int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
               for (int cptr = child_ptr[ni]; cptr<child_ptr[ni+1]; ++cptr) {
                  int ci = child_list[cptr-1]-1; // Note: child_list is 1-indexed
                  int cm = fronts[ci].get_nrow();
                  int cn = fronts[ci].get_ncol();
                  int cbm = cm-cn;
                  int npassl = fronts[ci].get_npassl();
                  // Number of cols assembled into parent contrib block
                  int npasscb = cbm-npassl;

                  if (npasscb>0) {
                     if ( (cptr-child_ptr[ni]) == c) {

                        int bx = (cpdata[cpi].cm-1) / HOGG_ASSEMBLE_TX + 1;
                        int by = (cpdata[cpi].cn-1) / HOGG_ASSEMBLE_TY + 1;

                        for (int blkj = 0; blkj < by; ++blkj) {
                           for (int blki = 0; blki < bx; ++blki) {

                              if ( ((blki+1)*HOGG_ASSEMBLE_TX) < ((blkj+1)*HOGG_ASSEMBLE_TY) )
                                 continue;  // Entirely in upper triangle

                              blkdata[bi].cp = cpi;
                              blkdata[bi].blk = blkj*bx + blki;
                              bi++;
                           }
                        }
                        
                     }
                     cpi++;
                  }
               }
            }
         }
         
         // std::cout << context << " nblk = " << nblk << ", bi = " << bi << std::endl;
                  
         ////////////////////
         // Send data to the GPU
         
         cudaError_t cuerr;

         ////////////////////
         // cpdata

         sylver::gpu::StackAllocGPU<sylver::spldlt::gpu::assemble_cp_type<TDev>> assemble_cp_type_alloc(dev_stack_alloc_);         
         dev_cpdata = assemble_cp_type_alloc.allocate(ncp); // Allocate memory on GPU for cpdata
         cuerr = cudaMemcpyAsync(
               dev_cpdata, &cpdata[0],
               ncp*sizeof(sylver::spldlt::gpu::assemble_cp_type<TDev>),
               cudaMemcpyHostToDevice,
               custream);
         sylver::gpu::cuda_check_error(cuerr, context, "Failed to send cpdata to the GPU");

         ////////////////////
         // blkdata

         sylver::gpu::StackAllocGPU<sylver::spldlt::gpu::assemble_blk_type> assemble_blk_type_alloc(dev_stack_alloc_);
         dev_blkdata = assemble_blk_type_alloc.allocate(nblk); // Allocate memory on GPU for blkdata
         cuerr = cudaMemcpyAsync(
               dev_blkdata, &blkdata[0],
               nblk*sizeof(sylver::spldlt::gpu::assemble_blk_type),
               cudaMemcpyHostToDevice,
               custream);
         sylver::gpu::cuda_check_error(cuerr, context, "Failed to send blkdata to the GPU");         

         ////////////////////
         // sync

         // cuerr = cudaMalloc(
         //       (void**)&dev_sync,
         //       (lvl_nch_+1)*sizeof(unsigned int));
         // sylver::gpu::cuda_check_error(cuerr, context);

         sylver::gpu::StackAllocGPU<unsigned int> uint_alloc(dev_stack_alloc_);
         dev_sync = uint_alloc.allocate(ncp+1);

      }

      void assemble_contrib(
            cudaStream_t& custream,
            std::vector<sylver::spldlt::NumericFrontGPU<T, TDev>>& fronts,
            int /*const*/ * dev_rlist_direct, long const* rptr,
            TDev /*const*/ *dev_cb_work_pre, std::vector<long>& lvlcbofs,
            int const* child_ptr, int const* child_list,
            TDev *dev_cb_work_lvl // Contrib entries in current level
            ) {

         std::string context = "NumericLevelGPU::assemble_contrib";

         // Return if level has no children nodes
         if (get_nch() <= 0) return;

         // assemble_cp_type allocator
         sylver::gpu::StackAllocGPU<sylver::spldlt::gpu::assemble_cp_type<TDev>> assemble_cp_type_alloc(dev_stack_alloc_);
         // assemble_blk_type allocator
         sylver::gpu::StackAllocGPU<sylver::spldlt::gpu::assemble_blk_type> assemble_blk_type_alloc(dev_stack_alloc_);
         // int allocator
         sylver::gpu::StackAllocGPU<unsigned int> uint_alloc(dev_stack_alloc_);
         
         // cpdata on GPU
         sylver::spldlt::gpu::assemble_cp_type<TDev> *dev_cpdata = nullptr;
         int ncp = 0;
         // blkdata on GPU
         sylver::spldlt::gpu::assemble_blk_type *dev_blkdata = nullptr;
         int nblk = 0;
         // sync on GPU
         unsigned int *dev_sync = nullptr;

         // Setup data structures and send them to the GPU
         assemble_contrib_setup(
               custream, fronts,
               dev_rlist_direct, rptr,
               dev_cb_work_pre, lvlcbofs,
               child_ptr, child_list,
               dev_cb_work_lvl,
               dev_cpdata, ncp,
               dev_blkdata, nblk,
               dev_sync);

         // std::cout << context << " ncp = " << ncp << ", nblk = " << nblk << std::endl;
         
         cudaError_t cuerr;
         int blkoffset = 0;
         
         sylver::spldlt::gpu::assemble(
               custream, nblk, blkoffset,
               dev_blkdata, ncp, dev_cpdata,
               dev_cb_work_pre, dev_cb_work_lvl,
               dev_sync);
         // cuerr = cudaStreamSynchronize(custream);
         // sylver::gpu::cuda_check_error(cuerr, context, "Failed synchronize with CUDA stream");

         // Free data
         // cudaFree(dev_cpdata);
         // cudaFree(dev_blkdata);
         // cudaFree(dev_sync);
         assemble_cp_type_alloc.deallocate(dev_cpdata, ncp);
         assemble_blk_type_alloc.deallocate(dev_blkdata, nblk);
         // assemble_delay_type_alloc.deallocate(dev_ddata, ndblk);
         uint_alloc.deallocate(dev_sync, ncp+1);

         
      }
      
      /// @brief Setup assembly of fully summed entries
      void assemble_fully_summed_setup(
            cudaStream_t& custream,
            std::vector<sylver::spldlt::NumericFrontGPU<T, TDev>>& fronts,
            int /*const*/ * dev_rlist_direct, long const* rptr,
            TDev /*const*/ *dev_cb_work, std::vector<long>& lvlcbofs,
            int const* child_ptr, int const* child_list,
            sylver::spldlt::gpu::assemble_cp_type<TDev> *& dev_cpdata, 
            sylver::spldlt::gpu::assemble_blk_type *& dev_blkdata, int& nblk,
            sylver::spldlt::gpu::assemble_delay_type<TDev> *& dev_ddata, int& ndblk,
            unsigned int *& dev_sync
            ) {

         std::string context = "NumericLevelGPU::assemble_fully_summed_setup";

         // Return if level has no children nodes
         if (get_nch() <= 0) return;
         
         ////////////////////
         // cpdata

         // Initialize child-parent data, count number of blocks at each level
         
         // FIXME: Do we need cpdata for every children nodes? Could
         // there be some nodes without contrib into parent's L part
         // i.e. npassl = 0?

         std::vector<sylver::spldlt::gpu::assemble_cp_type<TDev>> cpdata(get_nch());

         nblk = 0; // Number of blocks
         int cpi = 0; // cpdata index

         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed

            int dev_ldl = fronts[ni].get_dev_ldl(); // Parent's leading dimn
            int ndelay_in = fronts[ni].ndelay_in; // Parents incoming delayed
            TDev *dev_lcol = fronts[ni].dev_lcol;
            TDev *dev_lcol_nd = &dev_lcol[ndelay_in*(1+dev_ldl)]; // Ptr to nondelayed part in front  
            int blk = 0; // Block index
            
            for (int cptr = child_ptr[ni]; cptr<child_ptr[ni+1]; ++cptr) {
               int ci = child_list[cptr-1]-1; // Note: child_list is 1-indexed

               int cdev_ldl = fronts[ci].get_dev_ldl(); // Parent's leading dimn
               long cbofs = lvlcbofs[ci]; // Ptr to contrib data in child node
               int cm = fronts[ci].get_nrow();
               int cn = fronts[ci].get_ncol();
               int cbm = cm-cn;
               int npassl = fronts[ci].get_npassl();
               
               // Fill-in cpdata structure
               cpdata[cpi].pval = dev_lcol_nd;
               cpdata[cpi].ldp = dev_ldl;
               cpdata[cpi].cm = cbm; // Dimension of contrib block
               cpdata[cpi].cn = npassl; // number of cols in cb going to parent's fully summed part 
               cpdata[cpi].ldc = cbm; // Children's contrib block leading dimn
               cpdata[cpi].cvoffset = cbofs;
               // TODO: If gpucontrib ..
               cpdata[cpi].cv = &dev_cb_work[cbofs];
               cpdata[cpi].rlist_direct = &dev_rlist_direct[rptr[ci]+cn-1];               

               cpdata[cpi].sync_offset = std::max(0, cpi-1);
               cpdata[cpi].sync_wait_for = blk;
               int bx = (cpdata[cpi].cm-1) / HOGG_ASSEMBLE_TX + 1;
               int by = (cpdata[cpi].cn-1) / HOGG_ASSEMBLE_TY + 1;
               blk = calc_blks_lwr(bx, by, HOGG_ASSEMBLE_TX, HOGG_ASSEMBLE_TY);
               nblk += blk;
               cpi++;
            }
         }

         ////////////////////
         // blkdata

         // Initialize blkdata
         std::vector<sylver::spldlt::gpu::assemble_blk_type> blkdata(nblk);

         // std::cout << context << ", max_ch = " << max_nch << std::endl;
         
         int bi = 0; // Block index
         for (int c = 0; c < get_max_nch(); ++c) {
            cpi = 0;
            for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
               int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed

               if ( (child_ptr[ni]+c) < child_ptr[ni+1]) {

                  
                  // Compute tile indexes in contrib block
                  int bx = (cpdata[cpi+c].cm-1) / HOGG_ASSEMBLE_TX + 1;
                  int by = (cpdata[cpi+c].cn-1) / HOGG_ASSEMBLE_TY + 1;

                  for (int blkj = 0; blkj < by; ++blkj) {
                     for (int blki = 0; blki < bx; ++blki) {

                        if ( ((blki+1)*HOGG_ASSEMBLE_TX) < ((blkj+1)*HOGG_ASSEMBLE_TY) )
                           continue;  // Entirely in upper triangle

                        blkdata[bi].cp = cpi + c;
                        blkdata[bi].blk = blkj*bx + blki;
                        bi++;
                     }
                  }
               }
               
               cpi += child_ptr[ni+1]-child_ptr[ni];
            }
         }

         ////////////////////
         // ddata

         // Initialize ddata (for copying in any delays)
         // std::vector<sylver::spldlt::gpu::assemble_delay_type<T>> ddata(lvl_nch_);

         ndblk = 0;

         // for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
         //    int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
         //    int ndelay_in = fronts[ni].ndelay_in; // Parents incoming delayed
         //    T *dev_lcol = fronts[ni].dev_lcol;
         //    int dev_ldl = fronts[ni].get_dev_ldl();
            
         //    int nd = 0; // Position to next delayed column
         //    for (int cptr = child_ptr[ni]; cptr<child_ptr[ni+1]; ++cptr) {

         //       int ci = child_list[cptr-1]-1;
         //       int cm = fronts[ci].get_nrow();
         //       int cn = fronts[ci].get_ncol();
         //       T *cdev_lcol = fronts[ci].dev_lcol;
         //       int cdev_ldl = fronts[ci].get_dev_ldl();
         //       int cndelay_in = fronts[ci].ndelay_in; // Parents incoming delayed
         //       int cnelim = fronts[ci].nelim;
         //       if ((cn+cndelay_in) <= cnelim) continue; // No delays from this child

         //       ddata[ndblk].ldd = dev_ldl; // Leading dimn of dest front
         //       ddata[ndblk].dskip = ndelay_in - nd;
         //       ddata[ndblk].m = cm - cnelim;
         //       ddata[ndblk].n = cn - cnelim;
         //       ddata[ndblk].lds = cdev_ldl; // Leading dimn of src front
         //       ddata[ndblk].dval = &dev_lcol[(1+dev_ldl)*nd]; // Ptr to delayed column destination
         //       ddata[ndblk].sval = &cdev_lcol[(1+cdev_ldl)*cnelim]; // // Ptr to delayed column source 
         //       nd += ddata[ndblk].n;
         //       ndblk++;
         //    }
         // }

         // std::cout << context << " nblk = " << nblk << ", ndblk = " << ndblk << std::endl; 

         ////////////////////
         // Send data to the GPU
         
         cudaError_t cuerr;

         ////////////////////
         // cpdata

         // cuerr = cudaMalloc(
         //       (void**)&dev_cpdata,
         //       lvl_nch_*sizeof(sylver::spldlt::gpu::assemble_cp_type<T>));
         // sylver::gpu::cuda_check_error(cuerr, context);

         // GPU memory stack allocator
         sylver::gpu::StackAllocGPU<sylver::spldlt::gpu::assemble_cp_type<TDev>> assemble_cp_type_alloc(dev_stack_alloc_);         
         dev_cpdata = assemble_cp_type_alloc.allocate(get_nch()); // Allocate memory on GPU for cpdata
         // std::cout << context << " lvl_nch = " << lvl_nch_ << ", dev_cpdata = " << dev_cpdata << std::endl;
         cuerr = cudaMemcpyAsync(
               dev_cpdata, &cpdata[0],
               get_nch()*sizeof(sylver::spldlt::gpu::assemble_cp_type<TDev>),
               cudaMemcpyHostToDevice,
               custream);
         sylver::gpu::cuda_check_error(cuerr, context, "Failed to send cpdata to the GPU");

         ////////////////////
         // blkdata

         // cuerr = cudaMalloc(
         //       (void**)&dev_blkdata,
         //       nblk*sizeof(sylver::spldlt::gpu::assemble_blk_type));
         // sylver::gpu::cuda_check_error(cuerr, context);

         // GPU memory stack allocator
         sylver::gpu::StackAllocGPU<sylver::spldlt::gpu::assemble_blk_type> assemble_blk_type_alloc(dev_stack_alloc_);
         dev_blkdata = assemble_blk_type_alloc.allocate(nblk); // Allocate memory on GPU for blkdata
         cuerr = cudaMemcpyAsync(
               dev_blkdata, &blkdata[0],
               nblk*sizeof(sylver::spldlt::gpu::assemble_blk_type),
               cudaMemcpyHostToDevice,
               custream);
         sylver::gpu::cuda_check_error(cuerr, context, "Failed to send blkdata to the GPU");

         ////////////////////
         // ddata
         // sylver::gpu::StackAllocGPU<sylver::spldlt::gpu::assemble_delay_type<T>> assemble_delay_type_alloc(dev_stack_alloc_);
         // dev_ddata = assemble_delay_type_alloc.allocate(ndblk);
         // cuerr = cudaMemcpyAsync(
         //       dev_ddata, &ddata[0],
         //       ndblk*sizeof(sylver::spldlt::gpu::assemble_delay_type<T>),
         //       cudaMemcpyHostToDevice,
         //       custream);
         // sylver::gpu::cuda_check_error(cuerr, context, "Failed to send ddata to the GPU");

         ////////////////////
         // sync

         // cuerr = cudaMalloc(
         //       (void**)&dev_sync,
         //       (lvl_nch_+1)*sizeof(unsigned int));
         // sylver::gpu::cuda_check_error(cuerr, context);

         sylver::gpu::StackAllocGPU<unsigned int> uint_alloc(dev_stack_alloc_);
         dev_sync = uint_alloc.allocate(get_nch()+1);
                  
      }

      /// @brief Assemble fully summed entries: basic implementation
      /// processing each matrix separatetly in level 

      void assemble_fully_summed(
            cudaStream_t& custream,
            std::vector<sylver::spldlt::NumericFrontGPU<T, TDev>>& fronts,
            int *dev_rlist_direct, long const* rptr,
            TDev *dev_cb_work, std::vector<long>& lvlcbofs,
            int const* child_ptr, int const* child_list
            ) {

         std::string context = "NumericLevelGPU::assemble_fully_summed";

         // Return if level has no children nodes
         if (get_nch() <= 0) return;

         cudaError_t cuerr;

         // Allocators

         // assemble_cp_type allocator
         sylver::gpu::StackAllocGPU<sylver::spldlt::gpu::assemble_cp_type<TDev>> assemble_cp_type_alloc(dev_stack_alloc_);
         // assemble_blk_type allocator
         sylver::gpu::StackAllocGPU<sylver::spldlt::gpu::assemble_blk_type> assemble_blk_type_alloc(dev_stack_alloc_);
         // assemble_delay_type allocator
         sylver::gpu::StackAllocGPU<sylver::spldlt::gpu::assemble_delay_type<TDev>> assemble_delay_type_alloc(dev_stack_alloc_);
         // int allocator
         sylver::gpu::StackAllocGPU<unsigned int> uint_alloc(dev_stack_alloc_);
         
         // cpdata on GPU
         sylver::spldlt::gpu::assemble_cp_type<TDev> *dev_cpdata = nullptr;
         // blkdata on GPU
         sylver::spldlt::gpu::assemble_blk_type *dev_blkdata = nullptr;
         int nblk = 0;
         // ddata on GPU
         sylver::spldlt::gpu::assemble_delay_type<TDev> *dev_ddata = nullptr;
         int ndblk = 0;
         // sync on GPU
         unsigned int *dev_sync = nullptr;
         
         // Setup data structures and send them to the GPU
         assemble_fully_summed_setup(
               custream, fronts,
               dev_rlist_direct, rptr,
               dev_cb_work, lvlcbofs,
               child_ptr, child_list,
               dev_cpdata,
               dev_blkdata, nblk,
               dev_ddata, ndblk,
               dev_sync);
         
         // std::cout << context << " dev_cpdata = " << dev_cpdata << ", dev_blkdata = " << dev_blkdata
         //           << ", dev_sync = " << dev_sync
         //           << std::endl; 
         // std::cout << context << " nblk = " << nblk << ", ndblk = " << ndblk << std::endl; 
         
         
         int blkoffset = 0;
         
         sylver::spldlt::gpu::assemble(
               custream, nblk, blkoffset,
               dev_blkdata, get_nch(), dev_cpdata,
               dev_cb_work, dev_lcol,
               dev_sync);
         // cuerr = cudaStreamSynchronize(custream);
         // sylver::gpu::cuda_check_error(cuerr, context, "Failed synchronize with CUDA stream");

         // Assemble delays
         //..
         
         // Free data
         // cudaFree(dev_cpdata);
         // cudaFree(dev_blkdata);
         // cudaFree(dev_sync);
         assemble_cp_type_alloc.deallocate(dev_cpdata, get_nch());
         assemble_blk_type_alloc.deallocate(dev_blkdata, nblk);
         // assemble_delay_type_alloc.deallocate(dev_ddata, ndblk);
         uint_alloc.deallocate(dev_sync, get_nch()+1);

      }
      
      /// @brief Assemble fully summed entries: basic implementation
      /// processing each matrix separatetly in level 
      void assemble_fully_summed_basic(
            cudaStream_t& custream,
            std::vector<sylver::spldlt::NumericFrontGPU<T, TDev>>& fronts,
            int const* dev_rlist_direct, long const* rptr,
            TDev const* dev_cb_work, std::vector<long>& lvlcbofs,
            int const* child_ptr, int const* child_list
            ) {

         // spral_ssids_assemble_fully_summed(
         //       &custream_, nnodes, get_nch(), get_lvl(), &lvlptr_[0], &lvllist_[0],
         //       nodes, dev_cb_work_pre, &gpu_contribs[0],
         //       level.dev_lcol, symb_.dev_rlist_direct, symb_.child_ptr,
         //       symb_.child_list, &lvlcbofs[0], asminf, rptr, sptr,
         //       gpu_work_alloc, &info);

         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
            
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
            int dev_ldl = fronts[ni].get_dev_ldl();
            int m = fronts[ni].get_nrow();
            int n = fronts[ni].get_ncol();
            int ndelay_in = fronts[ni].ndelay_in;
            TDev *dev_lcol = &fronts[ni].dev_lcol[ndelay_in*(1+dev_ldl)]; // Delayed column are put at the begining

            int nch = child_ptr[ni+1] - child_ptr[ni];
            if (nch == 0) continue; 

            for (int cp = child_ptr[ni]; cp<child_ptr[ni+1]; ++cp) {

               int ci = child_list[cp-1]-1;
               int cm = fronts[ci].get_nrow();
               int cn = fronts[ci].get_ncol();
               int cbsz = cm-cn;
               
               // std::cout << "ci = " << ci << ", cm = " << cm << ", cn = " << cn << std::endl;
               
               long cbofs = lvlcbofs[ci];
               TDev const* dev_cb = &dev_cb_work[cbofs];
               int const* dev_crlist_direct = &dev_rlist_direct[rptr[ci]+cn-1];
               
               sylver::spldlt::gpu::assemble_fully_summed(
                     custream, cm, cn, dev_crlist_direct, dev_cb, cbsz,
                     m, n, dev_lcol, dev_ldl);

            }
         } 
      }
         
      /// @brief Init lcol with value from original matrix A
      void init_lcol_with_a(
            cudaStream_t& custream,
            std::vector<sylver::spldlt::NumericFrontGPU<T, TDev>>& fronts,
            long const* nptr, long const* rptr, long const* dev_nlist, int const* dev_rlist,
            TDev const* dev_aval, TDev const* dev_scaling) {

         assert(lvlsz_ > 0);
         if (lvlsz_ <= 0) return;
            
         // int info;
         // spral_ssids_init_l_with_a(
         //       &custream, nnodes, lvl_, &lvlptr_[0], &lvllist_[0], nodes,
         //       lvl_nnodes_, lvlsz_, nptr, rptr, dev_nlist, dev_rlist, dev_aval,
         //       dev_lcol, gpu_work_alloc, &info, dev_scaling);

         std::string context = "NumericLevelGPU::init_lcol_with_a";
         std::vector<sylver::spldlt::gpu::load_nodes_type<TDev>> lndata(lvl_nnodes_);

         // std::cout << context << "sizeof TDev=" << sizeof(TDev) << ", sizeof T=" << sizeof(T) << std::endl;
         
         typename std::vector<sylver::spldlt::gpu::load_nodes_type<TDev>>
            ::iterator ln = lndata.begin();
         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
            auto dev_ldl = fronts[ni].get_dev_ldl();
            auto ndelay_in = fronts[ni].ndelay_in;
            // Fill lndata for current node
            ln->offn = nptr[ni] - 1; // nptr is Fortran-indexed
            ln->nnz = nptr[ni+1] - nptr[ni];
            ln->lda = rptr[ni+1] - rptr[ni];
            ln->offr = rptr[ni] - 1; // rptr is Fortran-indexed
            ln->ldl = dev_ldl;
            ln->lcol = &fronts[ni].dev_lcol[ndelay_in*(1+dev_ldl)]; // Delayed column are put at the begining
            // Iterate over lndata
            ++ln;
         }

         // GPU memory stack allocator
         sylver::gpu::StackAllocGPU<sylver::spldlt::gpu::load_nodes_type<TDev>> lndata_alloc(dev_stack_alloc_);
         sylver::spldlt::gpu::load_nodes_type<TDev> *dev_lndata = nullptr;
         cudaError_t cuerr;
         // cuerr = cudaMalloc(
         //       (void**)&dev_lndata,
         //       lvl_nnodes_*sizeof(sylver::spldlt::gpu::load_nodes_type<T>));
         // sylver::gpu::cuda_check_error(cuerr, context);
         dev_lndata = lndata_alloc.allocate(get_nnodes());
         // std::cout << context << "dev_lndata=" << dev_lndata << std::endl;
         cuerr = cudaMemcpyAsync(
               dev_lndata, &lndata[0],
               lvl_nnodes_*sizeof(sylver::spldlt::gpu::load_nodes_type<TDev>),
               cudaMemcpyHostToDevice,
               custream);
         sylver::gpu::cuda_check_error(cuerr, context, "Failed to send lndata to the GPU device");
         
         // Zero factors for this level
         cuerr = cudaMemsetAsync(dev_lcol, 0, lvlsz_*sizeof(TDev), custream);
         sylver::gpu::cuda_check_error(cuerr, context);
         
         // Init fronts on GPU for this level
         if (dev_scaling) {
            // Perform scaling
            sylver::spldlt::gpu::load_nodes_scale(
                  custream, lvl_nnodes_, dev_lndata, dev_nlist, dev_rlist,
                  dev_scaling, dev_aval);
         }
         else {
            // No scaling
            sylver::spldlt::gpu::load_nodes(
                  custream, lvl_nnodes_, dev_lndata, dev_nlist, dev_aval);
         }
         // cudaFree(dev_lndata);
         lndata_alloc.deallocate(dev_lndata, get_nnodes());
      }

      std::size_t get_workspace_size(
            std::vector<sylver::spldlt::NumericFrontGPU<T, TDev>> const& fronts,
            int const* child_ptr, int const* child_list) const {
         
         std::size_t worksz = 0;

         ////////////////////////////////////////
         // Init L with A 

         std::size_t lndata_sz = sylver::gpu::aligned_size(
               get_nnodes()*sizeof(sylver::spldlt::gpu::load_nodes_type<TDev>));

         worksz = std::max(worksz, lndata_sz);

         ////////////////////////////////////////
         // Space for cpdata, blkdata, ddata and sync in assemble_fully_summed()

         std::size_t cpdata_sz = 0;
         std::size_t blkdata_sz = 0;
         std::size_t ddata_sz = 0;
         std::size_t sync_sz = 0;
         
         // cpdata
         cpdata_sz = sylver::gpu::aligned_size(
               get_nch()*sizeof(sylver::spldlt::gpu::assemble_cp_type<TDev>));

         // blkdata
         int nblk = 0;
         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
            for (int cp = child_ptr[ni]; cp<child_ptr[ni+1]; ++cp) {
               int ci = child_list[cp-1]-1;
               int cm = fronts[ci].get_nrow();
               int cn = fronts[ci].get_ncol();
               int bx = (cm-1) / HOGG_ASSEMBLE_TX + 1;
               int by = (cn-1) / HOGG_ASSEMBLE_TY + 1;
               nblk += bx*by;
            }
         }

         blkdata_sz = sylver::gpu::aligned_size(
               nblk*sizeof(sylver::spldlt::gpu::assemble_blk_type));

         // ddata         
         ddata_sz = sylver::gpu::aligned_size(
               get_nch()*sizeof(sylver::spldlt::gpu::assemble_delay_type<TDev>));
         
         // sync
         sync_sz = sylver::gpu::aligned_size(
               (get_nch()+1)*sizeof(unsigned int));
         
         worksz = std::max(
               worksz,
               cpdata_sz + blkdata_sz + ddata_sz + sync_sz);

         ////////////////////////////////////////
         // Space for data in factor()

         std::size_t multisymm_sz = 0; // msymmdata

         multisymm_sz = sylver::gpu::aligned_size(
               get_nnodes()*sizeof(sylver::spldlt::gpu::multisymm_type<TDev>));
         
         worksz = std::max(
               worksz,
               multisymm_sz);

         std::size_t panel_sz = 0; // panel
         std::size_t mrdata_sz = 0; // mrdata
         std::size_t medata_sz = 0; // medata
         std::size_t mnfdata_sz = 0; // mnfdata
         std::size_t mbfdata_sz = 0; // mbfdata
         std::size_t stat_sz = 0; // stat
         std::size_t aux_sz = 0; // stat

         int ncbr = 0;
         int ncbe = 0;
         int ncb = 0;
         
         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed

            int nrow = fronts[ni].get_nrow();
            int ncol = fronts[ni].get_ncol();

            // Panel buffer
            panel_sz += nrow;

            // mrdata
            ncbr += (nrow-1) / (32*REO_BLOCK_SIZE) + 2;
            
            // medata
            ncbe += ((nrow - 1) / 32 + 1) * ((ncol - 1) / 32 + 1);

            // mbfdata
            ncb += (nrow-1) / (BLOCK_SIZE * (MCBLOCKS - 1)) + 1;
         }

         // Panel buffer
         panel_sz *= BLOCK_SIZE;
         panel_sz = sylver::gpu::aligned_size(panel_sz*sizeof(T));

         // mrdata
         mrdata_sz = sylver::gpu::aligned_size(ncbr*sizeof(sylver::spldlt::gpu::multireorder_data));

         // medata
         medata_sz = sylver::gpu::aligned_size(ncbe*sizeof(sylver::spldlt::gpu::multielm_data));

         // mnfdata
         mnfdata_sz = sylver::gpu::aligned_size(
               get_nnodes()*sizeof(sylver::spldlt::gpu::multinode_fact_type<TDev>));

         // mbfdata
         mbfdata_sz = sylver::gpu::aligned_size(
               ncb*sizeof(sylver::spldlt::gpu::multiblock_fact_type<TDev>));
         
         // stat: nnodes stat variables for multinode_factor() plus
         // one stat variable for factor()
         stat_sz = sylver::gpu::aligned_size(sizeof(int));
         stat_sz += sylver::gpu::aligned_size(get_nnodes()*sizeof(int));

         // aux
         aux_sz = sylver::gpu::aligned_size(8*sizeof(int));
         // std::cout << "aux_sz =" << aux_sz << std::endl; 
         
         worksz = std::max(
               worksz,
               panel_sz +
               mrdata_sz +
               medata_sz +
               mnfdata_sz +
               mbfdata_sz +
               stat_sz +
               aux_sz);
         
         ////////////////////////////////////////
         // Space for gpu_msdata in form_contrib()

         std::size_t msdata_sz = 0;
         ncb = 0;
         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
            int m = fronts[ni].get_nrow();
            int n = fronts[ni].get_ncol();
            int cbm = m-n;
            int k = (cbm-1)/32 + 1;
            ncb += (k*(k+1))/2;
         }

         msdata_sz = sylver::gpu::aligned_size(
               ncb*sizeof(sylver::spldlt::gpu::multisyrk_type<TDev>));
         worksz = std::max(worksz, msdata_sz);

         ////////////////////////////////////////
         // Space for cpdata, blkdata and sync in assemble_contrib()

         nblk = 0;
         for (int p = lvlptr_[lvl_]; p<lvlptr_[lvl_+1]; ++p) {
            int ni = lvllist_[p-1]-1; // Note: lvllist is 1-indexed
            for (int cp = child_ptr[ni]; cp<child_ptr[ni+1]; ++cp) {
               int ci = child_list[cp-1]-1;
               int cm = fronts[ci].get_nrow();
               int cn = fronts[ci].get_ncol();
               int cbm = cm-cn;
               int npassl = fronts[ci].get_npassl();

               // Number of cols assembled into parent contrib block
               int npasscb = cbm-npassl; 
                  
               if (npasscb <= 0)
                  continue; // No contribution into parent's contrib block

               int bx = (npasscb-1) / HOGG_ASSEMBLE_TX + 1;
               int by = (npasscb-1) / HOGG_ASSEMBLE_TY + 1;

               nblk += bx*by;
            }
         }

         blkdata_sz = sylver::gpu::aligned_size(
               nblk*sizeof(sylver::spldlt::gpu::assemble_blk_type));

         worksz = std::max(
               worksz,
               cpdata_sz + blkdata_sz + sync_sz);
         
         return worksz;
      }
      
      /// @brief Return level index
      int get_lvl() const { return lvl_; }
      /// @brief Return number of children nodes
      int get_nch() const { return lvl_nch_; } 
      /// @brief Return number of nodes in level
      int get_nnodes() const { return lvl_nnodes_; } 
      /// @brief Return the number of (fully-summed) factor entries in
      // level
      long get_lvlsz() const { return lvlsz_; }
      /// @brief Return the maximum number of children per node
      int get_max_nch() const { return max_nch_; }
      
   public:
      TDev *dev_lcol; // Factor entries on GPU 
   private:
      int lvl_; // Level index
      std::vector<int> const& lvlptr_;
      std::vector<int> const& lvllist_;
      int lvl_nch_; // Number of child nodes
      int max_nch_; // Maximum number of children per node
      int lvl_nnodes_; // Number of nodes in level
      long lvlsz_; // Number of factor entries in level
      sylver::gpu::StackAllocGPU<TDev> const& dev_stack_alloc_;
   };
   
}} // End of sylver::spldlt namespace
