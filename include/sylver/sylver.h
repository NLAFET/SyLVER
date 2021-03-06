//!> \file
//!> \copyright 2018 The Science and Technology Facilities Council (STFC)
//!> \licence   BSD licence, see LICENCE file for details
//!> \author    Sebastien Cayrols
#ifndef SPLDLT_IFACE_H
#define SPLDLT_IFACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>


/************************************
 * Derived types
 ************************************/

typedef struct {
   int flag;
   int matrix_dup;
   int matrix_missing_diag;
   int matrix_outrange;
   int matrix_rank;
   int maxdepth;
   int maxfront;
   int num_delay;
   long num_factor;
   long num_flops;
   int num_neg;
   int num_sup;
   int num_two;
   int stat;
   int cuda_error;
   int cublas_error;
   char unused[80]; // Allow for future expansion
} sylver_inform_t; // EQUIV spral_ssids_inform struct

typedef struct {
   int array_base;
   /* Printing options */  
   int print_level;
   int unit_diagnostics; 
   int unit_error;
   int unit_warning;
   /* Options used in spldlt_analyse() and splu_analyse() */
   int ordering;
   int nemin;
   /* Tree partitioning */
   bool prune_tree;
   long min_gpu_work;
   /* Options used by spldlt_factor() and splu_factor() */
   int scaling;
   /* Options used by spldlt_factor() and splu_factor() for
      controlling pivoting */
   int pivot_method;
   double small;
   double u;
   /* CPU-specific */
   long small_subtree_threshold;
   int nb;
   int cpu_topology;
   /* Options used by spldlt_factorize() with posdef=.false. */
   bool action;
   /* GPU-specific */
   bool use_gpu;
   double gpu_perf_coeff;
   /* Undocumented */
   int failed_pivot_method;
   int scheduler;
} sylver_options_t;

extern void sylver_init(int ncpu, int ngpu);

extern void sylver_finalize();

extern void sylver_default_options(sylver_options_t *options);

extern void spldlt_analyse(
      int n, int *order,
      long const* ptr, int const* row, double const* val,
      void **akeep, bool check,
      sylver_options_t const* options, sylver_inform_t *inform);
   
//extern void spldlt_analyse_d( int               n,
//                              long              *ptr,
//                              int               *row,
//                              double            *val, //Optional here
//                              int               ncpu,
//                              void              **akeep,
//                              spldlt_options_t  *options,
//                              spldlt_inform_t   *info,
//                              int               dumpMat);//MPI rank, othewise -1

extern void spldlt_factorize(
      bool posdef, long const* ptr, int const* row, double const* val,
      double *scale, void *akeep, void **fkeep,
      sylver_options_t const* options, sylver_inform_t *inform);

extern void spldlt_solve(
      int job, int nrhs, double *x, int ldx, void *akeep, void *fkeep,
      sylver_options_t const* options, sylver_inform_t *inform);

//extern void spldlt_solve_d( int             job,
//                            int             nrhs,
//                            double          *x,
//                            int             ldx,
//                            void            *akeep,
//                            void            *fkeep,
//                            spldlt_inform_t *info,
//                            int             dumpRhs);//MPI rank, othewise -1

/* extern void spldlt_chkerr(int n, */
/*                           long *ptr, */
/*                           int *row, */
/*                           double *val, */
/*                           int nrhs, */
/*                           double *x, */
/*                           double *rhs); */

extern void spldlt_free_akeep(void **akeep);

extern void spldlt_free_fkeep(void **fkeep);

#ifdef __cplusplus
}
#endif
                        
#endif
