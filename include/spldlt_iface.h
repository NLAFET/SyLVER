!> \file
!> \copyright 2018 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Sebastien Cayrols
#ifndef SPLDLT_IFACE_H
#define SPLDLT_IFACE_H

//typedef struct{
//  void *akeep;
//  void *fkeep;
//} spldlt_data_t;

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

/************************************
 * Derived types
 ************************************/

typedef struct {
   int array_base; // Not in Fortran type
   int print_level;
   int unit_diagnostics;
   int unit_error;
   int unit_warning;
   int ordering;
   int nemin;
   bool ignore_numa;
   bool use_gpu;
   long min_gpu_work;
   float max_load_inbalance;
   float gpu_perf_coeff;
   int scaling;
   long small_subtree_threshold;
   int cpu_block_size;
   bool action;
   int pivot_method;
   double small;
   double u;
   char unused[80]; // Allow for future expansion
} spral_ssids_options_t; //Equiv spral_ssids_options

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
} spldlt_inform_t; // EQUIV spral_ssids_inform struct

typedef struct{
  spral_ssids_options_t options;
  int prune_tree;
} spldlt_options_t;

#define SPLDLT_OPTIONS_NULL(){.options={  \
                                .array_base=0,              \
                                .print_level=0,             \
                                .unit_diagnostics=0,        \
                                .unit_error=0,              \
                                .unit_warning=0,            \
                                .ordering=0,                \
                                .nemin=32,                  \
                                .ignore_numa=0,             \
                                .use_gpu=0,                 \
                                .min_gpu_work=0,            \
                                .max_load_inbalance=0,      \
                                .gpu_perf_coeff=0,          \
                                .scaling=0,                 \
                                .small_subtree_threshold=0, \
                                .cpu_block_size=32,         \
                                .action=0,                  \
                                .pivot_method=0,            \
                                .small=0.0,                 \
                                .u=0.0,                     \
                                },        \
                                .prune_tree=1}

extern void spldlt_analyse( int               n,
                            int               *ptr,
                            int               *row,
                            double            *val, //Optional here
                            void              **akeep,
                            spldlt_options_t  *options,
                            spldlt_inform_t   *info);

extern void spldlt_factor(int               posdef, //Boolean 
                          double            *val,
                          void              *akeep,
                          void              *fkeep,
                          spldlt_options_t  *options,
                          spldlt_inform_t   *info);

extern void spldlt_solve( int             job,
                          int             nrhs,
                          double          *x,
                          int             ldx,
                          void            *akeep,
                          void            *fkeep,
                          spldlt_inform_t *info);

extern int spllt_deallocate_akeep(void  **akeep);

extern int spllt_deallocate_fkeep(void  **fkeep);
                        
#endif
