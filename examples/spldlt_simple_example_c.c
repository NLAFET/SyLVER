#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "sylver/spldlt.h"

int main(int argc, char ** argv) {

  void *akeep = NULL;
  void *fkeep = NULL;
  long *ptr   = NULL;
  int *row    = NULL;
  double *val = NULL;
  int *order  = NULL;
  double *x   = NULL;
  double *rhs = NULL;
  double *y   = NULL;
  int n, nnz, nrhs;
  int ngpu, ncpu;
  int posdef;
  spldlt_inform_t info;
  spldlt_options_t options;
  int stat;

    // Create the matrix 
  // [  2 -1  0 ]
  // [ -1  2 -1 ]
  // [  0 -1  2 ]
  posdef  = 0;
  n       = 3;
  nnz     = 5;
  ptr = malloc((n+1) * sizeof(long));
  row = malloc(nnz * sizeof(int));
  val = malloc(nnz * sizeof(double));

  ptr[0] = 1; ptr[1] = 3; ptr[2] = 5; ptr[3] = 6;
  row[0] = 1; row[1] = 2; row[2] = 2; row[3] = 3; row[4] = 3;
  for(int i = 0; i < nnz; i++) val[i] = 2.0;
  val[1] = - 1.0; val[3] = - 1.0;

  order = malloc(n * sizeof(int));
  ncpu  = 1;
  ngpu  = 0;

  //Create RHS
  nrhs = 1;
  x     = malloc(n * nrhs * sizeof(double));
  rhs   = malloc(n * nrhs * sizeof(double));
  for(int i = 0; i < n; i++) rhs[i] = 1.0;
  memcpy(x, rhs, n * sizeof(double));

  spldlt_default_options(&options);
  options.options.ordering    = 1;//use Metis ordering
  options.options.scaling     = 0;//no scaling
  options.options.print_level = 1;//enable printing
  options.options.print_level = 0;//disable printing
  options.options.use_gpu     = 0;//disable GPU

  spldlt_analyse(n, ptr, row, val, ncpu, &akeep, &options, &info);

  //Initialize SpLDLT
  spldlt_init(ncpu, ngpu);

  spldlt_factor(posdef, val, akeep, &fkeep, &options, &info);

  spldlt_finalize();

  spldlt_solve(0, nrhs, x, n, akeep, fkeep, &info);

  spldlt_chkerr(n, ptr, row, val, nrhs, x, rhs);

  stat = spldlt_free_akeep(&akeep);

  stat = spldlt_free_fkeep(&fkeep);


  return 0;
}
