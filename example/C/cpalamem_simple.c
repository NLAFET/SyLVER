#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <spldlt_iface.h>

#define CPALAMEM_DEV
#include <cpalamem_macro.h>
#include <mat_csr.h>
#include <mat_load_mm.h>
#include <cpalamem_handler.h>
#include <cpalamem_instrumentation.h>

#define USAGE "Usage %s -m <matrixFileName> [--nrhs <integer DEFAULT:%d>] [--ncpu <integer DEFAULT:%d>] [--ngpu <integer DEFAULT:%d]"
#define FILENAMESIZE 256

int main(int argc, char ** argv){

  void *akeep = NULL;
  void *fkeep = NULL;
  long *ptr   = NULL;
  int *row    = NULL;
  double *val = NULL;
  double *x   = NULL;
  double *rhs = NULL;
  int n, nnz, nrhs = 1;
  spldlt_inform_t info;
  spldlt_options_t options;
  int stat;
  CPLM_Mat_CSR_t A      = CPLM_MatCSRNULL();
  CPLM_Mat_CSR_t U      = CPLM_MatCSRNULL();
  int ierr      = 0;
  int rank      = 0;
  int matrixSet = 0;
  int ncpu      = 1;
  int ngpu      = 0;
  int posdef    = 0;
  long size     = 0;
  char matrixFileName[FILENAMESIZE];

  CPLM_Init(&argc, &argv);
CPLM_PUSH
CPLM_BEGIN_TIME
CPLM_OPEN_TIMER

  //Handle parameters
  if(argc>0)
  {
    for(int i = 1; i < argc; i++)
    {
      if(!strcmp(argv[i],"-h"))
      {
        CPLM_Abort(USAGE, argv[0], nrhs, ncpu, ngpu);
      }
      else if(!strcmp(argv[i],"--ncpu"))
      {
        i++;
        ncpu = atoi(argv[i]);
      }
      else if(!strcmp(argv[i],"--ngpu"))
      {
        i++;
        ngpu = atoi(argv[i]);
      }
      else if(!strcmp(argv[i],"--nrhs"))
      {
        i++;
        nrhs = atoi(argv[i]);
      }
      else if(!strcmp(argv[i],"-m"))
      {
        i++;
        strcpy(matrixFileName, argv[i]);
        matrixSet = 1;
      }
      else{
        if(!rank)
        {
          CPLM_Abort(USAGE, argv[0], nrhs, ncpu, ngpu);
        }
      }
    }
  }

  if(!matrixSet)
  {
    CPLM_Abort("Error, you have to provide a matrix to factor");
  }

  ierr = CPLM_LoadMatrixMarket(matrixFileName, &A);CPLM_CHKERR(ierr);

  ierr = CPLM_MatCSRGetTriU(&A, &U);CPLM_CHKERR(ierr);
  CPLM_MatCSRConvertTo1BasedIndexing(&U);

  nnz = U.info.nnz;
  n   = U.info.n;
  ptr = malloc((n+1) * sizeof(long));
  for(unsigned int i = 0; i < n + 1; i ++)
    ptr[i] = U.rowPtr[i];
  row = U.colInd;
  val = U.val;

  //Create RHS
  x     = malloc(n * nrhs * sizeof(double));
  rhs   = malloc(n * nrhs * sizeof(double));
  for(int j = 0; j < nrhs; j++)
    for(int i = 0; i < n; i++) 
      rhs[i + j * nrhs] = 1.0 * (j + 1);
  memcpy(x, rhs, n * nrhs * sizeof(double));

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

  CPLM_MatCSRFree(&A);
  free(x);
  free(rhs);
CPLM_CLOSE_TIMER
CPLM_END_TIME
CPLM_POP
//CPLM_Finalize();
  return 0;
}
