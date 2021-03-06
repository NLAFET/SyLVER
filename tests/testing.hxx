#pragma once

// SyLVER
#include "common.hxx"

// STD
#include <cstring>
#include <cstdlib>
#include <iostream>

namespace sylver {
namespace tests {

   class Options {

   public:
      Options():
         ncpu(1), ngpu(0), m(256), n(256), k(256), nb(256), ib(256), posdef(false), 
         check(true), chol(false), diagdom(false), algo(sylver::tests::algo::SyLVER),
         usetc(true), prec(sylver::tests::prec::FP64), cond(1), itref(false), tol(1e-8),
         sched(Sched::LWS), u(0.01), small(1e-20), delays(false), singular(false)
      {}

      void parse_opts(int argc, char** argv) {
         
         for( int i = 1; i < argc; ++i ) {

            // Matrix properties
            if ((!strcmp("--nb", argv[i]) || !strcmp("-nb", argv[i])) &&
                 i+1 < argc ) {
               nb = std::atoi( argv[++i] );
            }
            else if ( !strcmp("--m", argv[i]) && i+1 < argc ) {
               m =  std::atoi( argv[++i] );
            }
            else if ( !strcmp("-m", argv[i]) && i+1 < argc ) {
               m =  std::atoi( argv[++i] );
            }
            else if ( !strcmp("--n", argv[i]) && i+1 < argc ) {
               n =  std::atoi( argv[++i] );
            }
            else if ( !strcmp("-n", argv[i]) && i+1 < argc ) {
               n =  std::atoi( argv[++i] );
            }
            else if ( !strcmp("--k", argv[i]) && i+1 < argc ) {
               k =  std::atoi( argv[++i] );
            }
            else if ( !strcmp("-k", argv[i]) && i+1 < argc ) {
               k =  std::atoi( argv[++i] );
            }
            else if ( !strcmp("--cond", argv[i]) && i+1 < argc ) {
               cond =  std::atoi( argv[++i] );
            }
            // Number of workers
            
            // Number of CPU
            else if ((!strcmp("--ncpu", argv[i]) || !strcmp("-ncpu", argv[i])) &&
                     i+1 < argc ) {
               ncpu =  std::atoi( argv[++i] );
            }
            // Number of GPU
            else if ((!strcmp("--ngpu", argv[i]) || !strcmp("-ngpu", argv[i])) &&
                      i+1 < argc )  {
               ngpu =  std::atoi( argv[++i] );
            }
            else if ( !strcmp("--disable-tc", argv[i]) ) {
               std::cout << "Tensor cores deactivated" << std::endl;
               usetc =  false;
            }
            else if ( !strcmp("--posdef", argv[i]) ) {
               posdef = true;
            }
            else if ( !strcmp("--no-check", argv[i]) ) {
               check = false;
            }

            // Factorization method

            // Use Cholesky factor
            else if ( !strcmp("--chol", argv[i]) ) {
               chol = true;
            }
            else if ( !strcmp("--diagdom", argv[i]) ) {
               diagdom = true;
            }

            // Select algorithm
            else if ( !strcmp("--algo=sylver", argv[i]) ) {
               algo = sylver::tests::algo::SyLVER;
            }
            else if ( !strcmp("--algo=sylver-hp", argv[i]) ) {
               algo = sylver::tests::algo::SyLVER_HP;
            }
            else if ( !strcmp("--algo=sylver-hp-c16", argv[i]) ) {
               algo = sylver::tests::algo::SyLVER_HP_C16;
            }
            else if ( !strcmp("--algo=sylver-hp-u32", argv[i]) ) {
               algo = sylver::tests::algo::SyLVER_HP_U32;
            }
            else if ( !strcmp("--algo=cusolver", argv[i]) ) {
               algo = sylver::tests::algo::cuSOLVER;
            }
            else if ( !strcmp("--algo=cusolver-hp", argv[i]) ) {
               algo = sylver::tests::algo::cuSOLVER_HP;
            }
            else if ( !strcmp("--algo=cutlass", argv[i]) ) {
               algo = sylver::tests::algo::CUTLASS;
            }
            else if ( !strcmp("--algo=cutlass-wmma-hp", argv[i]) ) {
               algo = sylver::tests::algo::CUTLASS_WMMA_HP;
            }

            // Working precision
            else if ( !strcmp("--fp16", argv[i]) ) {
               prec = sylver::tests::prec::FP16;
            }
            else if ( !strcmp("--fp32", argv[i]) ) {
               prec = sylver::tests::prec::FP32;
            }
            else if ( !strcmp("--fp64", argv[i]) ) {
               prec = sylver::tests::prec::FP64;
            }

            // Refinement
            else if ( !strcmp("--itref", argv[i]) ) {
               itref = true;
            }
            else if ( !strcmp("--tol", argv[i]) && i+1 < argc ) {
               tol =  std::atof( argv[++i] );
            }

            // Scheduler
            else if ( !strcmp("--sched=lws", argv[i]) ) {
               std::cout << "Using LWS (Locality Work Stealing) scheduler" << std::endl;
               sched = Sched::LWS;
            }
            else if ( !strcmp("--sched=hlws", argv[i]) ) {
               std::cout << "Using HLWS (Heterogeneous Locality Work Stealing) scheduler" << std::endl;
               sched = Sched::HLWS;
            }
            else if ( !strcmp("--sched=hp", argv[i]) ) {
               std::cout << "Using HP (Heteroprio) scheduler" << std::endl;
               sched = Sched::HP;
            }
            else if ( !strcmp("--sched=ws", argv[i]) ) {
               std::cout << "Using WS (Work Stealing) scheduler" << std::endl;
               sched = Sched::WS;
            }

            // Cause delays in the factorization
            else if ( !strcmp("--delays", argv[i]) ) {
               std::cout << "Cause delays" << std::endl;
               delays = true;
            }
            // Make matrix singular
            else if ( !strcmp("--sing", argv[i]) ) {
               std::cout << "Singular matrix" << std::endl;
               singular = true;
            }
            
            // Command error message 
            else {
               std::cout << "Unrecognized command " << i << std::endl;
            }
            
         }
         
      }

      int ncpu; // Number of CPUs
      int ngpu; // Number of GPUs
      int m; // no rows in matrix
      int n; // no columns in matrix
      int k;
      int nb; // block size
      int ib; // inner block size
      bool posdef;
      bool check;
      bool chol; // Use Cholesky factorization
      bool diagdom; // Diagonally dominant matrix (unsymmetric case)
      enum sylver::tests::algo algo; // Algorithm to use
      bool usetc; // Use tensor cores on GPU
      enum sylver::tests::prec prec; // Working precision
      int cond; // Condition number of test matrix
      // Iterative refinement
      bool itref; // Control the use of iterative refinement
      double tol; // Tolerence for iterative refinement
      // Scheduler
      Sched sched;
      double u;
      double small;
      bool delays;
      bool singular;
   };

}} // End of namespace sylver::tests
