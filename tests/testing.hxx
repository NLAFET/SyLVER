#pragma once

// SyLVER
#include "common.hxx"

// STD
#include <cstring>
#include <cstdlib>
#include <iostream>

namespace spldlt {

   class SpldltOpts {

   public:
      SpldltOpts():
         ncpu(1), ngpu(0), m(256), n(256), k(256), nb(256), ib(256), posdef(false), 
         check(true), chol(false), diagdom(false), algo(sylver::tests::algo::SyLVER),
         usetc(true)
      {}

      void parse_opts(int argc, char** argv) {
         
         for( int i = 1; i < argc; ++i ) {

            if ( !strcmp("--nb", argv[i]) && i+1 < argc ) {
               nb = std::atoi( argv[++i] );
            }
            else if ( !strcmp("-nb", argv[i]) && i+1 < argc ) {
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
            // Number of workers
            
            // Number of CPU
            else if ( !strcmp("--ncpu", argv[i]) && i+1 < argc ) {
               ncpu =  std::atoi( argv[++i] );
            }
            else if ( !strcmp("-ncpu", argv[i]) && i+1 < argc ) {
               ncpu =  std::atoi( argv[++i] );
            }
            // Number of GPU
            else if ( !strcmp("--ngpu", argv[i]) && i+1 < argc ) {
               ngpu =  std::atoi( argv[++i] );
            }
            else if ( !strcmp("-ngpu", argv[i]) && i+1 < argc ) {
               ngpu =  std::atoi( argv[++i] );
            }
            else if ( !strcmp("--notc", argv[i]) ) {
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
            else if ( !strcmp("--algo=cusolver", argv[i]) ) {
               algo = sylver::tests::algo::cuSOLVER;
            }
            else if ( !strcmp("--algo=cusolver-hp", argv[i]) ) {
               algo = sylver::tests::algo::cuSOLVER_HP;
            }
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
   };
}
