#pragma once

#include <cstring>
#include <cstdlib>

namespace spldlt {

   class SpldltOpts {

   public:
      SpldltOpts():
         ncpu(1), m(256), n(256), k(256), nb(256), ib(256), posdef(false), 
         check(true) {}

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
            else if ( !strcmp("--ncpu", argv[i]) && i+1 < argc ) {
               ncpu =  std::atoi( argv[++i] );
            }
            else if ( !strcmp("-ncpu", argv[i]) && i+1 < argc ) {
               ncpu =  std::atoi( argv[++i] );
            }
            else if ( !strcmp("--posdef", argv[i]) ) {
               posdef = true;
               ++i;
            }
            else if ( !strcmp("--no-check", argv[i]) ) {
               check = false;
               ++i;
            }

         }
         
      }

      int ncpu; // no cpu
      int m; // no rows in matrix
      int n; // no columns in matrix
      int k;
      int nb; // block size
      int ib; // inner block size
      bool posdef;
      bool check;
   };
}
