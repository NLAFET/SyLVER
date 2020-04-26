#!/bin/bash

mkdir builds/sylver-starpu
cd builds/sylver-starpu
cmake ../.. -D SYLVER_RUNTIME=StarPU -D SYLVER_ENABLE_CUDA=OFF -D SYLVER_ENABLE_OMP=OFF -D SYLVER_SPRAL_USE_INTERNAL=ON \
      -D SYLVER_BUILD_UNIT_TESTS=ON
# make
RESULT=$?
[ $RESULT -ne 0 ] && exit 1

exit 0
