#!/bin/bash

make builds
make builds/starpu
cd builds/starpu
cmake ../.. -DRUNTIME=StarPU
RESULT=$?
[ $RESULT -ne 0 ] && exit 1

exit 0
