# SyLVER

SyLVER is a sparse direct solver for symmetric systems which may be
positive-definite or indefinite. It implements DAG-based algorithms
that enable an efficient exploitation of multicore architectures and
heterogeneous GPU-accelerated systems. The code has been developed in
the context of the EU H2020 [NLAFET]((http://www.nlafet.eu/))
project. The parallel implementation relies on the
[StarPU](http://starpu.gforge.inria.fr/) runtime system developed and
maintained by the STORM team at Inria Bordeaux Sud-Ouest.

# Installation 

The compilation of the code is handled by the
[CMake](https://cmake.org/) tools. For example, the compilation can be
done as follow:

```bash
mkdir build # create build directory
cd build 
cmake <path-to-source> -DRUNTIME=StarPU # configure compilation
make # run compilation 
```

## Third-party libraries ##

### SPRAL ###

[SPRAL](https://github.com/ralna/spral) is an open-source library for
sparse linear algebra and associated algorithm and has numerous
important features used un SyLVER. The latest verion of SPRAL can be
found on its [GitHub
repository](https://github.com/ralna/spral/releases). The compilation
of SPRAL is handled by autotools and can be done as follow when using
the GCC compilers:

```bash
cd spral
mkdir build
cd build
../configure CXX=g++ FC=gfortran CC=gcc CFLAGS="-g -O2 -march=native" CXXFLAGS="-g -O2 -march=native" FCFLAGS="-g -O2 -march=native" --with-metis="-L/path/to/metis -lmetis" --with-blas="-L/path/to/blas -lblas" --with-lapack="-L/path/to/lapack -llapack" --disable-openmp --disable-gpu
make
```

Note that we use the `--disable-openmp` option because SyLVER works
with the serial version of SPRAL and in this example we disabled the
compilation of GPU kernels using the `--disable-gpu` option. Also,
note that **the compilation flags used for SPRAL must match the flags
used in the compilation of SyLVER**. Here we use the flags `-g -O2
-march=native` that corresponds to the `RelWithDebInfo` build type in
SyLVER.

**Sequential version** of BLAS and LAPACK should be used. We recommend
using the [MKL](https://software.intel.com/mkl) library for best
performance on Intel machines and
[ESSL](https://www.ibm.com/support/knowledgecenter/en/SSFHY8/essl_welcome.html)
on IBM machines. The [MKL link line
advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor)
can be useful to fill the `--with-blas` and `--with-lapack` options.

### MeTiS ###

The [MeTiS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)
partitioning library is needed by the SPRAL library and therefore it
is needed when linking the SyLVER package when generating examples and
test drivers.

### Runtime system ###

By default, CMake will confirgure the compilation for a serial version
of SyLVER that can be explicitly requested using the option
`-DRUNTIME=STF`.  The `-DRUNTIME=StarPU` option indicates that you
want to compile the parallel version of the code using StarPU in which
case the StarPU version needs to be at least 1.3.0. The StarPU library
is found with the `FindSTARPU.cmake` script located in the
`cmake/Modules` directory. For this script to be able to find the
StarPU library, you need to set the environment variable `STARPU_DIR`
to the path of you StarPU install base directory.

### BLAS and LAPACK libraries ###

The BLAS and LAPACK libraries play an important role in the
performance of the solver. We recommend using the
[MKL](https://software.intel.com/mkl) library for best performance on
Intel machines and the
[ESSL](https://www.ibm.com/support/knowledgecenter/en/SSFHY8/essl_welcome.html)
library when running on IBM machines. Alternative BLAS and LAPACK
libraries include [OpenBLAS](https://www.openblas.net/). Note that
SyLVER should be linked against the **sequential** BLAS and LAPACK
libraries.

These libraries are found via the CMake scripts
[FindBLAS](https://cmake.org/cmake/help/latest/module/FindBLAS.html)
and
[FindLAPACK](https://cmake.org/cmake/help/latest/module/FindBLAS.html)
and therefore it is possible to use the options `-DBLA_VENDOR` to
indicate which libraries to use. For example:

```bash
cmake <path-to-source> -DBLA_VENDOR=Intel10_64lp_seq # configure compilation
```

selects and locates the sequential BLAS and LAPACK implementation for
the compilation and when linking test drivers, example and tests.

If CMake is unable to locate the requested libraries via the
`-DBLA_VENDOR`, it is still possible to give them explicitly using the
`-DLLBAS` and `-DLLAPACK` options. For example:

```bash
cmake <path-to-source> -DLBLAS="/path/to/blas -lblas" -DLBLAS="/path/to/lapack -llapack" # configure compilation
```
