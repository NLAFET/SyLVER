# SpLDLT

SpLDLT is a DAG-based sparse direct solver for symmetric systems which
can solve positive-definite and indefinite problems. The parallel
execution of the DAG which includes task scheduling, dependency
management and data coherency is handled by the
[StarPU](http://starpu.gforge.inria.fr/) runtime system.

# Installation 

We use [CMake](https://cmake.org/) tools for the compilation of this
package. The solver can be installed using the following instructions:

```bash
mkdir build # create build directory
cd build 
cmake <path-to-source> # configure compilation
make # run compilation 
```

# Runtime systems

By default the code is compiled without any runtime system and is
therefore sequential. A parallel version of the solver can be obtained
by specifying the runtime system using the option `-DRUNTIME` when
running the cmake `cmake <path-to-source>` command. For now, the two
options for setting the RUNTIME option are either `STF` which provide
a sequential version or `StarPU` which uses the StarPU runtime system
for generating the parallel code.

For example, the compilation files for installing the sequential code
can be setup as following:

```bash
cmake -DRUNTIME=STF <path-to-source>

```

## StarPU

The compilation of a parallel version of the solver using the StarPU
runtime system can be configured with the following instructions:

```bash
cmake -DRUNTIME=StarPU <path-to-source>

```
