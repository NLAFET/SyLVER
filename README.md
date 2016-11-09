# SpLDLT

DAG-based sparse direct solver for indefinite symmetric systems
developed with a runtime system.

# Installation 

We use CMake tools for the compilation of this package. For example:

```bash
mkdir build # create build directory
cd build 
cmake <path-to-source> # configure compilation
make # run compilation 
```

# Runtime systems

By default the code is compiled in sequential mode but the choice of
the runtime system can be specified by setting the option
`-DRUNTIME`. For example, the sequential code can be configured as
following:

```bash
cmake -DRUNTIME=STF <path-to-source>

```

## StarPU

A parallel version of the code using the StarPU runtime system can be
obtained as following:

```bash
cmake -DRUNTIME=StarPU <path-to-source>

```
