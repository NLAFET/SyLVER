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

## STF

By default the code is compiled in sequential mode but can also be
specified by using the following option for the configuration:

```bash
cmake -DRUNTIME=STF <path-to-source>

```

## StarPU

The parallel version of the code with the StarPU runtime system can be
obtained by using the following option for the configuration:

```bash
cmake -DRUNTIME=StarPU <path-to-source>

```
