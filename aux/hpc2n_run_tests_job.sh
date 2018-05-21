#!/bin/bash
# Project/Account
#SBATCH -A SNIC2016-1-536
# Request exclusive access to the node
#SBATCH --exclusive
# One task
#SBATCH -n 1
# Number of cores per task
#SBATCH -c 28
# Runtime of this jobs is less then 10 hours.
#SBATCH --time=02:00:00
# Job name
#SBATCH -J spldlt_tests
# Ouput
#SBATCH --error=job.err
#SBATCH --output=job.out

echo "[HPC2N run_tests SpLDLT]"

# Choose build
# - starpu: SpLLT vesion StarPU (exploit tree prunning by default)
# - gnu_omp: SpLLT OpenMP version. GNU compiler 
# - intel_omp: SpLLT OpenMP version. Intel compiler 
# - parsec: PaRSEC version of SpLLT

build="starpu"

build_dir=`pwd`
id=`whoami`
outdir=../data
#outsuffix="_NOSUB"
outsuffix=

echo "[HPC2N run_tests SpLDLT] build dir: $build_dir"
#matrices=(JGD_Trefethen/Trefethen_20000)

mkdir -p $outdir

case $build in
    stf)
        mkdir -p $outdir/stf
        ;;
    parsec)
        mkdir -p $outdir/parsec
        mkdir -p $outdir/parsec/traces
        # make clean
        # make parsec
        ;;
    starpu)
        mkdir -p $outdir/starpu
        mkdir -p $outdir/starpu/traces
        ;;
    starpu_nested_stf)
        mkdir -p $outdir/starpu_nested_stf
        mkdir -p $outdir/starpu_nested_stf/traces
        ;;
    gnu_omp|gnu_omp_prune)
        mkdir -p $outdir/omp
        mkdir -p $outdir/omp/gnu
        mkdir -p $outdir/omp/gnu/traces
        ;;
    intel_omp|intel_omp_prune)
        mkdir -p $outdir/omp
        mkdir -p $outdir/omp/intel
        mkdir -p $outdir/omp/intel/traces
        ;;
    ma87)
        mkdir -p $outdir/ma87
        ;;
esac

ncpu_list=(27)
nb_list=(256 384 512 768 1024 1536)
nemin_list=(32)

. ./run_tests_loop.sh

# End of submit file
