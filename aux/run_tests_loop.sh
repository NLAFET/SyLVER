#!/bin/bash

hostname=`hostname`
echo "[run_tests] compute node: $hostname"

# list loaded modules
module list

for matrix in `cat list.matrix`
do
    matname=`echo $matrix | sed -e 's/\//_/g'`
    echo "[run_tests] test matrix: $matname"
    # set up matrix
    echo "[run_tests] extract matrix: $matname"
    ./prep2.sh $matrix

    for ncpu in ${ncpu_list[@]}
    do
        echo "[run_tests] ncpu: $ncpu"

        for nb in ${nb_list[@]}
        do
            echo "[run_tests] nb: $nb"

            for nemin in ${nemin_list[@]}
            do  
                echo "[run_tests] nemin: $nemin"

                # ./run_ma87
                echo "[run_tests] run SPLDLT"
                rm -rf $trace_dir/$prof_file
                # just to make sure
                export OMP_NUM_THREADS=1
                case $build in
                    starpu)
                        ../builds/starpu/spldlt_test --posdef --ncpu ${ncpu} --nb ${nb} --nemin ${nemin} > $outdir/starpu/${matname}_NCPU-${ncpu}_NB-${nb}_NEMIN-${nemin}${outsuffix}
                        # ../builds/starpu/spldlt_test --posdef --ncpu ${ncpu} --nb ${nb} --nemin ${nemin}
                        ;;
                esac

                if [ -f $trace_dir/$prof_file ];
                then
                    mv $trace_dir/$prof_file $outdir/starpu/traces/${matname}_NCPU-${ncpu}_NB-${nb}.prof
                    starpu_fxt_tool -c -i $outdir/starpu/traces/${matname}_NCPU-${ncpu}_NB-${nb}.prof -o $outdir/starpu/traces/${matname}_NCPU-${ncpu}_NB-${nb}.trace
                fi
            done
        done
    done
done
