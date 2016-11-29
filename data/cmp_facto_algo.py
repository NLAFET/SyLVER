#!/usr/bin/env python

import sys
import re

ftime = 'Factor took'

# Directiry containing the experimental results
outputdir = sys.argv[1]
# Matrix list
listmat = sys.argv[2]
flistmat = open(listmat)

# Counter
matcount = 1

# Number of CPUs (parallel mode)
ncpus = 27

## StarPU
# Scheduler
starpu_sched = 'lws'

# Blocksizes
blocksizes = [256, 384, 512, 768, 1024]

for mat in flistmat:

    # Get matrix name
    mat = mat.rstrip()
    pbl = re.sub(r'/', '_', mat)
    pbl = pbl.rstrip()

    # Supernodal variant
    sn_tf = []

    # Multifrontal variant
    mf_tf = []

    matcount = matcount+1 
