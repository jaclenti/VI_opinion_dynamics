#!/bin/bash
cliopts="$@"	# optional parameters
max_procs=8    # parallelism
timeout=21600	# timeout in seconds
T="128 512 2048 8192"
N="50 100 200 400"
date="240430_2flow"
initial_leaders_ratio="5 10 25 50 100"
method="svinormal sviNF mcmc abc"

rep="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"

parallel --timeout ${timeout} --ungroup --max-procs ${max_procs} "python inference_leaders.py {1} {2} {3} {4} {5} {6}" ::: $rep ::: $T ::: $N ::: $initial_leaders_ratio ::: $method ::: $date