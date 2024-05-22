#!/bin/bash

cliopts="$@"	# optional parameters
max_procs=8	    # parallelism
timeout=21600	# timeout in seconds
date="240515_2flows"
T="128 512 2048 8192"
max_f_possible="5 10 20 40 80"
method="svinormal sviNF mcmc abc"
rep="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"

parallel --timeout ${timeout} --ungroup --max-procs ${max_procs} "python inference_feed.py {1} {2} {3} {4} {5}" ::: $rep ::: $T ::: $max_f_possible ::: $method ::: $date