#!/bin/bash

set -x

export PATH=/usr/local/cuda-13.0/bin:$PATH
nsys status -e
# ./benchmark_optimized_ddp.sh tempura_final
# ./benchmark_bucketing.sh tempura_final
./benchmark_bucketing_nsys.sh tempura_nsys_final
./benchmark_optimized_ddp_nsys.sh tempura_nsys_final
# ./benchmark_optimized_bucketing.sh tempura_optimized_final
./run_tests.sh
