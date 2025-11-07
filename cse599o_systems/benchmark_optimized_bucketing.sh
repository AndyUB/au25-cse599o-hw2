#!/bin/bash

# Accept hostname as the first argument
# If none provided, default to tempura
HOSTNAME=${1:-tempura}

export DDP_OPTIMIZE_SINGLETON_BUCKETS=1
./benchmark_bucketing.sh $HOSTNAME
