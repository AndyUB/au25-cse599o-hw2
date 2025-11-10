#!/bin/bash

# Accept at most 1 arg
if [ "$#" -gt 2 ]; then
    echo "Usage: $0 [-d|-c|-cd]"
    exit 1
fi

# If -d (or -cd) is specified, debug with Gloo
if [ "$1" == "-d" ] || [ "$1" == "-cd" ]; then
    EXTRA_ARGS=(
        "--use_gloo"
    )
    DEVICE="cpu"
else
    EXTRA_ARGS=()
    DEVICE="gpu"
fi

# OUTPUT_DIR=../bench/allreduce
OUTPUT_DIR=../bench/allreduce/tempura
# If -c (or -cd) is specified, clean previous logs
if [ "$1" == "-c" ] || [ "$1" == "-cd" ]; then
    rm -rf $OUTPUT_DIR
fi
mkdir -p $OUTPUT_DIR

# Num processes: 2, 4, 8
# Tensor sizes: 1MB, 10MB, 100MB, 1GB
world_size_list=(2 4 8)
tensor_size_mb_list=(1 10 100 1000)

for world_size in "${world_size_list[@]}"; do
    for tensor_size_mb in "${tensor_size_mb_list[@]}"; do
        log_file=$OUTPUT_DIR/${tensor_size_mb}mb_${world_size}${DEVICE}.log
        uv run benchmark_allreduce.py \
            --tensor_size $tensor_size_mb \
            --world_size $world_size \
            --output_dir $OUTPUT_DIR \
            "${EXTRA_ARGS[@]}" > $log_file 2>&1
    done
done
