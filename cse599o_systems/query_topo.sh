#!/bin/bash

OUTPUT_DIR=../bench/allreduce/tempura
mkdir -p $OUTPUT_DIR

nvidia-smi topo -m > $OUTPUT_DIR/topo.tbl
nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv > $OUTPUT_DIR/links.csv
