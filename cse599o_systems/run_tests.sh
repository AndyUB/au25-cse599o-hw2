#!/bin/bash

TEST_DIR=../tests
LOG_DIR=../test_results
rm -rf $LOG_DIR
mkdir -p $LOG_DIR

for i in {1..5}; do
    uv run pytest $TEST_DIR/test_ddp_individual_parameters.py >$LOG_DIR/indy$i.log 2>&1
done

for i in {1..5}; do
    uv run pytest $TEST_DIR/test_ddp.py >$LOG_DIR/bucketed$i.log 2>&1
done

for i in {1..5}; do
    uv run pytest $TEST_DIR/test_sharded_optimizer.py >$LOG_DIR/sharding$i.log 2>&1
done
