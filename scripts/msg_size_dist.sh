#!/bin/bash

# Run 
for logf in {5..10}
do
{
    flit=$[ 2** $logf ]
    python3 focus.py -bm benchmark/multi-model-1.yaml -debug -d 8 -b 1 -fr $flit-$flit-512 d > /dev/null 2>fk.log
} &
done
