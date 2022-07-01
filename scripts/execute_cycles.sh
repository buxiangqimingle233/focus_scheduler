#!/bin/bash
# Execution cycles with different array sizes

cat /dev/null > result.out
# test
for expr in {5..8}
do
{
    width=$[ 2 ** $expr ]
    for diameter in {4..9}
    do
    {
        python3 focus.py -bm benchmark/16_16.yaml -debug -d $diameter -b 8 -fr $width-$width-512 ds > /dev/null 2>>result.out
        # echo "batch: $batch, link width: $width" >> result.out
    } &
    done
    # python3 focus.py -bm benchmark/16_16.yaml -debug -d 16 -b 8 -fr $width-$width-512 ds > /dev/null 2>>result.out
} &
done
