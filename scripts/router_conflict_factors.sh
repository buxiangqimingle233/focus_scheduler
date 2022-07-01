#!/bin/bash


for expr in {5..8}
do
{
    width=$[ 2 ** $expr ]
    python3 focus.py -bm benchmark/16_16.yaml -debug -d 8 -b 4 -fr $width-$width-512 ds > /dev/null 2>&1 &
    python3 focus.py -bm benchmark/8_8.yaml -debug -d 8 -b 4 -fr $width-$width-512 ds > /dev/null 2>&1 &
} &
done