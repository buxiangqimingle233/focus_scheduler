#!/bin/bash  


python3 focus.py -bm benchmark/16_16.yaml -d 20 -b 8 -fr 128-128-512 teds 2>>result.out

python3 focus.py -bm benchmark/4_4.yaml -d 5 -b 1 -fr 1024-1024-512 teds 2>>result.out