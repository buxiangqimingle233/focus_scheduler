#!/bin/bash
# cat /dev/null > nohup.out

python focus.py -bm benchmark/multi-model-1.yaml -d 32 -b 1 -fr 512-512-1024 ted
python focus.py -bm benchmark/multi-model-2.yaml -d 32 -b 1 -fr 512-512-1024 ted
python focus.py -bm benchmark/multi-model-3.yaml -d 32 -b 1 -fr 512-512-1024 ted
wait
