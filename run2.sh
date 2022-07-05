#!/bin/bash
# cat /dev/null > nohup.out

# python focus.py -bm benchmark/bert_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ted
# python focus.py -bm benchmark/bert_large_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ted
# python focus.py -bm benchmark/resnet50_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ted
# wait

python focus.py -bm benchmark/bert_8.yaml -d 32 -b 1 -fr 1024-1024-1024 d &
python focus.py -bm benchmark/mobilenet_v3_large_8.yaml -d 32 -b 1 -fr 1024-1024-1024 d &
python focus.py -bm benchmark/resnet50_8.yaml -d 32 -b 1 -fr 1024-1024-1024 d &
python focus.py -bm benchmark/wide_resnet50_2_8.yaml -d 32 -b 1 -fr 1024-1024-1024 d &
python focus.py -bm benchmark/multi-model-1.yaml -d 32 -b 1 -fr 1024-1024-1024 d &
python focus.py -bm benchmark/multi-model-2.yaml -d 32 -b 1 -fr 1024-1024-1024 d &
python focus.py -bm benchmark/multi-model-3.yaml -d 32 -b 1 -fr 1024-1024-1024 d &
wait