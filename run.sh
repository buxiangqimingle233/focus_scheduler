#!/bin/bash
# cat /dev/null > nohup.out

# python focus.py -bm benchmark/bert_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ted
# python focus.py -bm benchmark/bert_large_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ted
# python focus.py -bm benchmark/mobilenet_v3_large_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ted
# python focus.py -bm benchmark/resnet50_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ted
# python focus.py -bm benchmark/wide_resnet50_2_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ted
# python focus.py -bm benchmark/vgg16_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ted

# python focus.py -bm benchmark/bert_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ds &
# python focus.py -bm benchmark/bert_large_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ds & 
# python focus.py -bm benchmark/mobilenet_v3_large_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ds &
# python focus.py -bm benchmark/resnet50_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ds &
# python focus.py -bm benchmark/wide_resnet50_2_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ds &
# python focus.py -bm benchmark/vgg16_8.yaml -d 32 -b 1 -fr 1024-1024-1024 ds &

# python focus.py -bm benchmark/bert_8.yaml -d 32 -b 1 -fr 512-512-1024 ds &
# python focus.py -bm benchmark/bert_large_8.yaml -d 32 -b 1 -fr 512-512-1024 ds & 
# python focus.py -bm benchmark/mobilenet_v3_large_8.yaml -d 32 -b 1 -fr 512-512-1024 ds &
# python focus.py -bm benchmark/resnet50_8.yaml -d 32 -b 1 -fr 512-512-1024 ds &
# python focus.py -bm benchmark/wide_resnet50_2_8.yaml -d 32 -b 1 -fr 512-512-1024 ds &
# python focus.py -bm benchmark/vgg16_8.yaml -d 32 -b 1 -fr 512-512-1024 ds &

python focus.py -bm benchmark/bert_8.yaml -d 32 -b 1 -fr 512-512-1024 ds &
python focus.py -bm benchmark/mobilenet_v3_large_8.yaml -d 32 -b 1 -fr 512-512-1024 ds &
python focus.py -bm benchmark/resnet50_8.yaml -d 32 -b 1 -fr 512-512-1024 ds &
python focus.py -bm benchmark/wide_resnet50_2_8.yaml -d 32 -b 1 -fr 512-512-1024 ds &

wait
