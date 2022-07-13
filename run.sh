#!/bin/bash
# cat /dev/null > nohup.out

python3 focus.py -bm benchmark/gpt2-xl.yaml -graph --graph_name ./op_graph_output/gpt2-xl-1024.gpickle -d 128 -b 8 -fr 1024-1024-1024 ds

# python3 compiler/graph_analyzer2.py --op_file ./op_graph_output/dall-e-1024.gpickle --diameter 128 --reticle_size 16 --reticle_cycle 5 

#if you want to verify the result with the simulator, run the following command only
# python3 focus.py -bm benchmark/mobilenet_v3_large_8.yaml --graph_name ./op_graph_output/mobilenet_v3_large_8.gpickle -d 32 -b 1 -fr 1024-1024-1024 ds

# python3 focus.py -bm benchmark/mobilenet_v3_large_8.yaml -graph --graph_name ./op_graph_output/mobilenet_v3_large_8.gpickle -d 32 -b 2 -fr 2048-2048-1024 ds

# python3 compiler/graph_analyzer2.py -debug --op_file ./op_graph_output/mobilenet_v3_large_8.gpickle --diameter 32 --reticle_size 16 --reticle_cycle 5 