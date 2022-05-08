
# cat /dev/null > nohup.out

# search
# python focus.py -bm benchmark/pipeline.yaml -d 16 -f 1024 te
# python focus.py -bm benchmark/multi-model-1.yaml -d 16 -f 1024 te
# python focus.py -bm benchmark/multi-model-2.yaml -d 16 -f 1024 te
# python focus.py -bm benchmark/multi-model-3.yaml -d 16 -f 1024 te


# Generate traffic trace from focus toolchain
# python focus.py -bm benchmark/multi-model-1.yaml -d 16 -fr 1024-4096-512 d &
# python focus.py -bm benchmark/multi-model-2.yaml -d 16 -fr 1024-4096-512 d &
# python focus.py -bm benchmark/multi-model-3.yaml -d 16 -fr 1024-4096-512 d &
# python focus.py -bm benchmark/pipeline.yaml -d 16 -fr 1024-4096-512 d &
# wait


# Invoke the simulator
python focus.py -bm benchmark/multi-model-1.yaml -d 16 d
# python focus.py -bm benchmark/multi-model-2.yaml -d 16 d
# python focus.py -bm benchmark/multi-model-3.yaml -d 16 d
# python focus.py -bm benchmark/pipeline.yaml -d 18 d
# wait


# Invoke the focus-scheduler
# python focus.py -bm benchmark/multi-model-1.yaml -d 16 s

# test
python3 focus.py -bm benchmark/test.yaml -debug -d 4 d
