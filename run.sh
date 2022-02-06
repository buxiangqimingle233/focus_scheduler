
# cat /dev/null > nohup.out

# search
# python focus.py -bm runfiles/pipeline.yaml -d 16 -f 1024 te
# python focus.py -bm runfiles/multi-model-1.yaml -d 16 -f 1024 te
# python focus.py -bm runfiles/multi-model-2.yaml -d 16 -f 1024 te
# python focus.py -bm runfiles/multi-model-3.yaml -d 16 -f 1024 te


# Generate traffic trace from focus toolchain
# python focus.py -bm runfiles/multi-model-1.yaml -d 16 -fr 1024-4096-512 d
# python focus.py -bm runfiles/multi-model-2.yaml -d 16 -fr 1024-4096-512 d
# python focus.py -bm runfiles/multi-model-3.yaml -d 16 -fr 1024-4096-512 d
# python focus.py -bm runfiles/pipeline.yaml -d 16 -fr 1024-4096-512 d



# Invoke the simulator in parallel and wait the sub-processes to finish
# python focus.py -bm runfiles/multi-model-2.yaml -d 16 s &
# python focus.py -bm runfiles/multi-model-1.yaml -d 16 s &
# python focus.py -bm runfiles/multi-model-3.yaml -d 16 s &
# python focus.py -bm runfiles/pipeline.yaml -d 16 s &
# wait


