
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
# python focus.py -bm benchmark/multi-model-1.yaml -d 16 d
# python focus.py -bm benchmark/multi-model-2.yaml -d 16 d
# python focus.py -bm benchmark/multi-model-3.yaml -d 16 d
# python focus.py -bm benchmark/pipeline.yaml -d 18 d
# wait


# Invoke the focus-scheduler
# python focus.py -bm benchmark/multi-model-1.yaml -d 16 s

# python focus.py -bm benchmark/test.yaml -d 10 -b 1 d

# ulimit -u 30

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

