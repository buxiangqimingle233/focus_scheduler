
# test
# for expr in {3..6}
# do
# {
#     width=$[ 2 ** $expr ]
#     python3 focus.py -bm benchmark/4_4.yaml -d 5 -b 1 -fr $width-$width-512 teds > /dev/null 2>>result.out
#     python3 ./compiler/graph_analyzer.py > result.out

# } &
# done
# python3 focus.py -bm benchmark/5_5.yaml -d 5 -b 1 -fr 128-128-512 ds



diameter=$[8]
diameter2=$[20]

# python3 focus.py -bm benchmark/[$diameter]_$diameter.yaml -d $diameter2 -b 1 -fr 8-8-512 ds > /dev/null 2>>result.out
# python3 ./compiler/graph_analyzer.py >> result.out

# python3 focus.py -bm benchmark/[$diameter]_$diameter.yaml -d $diameter2 -b 1 -fr 16-16-512 ds > /dev/null 2>>result.out
# python3 ./compiler/graph_analyzer.py >> result.out

# python3 focus.py -bm benchmark/[$diameter]_$diameter.yaml -d $diameter2 -b 1 -fr 32-32-512 ds > /dev/null 2>>result.out
# python3 ./compiler/graph_analyzer.py >> result.out

# python3 focus.py -bm benchmark/[$diameter]_$diameter.yaml -d $diameter2 -b 1 -fr 64-64-512 ds > /dev/null 2>>result.out
# python3 ./compiler/graph_analyzer.py >> result.out

# python3 focus.py -bm benchmark/[$diameter]_$diameter.yaml -d $diameter2 -b 1 -fr 128-128-512 ds > /dev/null 2>>result.out
# python3 ./compiler/graph_analyzer.py >> result.out



python3 focus.py -bm benchmark/16_16.yaml -d 20 -b 1 -fr 8-8-512 ds > /dev/null 2>>result.out
python3 ./compiler/graph_analyzer.py >> result.out

python3 focus.py -bm benchmark/16_16.yaml -d 20 -b 1 -fr 16-16-512 ds > /dev/null 2>>result.out
python3 ./compiler/graph_analyzer.py >> result.out

python3 focus.py -bm benchmark/16_16.yaml -d 20 -b 1 -fr 32-32-512 ds > /dev/null 2>>result.out
python3 ./compiler/graph_analyzer.py >> result.out

python3 focus.py -bm benchmark/16_16.yaml -d 20 -b 1 -fr 64-64-512 ds > /dev/null 2>>result.out
python3 ./compiler/graph_analyzer.py >> result.out

python3 focus.py -bm benchmark/16_16.yaml -d 20 -b 1 -fr 128-128-512 ds > /dev/null 2>>result.out
python3 ./compiler/graph_analyzer.py >> result.out