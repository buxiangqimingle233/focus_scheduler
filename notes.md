
# 奇技淫巧之加快timeloop搜索速度
./run_mapper.sh搜512的
调layer.py dump出constraints
mv constraint到constraints文件夹下
注释掉其他的constraint（用dummy_constraint补）
调comm_latency_generator.py，设置search为True


# Datatype 编号：
* Input Activation: 0
* Weight Activation: 1
* Output Activation: 2
