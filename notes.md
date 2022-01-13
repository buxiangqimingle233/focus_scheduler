
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


* FOCUS Configs

focus.py为入口文件，控制参数都在utils/global_control里

Dependency: timeloop-pro, OMNET++, HNOCS

## Hardware Setup

Tesla Dojo (roughly estimated): 
* compute core: 1TFlop + 1.25MB
* core array: 19 * 19
* interconnect: 512GB/s
* off-chip BW: 4*9 TB/s
* Frequency: 2GHz

METRO: 
* compute core: 512GFlop + 640KB (128KB GLB, 8\*16KB private input, 4\*16KB private weight)
* core array: 19 * 19
* general interconnect: 256GB/s -- 2048 bits * 1GHz
* 单侧出线 ( Serdes 112GB/s * 5 ) , 512GB/s/core, 共9.5TB/s
* off-chip BW: HBM 900GB/s
* Frequency: 1GHz