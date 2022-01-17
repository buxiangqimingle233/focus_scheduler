
## 奇技淫巧之加快timeloop搜索速度
./run_mapper.sh搜512的
调layer.py dump出constraints
mv constraint到constraints文件夹下
注释掉其他的constraint（用dummy_constraint补）
调comm_latency_generator.py，设置search为True


# Timeloop Setups
* Input Activation: 0
* Weight Activation: 1
* Output Activation: 2


focus.py为入口文件，控制参数都在utils/global_control里

## Intermediate File Formats: 

Trace File: 
* 我们以Flow为基本单位来描述传输，Flow表示着一个以一定间隔规律产生数据传输的通路，我们将其刻画为FSM，包括 **interval, max_iteration_cnt, waiting_flow_cnt, flits_per_message, dest_id, source_id** 六个属性。

* 我们将Flow依据源节点组织为node，一个Node有nid，# flows两个global参数，后跟着# flows个Flow的描述。

* Node间可以没有顺序，且无需全部描述

例子如下：
```
0 2
4608 32 2 17 51 0
4608 32 2 17 143 0
```


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