* FOCUS

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