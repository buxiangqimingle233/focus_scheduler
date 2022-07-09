# Setups
Basic link width: 1024-bit, memory controller 1024*16-bit (1024 GB/s, 512Mhz)

MAERI: two broadcast & reduce trees
* broadcast tree: binary, depth=10, leaf link 1024-bit, scale up factors (from bottom up) 1x-1x-1x-1x-1x-2x-2x-2x-2x-2x
* reduce tree: binary, depth=10, leaf link 1024-bit, unified bandwidth (all scale up factors are 1)

# Performance

## MAERI: 
In micro_op_graph.py/evaluate_maeri
* Conflicts happen on each tree link, once a conflict happen, the latency increases with 1 cycle.
* Delay = $\sum_{i=0}^{10} \frac{1}{2^{i}} * penalty^i$ + 1
* For broadcast tree, the delay penalty are $i$, for $i\in [0, 6]$, $0$ for $i \in [7, 10]$
* For reduce tree, the delay penalty are $i$ for $i \in [0, 10]$. 
* We sample the delay for each flits to compute the transmission time for every operands. Then we add the transmission delay to tensor operators' computation time. We let the critical path of operator graph as the total computation time of the application.

## Eyeriss v2
In micro_op_graph.py/evaluate_eyeriss
* We assume no conflicts for Eyeriss-v2
* For messages within the same cluster, we assume two cycle delay for each flits; for messages traversing different clusters, we assume ten (2+8) cycle delay for the each flit.

## Whirl & RPM
Set global_control.py/Router as the router to test, then generate trace and simualte
* Simulator assumes a 4-cycle pipeline, for Whirl, we scale the traffic time to 2/3 given by simulation.
* Traffic time = overall execution time - critical path for tensor operator graph

## METRO
Graph Analyzer + Steineer tree


# Core Utilization
* We only consider the ``activated cycles`` of each core: 
```pseudo code
utilization = task compute time / (task end time - task start time)
```
* MAERI: in micro_op_graph.py/evaluate_maeri
* Eyeriss-v2: in micro_op_graph.py/evaluate_eyeriss
* RPM / Whirl: reported by simulator
* METRO: reported graph analysis framework


# Coefficient of Channel Variation
* Calculate the load of each channel, and calculate the coefficient of channel variation

# Channel Utilization
* Total injected flits / (execution cycles * total bandwidth)
