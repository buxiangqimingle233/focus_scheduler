from functools import reduce
import re
import os
import numpy as np
import pandas as pd
from copy import deepcopy
import random
import yaml
from time import time

INF = 1e10


class XYRouter:
    port_number = {"input": 0, "output": 1, "north": 2, "south": 3, "west": 4, "east": 5}

    def __init__(self, shape:tuple):
        self.shape = shape
        assert(len(shape) == 2)
    
    def getPath(self, src, dst):
        '''
            Return:
                Path from `src` to `dst`, in the format of a list of (router, output port)
        '''
        isize, jsize = self.shape
        
        def getCoordinates(index):
            return index // isize, index % isize
        
        def getIndex(i, j):
            return i * jsize + j

        isrc, jsrc = getCoordinates(src)
        idst, jdst = getCoordinates(dst)

        istep = 1 if isrc < idst else -1
        jstep = 1 if jsrc < jdst else -1

        router_path = []
        # x-routing, keep i
        router_path += [getIndex(isrc, j) for j in range(jsrc, jdst, jstep)]
        # y-routing, keep j
        router_path += [getIndex(i, jdst) for i in range(isrc, idst, istep)]

        # get ports
        router_path += [getIndex(idst, jdst)] * 2   # append the last router to calculate the output port of last router
        oport_path = [self.__getOutPort(router_path[i], router_path[i + 1]) for i in range(len(router_path) - 1)]

        return list(zip(router_path, oport_path))

    def __getOutPort(self, from_, to_):
        bias = to_ - from_
        
        # east
        if bias == 1:
            return self.port_number["east"]
        elif bias == -1:
            return self.port_number["west"]
        elif bias == self.shape[0]:
            return self.port_number["south"]
        elif bias == -self.shape[0]:
            return self.port_number["north"]
        elif bias == 0:
            return self.port_number["output"]
        else:
            raise Exception("The two nodes are not neighbours!")


class FocusLatencyModel():

    packets = pd.DataFrame(columns=["id", "src", "dst", "flits", "interval", "path", "issue_time", "count"])
    routers = pd.DataFrame(columns=["rid", "coordinate", "port", "grab_start", "grab_end"])

    def __init__(self, array_shape):
        self.array_shape = array_shape
        self.array_size = reduce(lambda x, y: x*y, array_shape)

        ids = np.zeros(array_shape)
        coordinates = np.argwhere(ids == 0)
        self.routers = pd.DataFrame(
            data=[[idr, coordinate, port, 0, 0] for idr, coordinate in zip(np.arange(ids.size), coordinates) for port in range(6)], 
            columns=self.routers.columns
        )
    
    def run(self, packets):

        working_pkts = packets.copy()

        clk = 0
        working_pkts["unsolved"] = True
        working_pkts["delay"] = 0

        iter_cnt = 0
        while any(working_pkts["unsolved"]):
            
            iter_cnt += 1

            # if iter_cnt % 500 == 0:
            #     print("iteration: {}, remained packets: {}".format(iter_cnt, (working_pkts["unsolved"].value_counts())[True]))

            issued_pkt = working_pkts[working_pkts["unsolved"]].sort_values("issue_time").iloc[0]

            path = issued_pkt["path"]
            path_ids = list(map(lambda x: x[0] * 6 + x[1], path))

            # router & port in path
            sel = np.zeros(self.routers.shape[0], dtype=bool)
            sel[path_ids] = True

            # release time in path
            grab_time = np.zeros(self.routers.shape[0])
            grab_time[path_ids] = issued_pkt["flits"] + np.arange(len(path_ids)) + 1

            wait_until = self.routers["grab_end"][sel].max()

            issue_time = issued_pkt["issue_time"]

            # issue the packet
            if issue_time >= wait_until:
                self.routers.loc[sel, "grab_start"] = issue_time
                self.routers.loc[:, "grab_end"] = issue_time + grab_time

                # mark this packet to have been issued
                remain_count = working_pkts.loc[issued_pkt["id"], "count"]
                working_pkts.loc[issued_pkt["id"], "count"] = remain_count - 1

                if remain_count <= 0:
                    working_pkts.loc[issued_pkt["id"], "unsolved"] = False
                else:
                    working_pkts.loc[issued_pkt["id"], "delay"] += \
                        grab_time.max() + issue_time \
                        - (packets.loc[issued_pkt["id"], "count"] - working_pkts.loc[issued_pkt["id"], "count"]) * working_pkts.loc[issued_pkt["id"], "interval"]
                    working_pkts.loc[issued_pkt["id"], "delay"] = max(0, working_pkts.loc[issued_pkt["id"], "delay"])
                    working_pkts.loc[issued_pkt["id"], "issue_time"] += working_pkts.loc[issued_pkt["id"], "interval"]

            # delay the packet
            else:
                issued_pkt["issue_time"] = wait_until
                working_pkts.loc[issued_pkt["id"]] = issued_pkt

        working_pkts["is_bound"] = working_pkts["delay"] > 0

        print("Iteration counts: {}".format(iter_cnt))
        return working_pkts


class FocusTemporalMapper():
    packets = pd.DataFrame(columns=["id", "src", "dst", "flits", "interval", "path", "issue_time", "count"])

    def __init__(self):
        pass

    def temporal_map(self, packets):
        ret = packets.sort_values("flits")
        ret["issue_time"] = 0
        return ret


class Individual():

    def __init__(self, trace, array_shape, iter_episode=1):
        
        trace.columns = ["src", "dst", "interval", "flits", "count"]
        trace["id"] = trace.index
        trace["intermediate"] = [[] for _ in range(trace.shape[0])] 
        trace["path"] = [[] for _ in range(trace.shape[0])]

        # For reducing the time of simulating
        trace["count"] = iter_episode

        self.trace = trace
        self.array_shape = array_shape
        self.array_size = reduce(lambda x, y: x*y, self.array_shape)
    
    def mutate(self,inplace=False):
        # start = time.time()

        if inplace:
            new=self
        else:
            new=deepcopy(self)
        for _ in range(np.random.randint(50)):
            if random.random() > 0.5:
                new.addImNode()
            else:
                new.rmImNode()
        # end = time.time()
        # print("Used time: {}s".format(end - start))

        return new

    @staticmethod
    def crossover(left, right):
        ltrace, rtrace = left.getTrace(), right.getTrace()
        left_sel_idx = random.sample(range(ltrace.shape[0]), int(ltrace.shape[0]/2))

        child = deepcopy(right)
        ctrace = child.getTrace()
        ctrace.iloc[left_sel_idx] = ltrace.iloc[left_sel_idx]
        child.setTrace(ctrace)
        
        return child

    def getTrace(self):
        return self.trace

    def setTrace(self, trace):
        self.trace = trace

    def addImNode(self):
        sel_idx = random.choice(range(self.trace.shape[0]))
        sel_pkt = self.trace.iloc[sel_idx]
        path = sel_pkt.loc["intermediate"]
        
        if self.array_size != len(path):
            path.append(random.choice(list(set(range(self.array_size)) - set(path))))

    def rmImNode(self):
        sel_idx = random.choice(range(self.trace.shape[0]))
        sel_pkt = self.trace.iloc[sel_idx]

        if sel_pkt["intermediate"]:
            sel_pkt["intermediate"].pop(random.choice(range(len(sel_pkt["intermediate"]))))
        
    def evaluate(self):
        start_time = time()

        working_trace = self.trace.copy()
        
        router = XYRouter(self.array_shape)
        for idx, row in working_trace.iterrows():
            milestones = [row["src"]] + row["intermediate"] + [row["dst"]]
            path = []

            # do routing
            for i in range(len(milestones)-1):
                segment_path = router.getPath(milestones[i], milestones[i+1])
                # drop the output port of intermediate nodes
                if i != len(milestones) - 1:
                    segment_path = segment_path[:-1]
                path += segment_path

            # write back
            row["path"] = deepcopy(path)
            working_trace.loc[idx] = row

        # temporal map
        # print("begin temporal mapping")
        temporal_mapper = FocusTemporalMapper()
        working_trace = temporal_mapper.temporal_map(working_trace)

        # print("begin estimating")
        # estimate latency
        latency_model = FocusLatencyModel(self.array_shape)
        working_trace = latency_model.run(working_trace)
        
        end_time = time()
        print("Evaluate time: {} Score: {}".format(end_time-start_time, working_trace["delay"].sum()))

        return -working_trace["delay"].sum()


if __name__ == "__main__":
    trace_file = "focus/ts_scheduler/trace.dat"
    
    ind1 = Individual(pd.read_csv(trace_file, header=0), (15, 15))
    ind2=ind1.mutate()
    ind1.crossover(ind1,ind2)

    # print(Individual.crossover(ind1, ind2).getTrace())
    # r = XYRouter((4, 4))
    # # trace["path"] = 
    
    # path = r.getPath(7, 0)
    # print(path)