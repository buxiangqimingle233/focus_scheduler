from copy import deepcopy
from logging import debug
from math import log2
from tokenize import group
import pandas as pd
import numpy as np
from functools import reduce




class Pattern:
    def __init__(self, srcs, dsts, flit, interval) -> None:
        self.srcs_ = deepcopy(srcs)
        self.dsts_ = deepcopy(dsts)
        self.flit_ = flit
        self.interval_ = interval

    def getFanout(self):
        return len(self.dsts_)

    def getFanin(self):
        return len(self.srcs_)

    def returnDict(self):
        return {"src": self.srcs_, "dst": self.dsts_, "flit": self.flit_, "interval": self.interval_}

    def asSeries(self):
        # FIXME: fix layer
        ser = pd.Series({
            "layer": None,
            "src": self.srcs_,
            "dst": self.dsts_, 
            "interval": self.interval_,
            "flit": self.flit_,
            "counts": 10000,
        })
        return ser


class DataDistribution():
    def __init__(self, srcs, dsts, flit, interval, datatype, layer) -> None:
        self.datatype_ = datatype
        self.delay_ = np.nan
        self.layer_ = layer

        def factors(n):
            return list(set(reduce(list.__add__, 
                    ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

        # self.group_cnt = np.random.choice([2**i for i in range(int(log2(len(dsts)) + 1))], 1)[0]
        self.group_cnt = np.random.choice(factors(len(dsts)), 1)[0]
        self.group_size = len(dsts) // self.group_cnt
        self.groups = []

        for i in range(self.group_cnt):
            # Weight must be fetched from off-chip memory, denoted by -3
            if datatype == "weight":
                self.groups.append(Pattern([-3], dsts[i * self.group_size: (i+1) * self.group_size], 
                                           flit, interval))
            # Input are fetched from either last layer or off-chip memory, denoted by -1
            elif datatype == "input":
                self.groups.append(Pattern([-1], dsts[i * self.group_size: (i+1) * self.group_size], 
                                           flit, interval))

    def asDataFrame(self) -> pd.DataFrame:
        df = pd.concat([group.asSeries() for group in self.groups], axis=1).T
        df["datatype"] = self.datatype_
        df["delay"] = 0
        df["layer"] = self.layer_
        df.astype({"flit": "int32", "interval": "int32"})
        return df


class DataReduction():
    def __init__(self, srcs, dsts, flit, interval, datatype, layer) -> None:
        self.datatype_ = datatype
        self.delay_ = np.nan
        self.layer_ = layer
        self.group_cnt = np.random.choice([2**i for i in range(int(log2(len(srcs)) + 1))], 1)[0]
        self.group_size = len(srcs) // self.group_cnt
        self.groups = []

        for i in range(self.group_cnt):
            # Output are accumulated by a selected hub node, denoted by -2
            self.groups.append(Pattern(srcs[i * self.group_size: (i+1) * self.group_size], [-2],
                                        flit, interval))
    def asDataFrame(self) -> pd.DataFrame:
        df = pd.concat([group.asSeries() for group in self.groups], axis=1).T
        df["datatype"] = self.datatype_
        df["delay"] = df.iloc[0]["interval"]
        df["layer"] = self.layer_
        return df

# TODO: To be done
class DataSync():
    def __init__(self) -> None:
        pass


if __name__ == "__main__":
    a = DataDistribution([], range(16), 45, 5, "input")
    print(a.asDataFrame())
    