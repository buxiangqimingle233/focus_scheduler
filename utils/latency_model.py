import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import copy
import numpy as np
from collections import Counter
import routing as RS

PINF = 1e10


# Wormhole arbitration only
class LatencyModel():
    ''' Estimating average packet latency for a given mesh and communication workload

    NoC Setups: 
        topology: mesh
        routing: XY-routing
        arbitration: wormhole 
        traffic trace: supported

    '''

    arch_arg = {}
    task_arg = {}
    cache = {}

    def __init__(self):
        super().__init__()

    def runModel(self, task: dict, arch: dict) -> list:
        '''Analyzing latency of each traffic

        Architecture configurations:
            d: diameter
            n: router size
            p: channels per router
            tr: arbitration delay (cycles)
            ts: switching delay (cycles)
            tw: adjacent router wire delay (cycles)
            cp_if: input buffer capacity (flits)
            cp_of: output buffer capacity (flits)
            w: bits per flit (bits)

        Task configurations:
                graph: List[(src_id, dst_id, packets/cycle)]
                cv: average coefficiency variants of traffic
                avg_l: average packet size (flits)

        Return:
            A list of estimated transmission latency of requests with the same order of task["graph"].
        '''
        self._initializeEngine(task, arch)

        self._setupVariables()

        # update W first
        self._updateRouterBlockingTime(0)

        # Calculate blocking time of each router
        RH = self.cache["RH"]
        max_rh = np.max(RH[RH != PINF])
        for rh in range(1, max_rh + 1):
            self._updateOCServiceTime(rh)
            self._updateRouterBlockingTime(rh)

        # Sum up blocking cycles along the path
        total_traffic_time = self._accumulateResults()
        if not self._checkResults(total_traffic_time):
            # print(total_traffic_time)
            # exit(1)
            raise Exception("Negative communication time!")
        return total_traffic_time

    def _initializeEngine(self, task, arch):
        self.cache.clear()

        # Check fields of architecture specification and task specification
        for arch_field in ["d", "n", "p", "tr", "ts", "tw", "cp_if", "cp_of", "w"]:
            if arch_field not in arch:
                raise Exception("Please specify {} in architecture".format(arch_field))
        for task_field in ["graph", "cv", "avg_l"]:
            if task_field not in task:
                raise Exception("Please specify {} in task".format(task_field))

        # Check array shape
        if arch["n"] != arch["d"]**2:
            raise Exception("Invalid hardware configuration: d = {}, n = {}".format(arch["d"], arch["n"]))
        # Check traffic graph
        # if max(Counter([(r[0], r[1]) for r in task["graph"]]).values()) > 1:
        #     raise Exception("Invalid task configuration, duplicated traffic for the same path")

        self.task_arg, self.arch_arg = copy.deepcopy(task), copy.deepcopy(arch)
        self.rter = RS.XYRouting(self.arch_arg)

    def _setupVariables(self):
        '''Preprocess the task graph and extract its features
        Step 1 & 2 in the article: Calculating P(s->d), L, cv, P_p2p, L_p2p, L_p, RH
            Return:
                P_s2d: A list with factors' format as (src_rt, dst_rt, ratio), which denotes the proportion of
                    trasmission volume of the request (src_rt, dst_rt)
                L: A (n, ) ndarray, where L[i] denotes average injection rate of router i
                L_p: A (n, p) ndarray, where Lp[i, j] denotes packet arrival rate to the output channel j of router i
                P_p2p: A (n, p, p) ndarray, where P_p2p[i, j, k] denotes probability of a packet entered
                    from channel j is routed to channel k in router i
                L_p2p: A (n, p, p) ndarray, where L_p2p[i, j, k] denotes trasmission rate from channel j to channel k
                    in router i
                RH: A (n, p) ndarray, where RH[i, j] denotes the longest residual hops of packets
                    passing router i output channel j
        '''

        G = self.task_arg["graph"]

        Vol = np.asarray([r[2] for r in G])
        P_s2d = Vol / np.sum(Vol)
        # P_s2d = np.ones(Vol.shape)

        # Set up L_p2p, L, RH, pkt_path
        n, p = self.arch_arg["n"], self.arch_arg["p"]
        L_p2p = np.zeros((n, p, p))
        L = np.zeros(n)
        RH = np.tile(PINF, (n, p)).astype(np.int)
        pkt_paths = self.rter.path(G)

        for request, path, proportion in zip(G, pkt_paths, P_s2d):
            router_path, inport_path, outport_path = zip(*path)
            vol = request[2]
            L_p2p[router_path, inport_path, outport_path] += vol * proportion
            L[request[0]] += vol * proportion
            residual_hops = np.asarray([i for i in range(len(outport_path)-1, -1, -1)])
            RH[router_path, outport_path] = np.minimum(RH[router_path, outport_path], residual_hops)

        # Setup P_p2p
        L_p2p_t = np.transpose(L_p2p, (1, 0, 2))    # TODO: Transpose should be replaced to optimize the performance
        L_p = np.sum(L_p2p_t, axis=0)
        P_p2p_t = L_p2p_t / (L_p + 1e-10)
        P_p2p = np.transpose(P_p2p_t, (1, 0, 2))

        # Setup S, S2
        S, S2 = np.zeros((n, p)) + 1e-10, np.zeros((n, p)) + 1e-10    # for getting rid of dividing zeros
        ts, tw, l_, cv = self.arch_arg["ts"], self.arch_arg["tw"], self.task_arg["avg_l"], self.task_arg["cv"]
        lb = (l_ - 1) * max(ts, tw)
        S[RH == 0] = ts + tw + lb
        S2[RH == 0] = (ts + tw + lb)**2 / (cv**2 + 1)

        # Setup W
        W = np.zeros((n, p, p)) + 1e-10

        # Store all of them 3479.568096328011, 4821.314443450869
        c = self.cache
        c["P_s2d"], c["L"], c["L_p"], c["P_p2p"], c["L_p2p"], c["RH"], c["S"], c["S2"], c["pkt_path"], c["W"]  \
            = P_s2d, L, L_p, P_p2p, L_p2p, RH, S, S2, pkt_paths, W

    def _updateOCServiceTime(self, rh):
        '''Update s_i^M and s_i^M^2
        Formula 16
            S: A (n, p) ndarray, where S[i, j] dentoes the first moment of
                service time of output channel j of router i
            S2: A (n, p) ndarray, where S[i, j] dentoes the second moment of
                service time of output channel j of router i
            W: A (n, p, p) ndarray, where W[i, j, k] denotes bloking time spent on queuing from
                input channel j to output channel k in router i
            P_p2p: A (n, p, p) ndarray, where P_p2p[i, j, k] denotes probability of a packet entered
                    from channel j is routed to channel k in router i
            RH: A (n, p) ndarray, where RH[i, j] indicates the longest residual hops of packets
                    passing router i output channel j
            rh: The present residual hop (step) we are working on
        '''
        assert rh > 0
        ts, tr, tw = self.arch_arg["ts"], self.arch_arg["tr"], self.arch_arg["tw"]
        cp_if, cp_of = self.arch_arg["cp_if"], self.arch_arg["cp_of"]
        n, p = self.arch_arg["n"], self.arch_arg["p"]
        S, S2, W = self.cache["S"], self.cache["S2"], self.cache["W"]
        P_p2p, RH = self.cache["P_p2p"], self.cache["RH"]

        Upd_s = np.zeros((n, p)).astype("float128")
        Upd_s2 = np.zeros((n, p)).astype("float128")
        P_p2p = P_p2p.astype("float128")
        R, C = np.where(RH == rh)
        for r, c in zip(R, C):
            dst_r, dst_ic = self.rter.pointTo(r, c)
            for dst_oc in range(6):
                latency = ts + tr + tw
                latency += W[dst_r, dst_ic, dst_oc]
                latency += S[dst_r, dst_oc]
                latency -= max(ts, tw) * (cp_if + cp_of)
                latency = max(latency, 0)
                Upd_s[r, c] += P_p2p[dst_r, dst_ic, dst_oc] * latency
                try:
                    Upd_s2[r, c] += P_p2p[dst_r, dst_ic, dst_oc] * latency**2
                except Exception as e:
                    print("latency: {}".format(latency))
                    raise e
        S += Upd_s
        S2 += Upd_s2

    def _updateRouterBlockingTime(self, rh):
        '''Update W
        Formula 13
            W: A (n, p, p) ndarray, where W[i, j, k] denotes bloking time spent on queuing from
                input channel j to output channel k in router i
            S: A (n, p) ndarray, where S[i, j] dentoes the first moment of service time of
                output channel j of router i
            S2: A (n, p) ndarray, where S[i, j] dentoes the second moment of service time of
                output channel j of router i
            L_p: A (n, p) ndarray, where Lp[i, j] denotes packet arrival rate to the
                output channel j of router
            L_p2p: A (n, p, p) ndarray, where L_p2p[i, j, k] denotes trasmission rate from
                channel j to channel k in router i
            RH: A (n, p) ndarray, where RH[i, j] indicates the longest residual hops of packets
                passing router i output channel j
            rh: The present residual hop (step) we are working on
        '''
        S, S2, W = self.cache["S"], self.cache["S2"], self.cache["W"]
        L_p, L_p2p, RH = self.cache["L_p"], self.cache["L_p2p"], self.cache["RH"]

        S, S2 = S.astype("float128"), S2.astype("float128")

        lr, lc = np.where(RH == rh)
        for r, oc in zip(lr, lc):
            p = self.arch_arg["p"]
            L = np.asarray([np.sum(L_p2p[r, :i, oc]) for i in range(p)])
            L[0] = L_p2p[r, 0, oc]                  # input channel
            Service_rate = 1 / S[r, oc]
            L[1:] = 2 * (Service_rate - L[1:])**2
            L[0] = 2 * (Service_rate - L[0])        # input channel
            if (L < 0).any():
                raise Exception("Service rate < incoming rate: {}", list(L < 0))
            arrival_rate = L_p[r, oc]
            ca2 = self.task_arg["cv"]**2
            cs2 = S2[r, oc] / S[r, oc]**2 - 1
            L = arrival_rate * (ca2 + cs2) / L
            L[0] = L[0] / Service_rate              # input channel
            W[r, :, oc] = L

        if np.nan in W:
            raise Exception("NAN in W: rh = {}".format(rh))

    def _accumulateResults(self):
        l_ = self.task_arg["avg_l"]
        ts, tw, tr = self.arch_arg["ts"], self.arch_arg["tw"], self.arch_arg["tr"]
        W, Path = self.cache["W"], self.cache["pkt_path"]
        lb = max(ts, tw) * (l_ - 1)
        Time = []
        for path in Path:
            time = lb
            for r, ic, oc in path:
                time += tr + W[r, ic, oc] + ts + tw
            Time.append(float(time))
        return Time

    def _checkResults(self, times):
        return all([t > 0 for t in times])

if __name__ == "__main__":
    task = {
        "graph": [(0, 1, 0.05), (2, 1, 0.05)],
        "cv": 0,
        "avg_l": 2
    }
    arch = {
        "d": 2, "n": 4, "p": 6, "tr": 0, "ts": 1, "tw": 1, "cp_if": 1, "cp_of": 0, "w": 64
    }

    pm = LatencyModel()
    print(pm.runModel(task, arch))
