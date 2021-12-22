from os import link
from typing import ByteString
import pandas as pd
import utils.global_control as gc

def link_length(node_list):
    d = gc.arch_config["d"]
    links = 0
    for i, ii in zip(node_list, node_list[1:] + [node_list[-1]]):
        links += abs(i % d - ii % d) + abs(i // d + ii // d)
    return links


best_trace = pd.read_json("best_scheduling.json")
best_trace["captain"] = best_trace["captain"].map(lambda x: [x] if not pd.isna(x) else x)
best_trace.loc[best_trace["captain"].isna(), "captain"] = best_trace[best_trace["captain"].isna()]["src"]

with open("utilization_result.csv", "w") as f:
    analyze_start = 10000
    for s in range(25):
        begin, end = analyze_start + s * 400, analyze_start + (s + 1) * 400
        best_trace["real_interval"] = best_trace["delay"] + best_trace["interval"]
        injected_cnts = end // best_trace["real_interval"] - begin // best_trace["real_interval"]
        injected_flits = injected_cnts * best_trace["flit"]

        total_flit = injected_flits.sum()

        max_usage = 0
        # channel 的峰值使用量
        for t in range(begin, end):
            mod = t % best_trace["real_interval"]
            mod[best_trace["real_interval"] > t] = 0
            sel = mod < best_trace["flit"]

            node_lists = best_trace[sel]["src"] + best_trace[sel]["captain"]
            used_channel = mod + best_trace[sel]["dst"].map(len)
            
            # sel = mod[sel] > node_lists.map(link_length)
            # used_channel[sel] -= node_lists.map(link_length)[sel]

            # if mod > link_length:
            #     used_channel = mod + best_trace[sel]["dst"].map(len) - link_length
            # else:
            #     used_channel = mod + best_trace[sel]["dst"].map(len)
            
            # node_lists.map(occupied_links)

            # for _, row in best_trace.iterrows():
            #     mod = t % row["real_interval"]
            #     # is active
            #     if mod < row["flit"]:
            #         used_channel += mod + len(row["dst"])
                    # used_channel += row["flit"] + len(row["dst"])
            max_usage = max(max_usage, used_channel.sum())
        max_usage_ratio = min(1, max_usage / gc.array_size / 6)
        print(total_flit, max_usage_ratio, file=f)