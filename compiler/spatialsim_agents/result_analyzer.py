import os
import pandas as pd
from compiler import global_control as gc

# FIXME: 
class Analyzer:

    def analyzeSimResult(self):
        result = pd.read_csv(
            os.path.join(gc.spatial_sim_root, "test", gc.taskname, "brief_report.csv"),
            header=None, index_col=None,
            names=["name", "flit", "cycle"]
        )

        # restore original clock frequency
        result["cycle"] *= gc.overclock
        result = result.sort_values(by=["flit"], ascending=True)

        compute_time = (self.traffic["interval"] * self.traffic["counts"]).quantile(gc.quantile_)

        result.loc[:, "slowdown"] = result["cycle"] / compute_time
        result.to_csv(os.path.join(gc.result_root, "baseline_{}.csv".format(gc.taskname)))
        return result
    
    def getFocusResult(self):
        result = pd.read_json(
            os.path.join(gc.focus_buffer, gc.taskname, "solution_{}.json".format(gc.flit_size))
        )

        compute_time = (result.groupby(by="layer").apply(lambda x: (x["interval"] * x["counts"]).max())).quantile(gc.quantile_)
        communicate_time = (result.groupby(by="layer").apply(lambda x: ((x["delay"] + x["interval"]) * x["counts"]).max())).quantile(gc.quantile_)
        slowdown = communicate_time / compute_time

        # Update the result file
        result_file = os.path.join(gc.result_root, "focus_{}.csv".format(gc.taskname))
        try:
            result = pd.read_csv(result_file, index_col=0)
        except FileNotFoundError:
            result = pd.DataFrame(columns=["name", "flit", "cycle", "slowdown"])
        
        inserted_line = ["{}_{}".format(gc.taskname, gc.flit_size), gc.flit_size, communicate_time, slowdown]

        # FUCK YOU Pandas !
        if not (result["flit"] == gc.flit_size).any():
            result = result.append(dict(zip(result.columns, inserted_line)), ignore_index=True)
        else:
            result[result["flit"] == gc.flit_size] = inserted_line

        result.to_csv(os.path.join(gc.result_root, "focus_{}.csv".format(gc.taskname)))