import focus
import multiprocessing
import os
from utils import global_control as gc
import numpy as np

widths = range(512, 4097, 512)
search_dataflow = False
result_file = "fake_trace.csv"

#np.random.seed(1145)

def run():
    gc.trace_gen_backend = "generator"
    gc.search_dataflow = False
    gc.result_file = result_file

    pool = multiprocessing.Pool(processes=8)
    result_run = []
    id = 0
    for w in widths:
        id += 1
        result_run.append(pool.apply_async(focus.run,(id,)))
    pool.close()
    pool.join()
    for res in result_run:
        print (res.get())
    with open(os.path.join("focus-final-out", f"booksim_slowdown_parallel.csv"), "a") as wf:
        # print(arch_config["w"], mean_slowdown, mean_delay, file=wf, sep="\t")
        for res in result_run:
            print(res.get(), file=wf)
    
if __name__ == "__main__":
    run()