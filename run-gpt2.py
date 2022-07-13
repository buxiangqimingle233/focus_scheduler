import subprocess
import time

start = time.time()

diameter = 128
focus_proc = []
graph_analyzer_proc = []
benchmark = 'dall-e'

for i in range(6, 14):
    flit_size = pow(2, i)
    proc = subprocess.Popen(f"python3 focus.py -bm benchmark/{benchmark}.yaml -graph --graph_name ./op_graph_output/{benchmark}-{flit_size}.gpickle -d {diameter} -b 8 -fr {flit_size}-{flit_size}-{flit_size} ds", shell=True)
    focus_proc.append(proc)

for proc in focus_proc:
    proc.communicate()

for i in range(6, 14):
    flit_size = pow(2, i)
    proc = subprocess.Popen(f"python3 compiler/graph_analyzer2.py --op_file ./op_graph_output/{benchmark}-{flit_size}.gpickle --diameter {diameter} --reticle_size 16 --reticle_cycle 5 2>>./{benchmark}-result/flit_{flit_size}.log", shell=True)
    graph_analyzer_proc.append(proc)

for proc in graph_analyzer_proc:
    proc.communicate()

end = time.time()
print(f'Finished in {(end - start):.3} seconds')