# Global Variables for controling
import os

# --------------------- Global Parameters --------------------

taskname = "TBD"

prj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

buffer_root = os.path.join(prj_root, "buffer")
focus_buffer = os.path.join(buffer_root, "focus")
timeloop_buffer = os.path.join(buffer_root, "timeloop-512g")
op_graph_buffer = os.path.join(buffer_root, "op_graph")
# dumped_data_buffer = os.path.join(buffer_root, "dumped_data")

visualization_root = os.path.join(prj_root, "visualization")
spatial_sim_root = os.path.join(prj_root, "simulator")
result_root = os.path.join(prj_root, "results")

# debug
# spatial_sim_root = "/home/wangzhao/simulator/spatial_sim"


dataflow_engine = "timeloop"      # Generate traffic trace from real-world workloads,
                                    # feeding the backends of focus and booksim
# trace_gen_backend = "fake"           # Generate traffic trace by randomly mixing traffic operations


# To mitigate the long-tail issue caused by timeloop, we treat a quantile 
# of PE ending times as the execution latency, but not their maximum. 
quantile_ = 0.9

# -------------------- Toolchain Specs -------------------------
compile_task = True

# -------------------- Timeloop Specs -------------------------

# whether to invoke timeloop-mapper
search_dataflow = False
# whether to invoke timeloop-model
extract_traffic = False
timeloop_verbose = False
# Search time
timeout = 60
# Core numbers (default: number specified in database/arch/arch.yaml)
top_level_cnt = None

# -------------------- Task Mapper Specs -------------------------

mapper_verbose = True
virtualization = True
# mapping_style = "Tetris"
# mapping_style = "Zig-Zag"
mapping_style = "Hilbert"


# -------------------- METRO Specs -------------------------

# Enabling signal
focus_schedule = True

scheduler_verbose = False
n_workers = 30
population_size = 30
n_evolution = 50

# -------------------- Spatial Simulator Specs -------------------------

simulate_baseline = True

# To accelerate simulation, we assume the higher clock frequency for both 
# PEs and NoCs. 
# This parameter reduces the packet size and computing time simultaneously.
# It only effects baseline simulation
overclock = 1

# This parameter describes how much iteration we take.
# It effects both baseline simulation and focus software
shrink = 0.5

# -------------------- Hardware Descriptions -------------------------

# hardware setup

array_diameter = 4                              # d
array_size = array_diameter**2                  # n = d * d
flit_size = 4096                                # in bits

# -------------------- Task Descriptions -------------------------

models = ["bert"]
layer_names = ["{}_layer{}".format(model, i + 1) for i in range(4) for model in models]
cores = [2 for _ in layer_names]
batch = 1

# --------------------- Trace Generator Specs --------------------

# Packet intensity ~ Guassian Distribution
intensity_distributions = {
    0.6: (0.01, 0.005), 0.3: (0.1, 0.02), 0.1: (0.3, 0.05)
}
# Packet interval ~ Gaussian Distribution
interval_distributions = {
    0.5: (1000, 100), 0.3: (10000, 1000), 0.2: (500, 10)
}

# Region size ~ Zipf Distribution
zipf_alpha = 3


# -------------------- Helping Funcions -------------------------

def debug_show(item):
    from pprint import PrettyPrinter
    pp = PrettyPrinter(indent=2)
    pp.pprint(item)
    exit(0)

def get_ea_logpath():
    return os.path.join(focus_buffer, taskname, "ea_output")