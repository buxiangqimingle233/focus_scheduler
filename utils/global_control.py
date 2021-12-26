# Global Variables for controling

# -------------------- Hardware Setup -------------------------
result_dir = "result-512g"

# --------------------- Trace Gen Backends --------------------

trace_gen_backend = "timeloop"

# --------------------- Sampling Trace Generator --------------------

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


# -------------------- Timeloop -------------------------

timeloop_verbose = False    
# whether to invoke timeloop-mapper
search_dataflow = False
# whether to invoke timeloop-model
dump_comm_status = False
# Search time
timeout = 30
# Core numbers (default: number specified in db/arch/arch.yaml)
top_level_cnt = None

# data orders
datatype = ["weight", "input", "output"]
# -------------------- Hardware -------------------------

# hardware setup
array_diameter = 4
array_size = array_diameter**2
arch_config = {
    "p": 6, "cp_if": 6, "cp_of": 0, "tr": 1, "ts": 2, "tw": 1,
    "n": array_diameter**2,
    "d": array_diameter,
    "w": 4096
}

# mapping_style = "Tetris"
# mapping_style = "Zig-Zag"
mapping_style = "Hilbert"


# -------------------- Tasks -------------------------


# model = "bert"
# layer_names = ["{}_layer{}".format(model, i + 1) for i in range(4)]
# cores = [2 for _ in layer_names]

model = "flappybird"
layer_names = ["{}_layer{}".format(model, i + 1) for i in range(4)]
cores = [2 for _ in layer_names]

# -------------------- Task Mapper -------------------------

mapper_verbose = True
# .cfg formatted file corresponding to the hardware setups
# conf_filename = "ML_mapper.cfg"

# -------------------- HNOCS -------------------------

simulate_baseline = True
hnocs_working_path = "/home/wangzhao/simulator/HNOCS/simulations"

# -------------------- BookSim -------------------------

# simulate_baseline = False
booksim_working_path = "/home/wangzhao/simulator/booksim2/src"


# -------------------- FOCUS Scheduler -------------------------

focus_schedule = False
scheduler_verbose = False
n_workers = 28
population_size = 5
n_evolution = 5

result_file = "slowdown.csv"

# -------------------- [Optional] Estimator -------------------------

use_estimator = False
cv = 2


# -------------------- Helping Funcions -------------------------

def debug_show(item):
    from pprint import PrettyPrinter
    pp = PrettyPrinter(indent=2)
    pp.pprint(item)
    exit(0)