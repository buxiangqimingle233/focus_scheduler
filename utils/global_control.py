# Global Variables for controling

# --------------------- Global Parameters --------------------


trace_gen_backend = "timeloop"      # Generate traffic trace from real-world workloads, 
                                    # feeding the backends of focus and booksim

# trace_gen_backend = "fake"           # Generate traffic trace by randomly mixing traffic operations

result_dir = "result-512g"
result_file = "slowdown.csv"

# -------------------- Timeloop Control -------------------------

# whether to invoke timeloop-mapper
search_dataflow = False
# whether to invoke timeloop-model
dump_comm_status = False
timeloop_verbose = False
# Search time
timeout = 30
# Core numbers (default: number specified in db/arch/arch.yaml)
top_level_cnt = None

# data orders
datatype = ["weight", "input", "output"]


# -------------------- Task Mapper Specs -------------------------

mapper_verbose = True
# mapping_style = "Tetris"
# mapping_style = "Zig-Zag"
mapping_style = "Hilbert"


# -------------------- METRO Specs -------------------------

# Enabling signal
focus_schedule = True

scheduler_verbose = False
n_workers = 28
population_size = 5
n_evolution = 5

# -------------------- Spatial Simulator Specs -------------------------

simulate_baseline = True
booksim_working_path = "/home/wangzhao/simulator/booksim2"

# -------------------- Hardware Descriptions -------------------------

# hardware setup

array_diameter = 4                 # d
array_size = array_diameter**2     # n = d * d
flit_size = 4096                   # in bits


# -------------------- Task Descriptions -------------------------

models = ["bert"]
layer_names = ["{}_layer{}".format(model, i + 1) for i in range(4) for model in models]
cores = [2 for _ in layer_names]

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

# -------------------- Helping Funcions -------------------------

def debug_show(item):
    from pprint import PrettyPrinter
    pp = PrettyPrinter(indent=2)
    pp.pprint(item)
    exit(0)