# Global Variables for controling



# -------------------- Hardware Setup -------------------------

result_dir = "result-512g"



# -------------------- Timeloop -------------------------



timeloop_verbose = False

# whether to invoke timeloop-mapper

search_dataflow = False
# Search time

timeout = 30

# Core numbers (default: number specified in db/arch/arch.yaml)

top_level_cnt = None

# whether to invoke timeloop-model

dump_comm_status = False


# data orders

datatype = ["weight", "input", "output"]

# -------------------- Hardware -------------------------



# hardware setup

array_diameter = 16

array_size = array_diameter**2

arch_config = {

    "p": 6, "cp_if": 6, "cp_of": 0, "tr": 1, "ts": 2, "tw": 1,

    "n": array_diameter**2,

    "d": array_diameter,

    "w": 512

}



# mapping_style = "Tetris"

# mapping_style = "Zig-Zag"

mapping_style = "Hilbert"





# -------------------- Tasks -------------------------





model = "vgg16"

layer_names = ["unet_layer1", "unet_layer2", "unet_layer3", "unet_layer4", "unet_layer5", "unet_layer6", "unet_layer7", "unet_layer8", "resnet50_layer1", "resnet50_layer2", "resnet50_layer3", "resnet50_layer4", "bert-large_layer1", "bert-large_layer2", "bert-large_layer3", "bert-large_layer4", "bert-large_layer5", "bert-large_layer6", "bert-large_layer7", "bert-large_layer8", "bert-large_layer9", "bert-large_layer10", "bert-large_layer11", "bert-large_layer12", "bert-large_layer13", "bert-large_layer14", "bert-large_layer15", "bert-large_layer16", "bert-large_layer17", "bert-large_layer18", "bert-large_layer19", "bert-large_layer20", "bert-large_layer21", "bert-large_layer22", "bert-large_layer23", "bert-large_layer24", "bert-large_layer25", "bert-large_layer26", "bert-large_layer27", "bert-large_layer28", "bert-large_layer29", "bert-large_layer30", "bert-large_layer31", "bert-large_layer32", "ssd_r34_layer1", "ssd_r34_layer2", "ssd_r34_layer3", "ssd_r34_layer4"]
cores = [8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 16, 16, 16, 16]




# -------------------- Task Mapper -------------------------



mapper_verbose = True

# .cfg formatted file corresponding to the hardware setups

conf_filename = "ML_mapper.cfg"



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

population_size = 100

n_evolution = 50



slowdown_result = "slowdown_unet_resnet50_bert-large_ssd_r34.csv"


# -------------------- [Optional] Estimator -------------------------



use_estimator = False

cv = 2





# -------------------- Helping Funcions -------------------------



def debug_show(item):

    from pprint import PrettyPrinter

    pp = PrettyPrinter(indent=2)

    pp.pprint(item)

    exit(0)
