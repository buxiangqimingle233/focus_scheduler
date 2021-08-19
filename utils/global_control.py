# Global Variables for controling



# -------------------- Hardware Setup -------------------------

result_dir = "result-512g"



# -------------------- Timeloop -------------------------



timeloop_verbose = False

# whether to invoke timeloop-mapper

search_dataflow = True
# Search time

timeout = 30

# Core numbers (default: number specified in db/arch/arch.yaml)

top_level_cnt = None

# whether to invoke timeloop-model

dump_comm_status = True


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

    "w": 256

}



# mapping_style = "Tetris"

# mapping_style = "Zig-Zag"

mapping_style = "Hilbert"





# -------------------- Tasks -------------------------





model = "vgg16"

layer_names = ["bert_layer1", "bert_layer2", "bert_layer3", "bert_layer4", "bert_layer5", "bert_layer6", "bert_layer7", "bert_layer8", "bert_layer9", "bert_layer10", "bert_layer11", "bert_layer12", "bert_layer13", "bert_layer14", "bert_layer15", "bert_layer16", "bert_layer17", "bert_layer18", "bert_layer19", "bert_layer20", "bert_layer21", "bert_layer22", "bert_layer23", "bert_layer24", "bert_layer25", "bert_layer26", "bert_layer27", "bert_layer28", "bert_layer29", "bert_layer30", "bert_layer31", "bert_layer32", "bert_layer33", "bert_layer34", "bert_layer35", "bert_layer36", "bert_layer37", "bert_layer38", "bert_layer39", "bert_layer40", "bert_layer41", "bert_layer42", "bert_layer43", "bert_layer44", "bert_layer45", "bert_layer46", "bert_layer47", "bert_layer48", "bert_layer49", "bert_layer50", "bert_layer51", "bert_layer52", "bert_layer53", "bert_layer54", "bert_layer55", "bert_layer56", "bert_layer57", "bert_layer58", "bert_layer59", "bert_layer60", "bert_layer61", "bert_layer62", "bert_layer63", "bert_layer64"]
cores = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]




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



slowdown_result = "slowdown_bert.csv"


# -------------------- [Optional] Estimator -------------------------



use_estimator = False

cv = 2





# -------------------- Helping Funcions -------------------------



def debug_show(item):

    from pprint import PrettyPrinter

    pp = PrettyPrinter(indent=2)

    pp.pprint(item)

    exit(0)





def generate_config_file():

    # Generate such configure file

    import configparser as cp

    config = cp.ConfigParser()

    config["accelerator_config"] = {

        "tile_array_height": array_diameter, 

        "tile_array_width": array_diameter

    }

    config["layer_config"] = {

        "layer_num": len(cores),

        "layer_MACs": ":".join(map(lambda x: str(x), cores))

    }

    config["mapping_style"] = {

        "mapping_style": mapping_style

    }

    config.write(open(conf_filename, "w"))



generate_config_file()
