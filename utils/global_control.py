# Global Variables for controling


# -------------------- Layer -------------------------

# whether to invoke timeloop-mapper
search_dataflow = False
# Search time
timeout = 300
# Core numbers (default: number specified in db/arch/arch.yaml)
top_level_cnt = None
# whether to invoke timeloop-model
dump_comm_status = False

# -------------------- Hardware -------------------------

# hardware setup
array_diameter = 8
array_size = array_diameter**2
arch_config = {
    "p": 6, "cp_if": 6, "cp_of": 0, "tr": 1, "ts": 2, "tw": 1,
    "n": array_diameter**2,
    "d": array_diameter,
    "w": 2048
}


# -------------------- Tasks -------------------------

# layer_names = [
#     "resnet50_layer43", "resnet50_layer44", "resnet50_layer45", "resnet50_layer46", \
#     "vgg16_layer1", "vgg16_layer2", "vgg16_layer3", "vgg16_layer4", \
#     "inception_layer1", "inception_layer2", "inception_layer3", "inception_layer4"
# ]

# layer_names = [
#     "resnet50_layer43", "resnet50_layer44", \
#     "vgg16_layer1", "vgg16_layer2", \
#     "inception_layer1", "inception_layer2"
# ]

# cores = [
#     8, 16, 1, 8, 16, 2
# ]

layer_names = [
    "resnet50_layer43", "resnet50_layer44"
]
cores = [
    8, 16
]


# -------------------- Task Mapper -------------------------

verbose_mapper = True
# .cfg formatted file corresponding to the hardware setups
conf_filename = "ML_mapper.cfg"

# -------------------- HNOCS -------------------------

simulate_baseline = True
hnocs_working_path = "/home/wangzhao/simulator/HNOCS/simulations"

# -------------------- FOCUS Scheduler -------------------------



# -------------------- [Optional] Estimator -------------------------

use_estimator = False
cv = 2


# -------------------- Helping Funcions -------------------------

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
    config.write(open(conf_filename, "w"))

generate_config_file()