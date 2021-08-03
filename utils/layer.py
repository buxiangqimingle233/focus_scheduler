import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import yaml
import time
import subprocess
import signal
import types
from functools import reduce
from loop2map import Loop2Map
import random
import global_control as gc


class Layer:
    '''Conducting analysis for a single layer\n
    Side effects:\n
        1. Create root_dir/result directorty & create /result/`layer_name` directory
        2. Set working directory to root_dir/result/`layer_name` (When function "execute" is called)
        3. Invoke timeloop-mapper based on specifications in root_dir/db/*
        4. Invoke timeloop-model and store communication status in root_dir/result/`layer_name`
    Inputs:\n
        layer_file_: yaml file of the layer (termed as problem specification file)
        model_dir: subdirectory which contains problem specification file of this layer (`prob` in default)
        prj_root: full path to the parent folder of db (parent folder of this python script in default)
    '''

    layer_name = None
    working_dir = None
    top_level_cnt = None
    mapper_stats_file = "timeloop-mapper.stats.txt"
    mapper_map_file = "timeloop-mapper.map.txt"
    model_stats_file = "timeloop-model.stats.txt"

    dump_mapping_file = "dump_mapping.yaml"
    comm_report_file = "communication.yaml"

    def __init__(self, layer_file, model_dir="prob", prj_root=None, dram_spatial_size=None):
        if not prj_root:
            self.prj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.prj_root = prj_root

        result_root = os.path.join(self.prj_root, "result")
        if not os.path.exists(result_root):
            os.makedirs(result_root)

        self.layer_name = layer_file[:-5]       # xxxx.yaml
        if not dram_spatial_size:
            self.working_dir = os.path.join(result_root, self.layer_name)
        else:
            self.working_dir = os.path.join(result_root, self.layer_name + "_" + str(dram_spatial_size))
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)

        self.top_level_cnt = dram_spatial_size
        self.arch_specs = self._get_specs_from_database("arch")
        self.mapper_specs = self._get_specs_from_database("mapper")
        self.constraint_specs = self._get_specs_from_database("constraints")
        self.prob_specs = self._get_specs_from_database(model_dir, layer_file)

    def _get_specs_from_database(self, specs_dir, file_name=None):
        ''' Traverse files in `specs_dir` directory and concatenate their full path divided by a space\n
        (Just return the full path of `file_name` if it is sepcified)
        '''
        database_path = os.path.join(self.prj_root + "/db")
        if not file_name:
            specs = []
            sub_dir_fullpath = os.path.join(database_path, specs_dir)
            if not os.path.exists(sub_dir_fullpath):
                raise Exception("directory " + sub_dir_fullpath + " does not exist!!")
            for root, _, files in os.walk(sub_dir_fullpath):
                for file in files:
                    specs.append(os.path.join(root, file))
            return " ".join(specs)
        else:
            spec = os.path.join(database_path, specs_dir, file_name)
            if not os.path.exists(spec):
                raise Exception("file " + spec + " does not exist!!")
            return spec

    def _get_top_arch_specification(self):
        for arch_spec in self.arch_specs.split():
            obj = yaml.load(open(arch_spec, "r"), Loader=yaml.FullLoader)
            if "architecture" in obj:
                return obj, arch_spec
        raise "keyword `architecture` has not been found"

    def _modify_arch_specification(self, top_level_pe_cnt):
        '''
            Create a modified architecture with `top_level_pe_cnt` based on the architecture
            specified by `arch_spec`, noted that we just change the number of second-level component\n
            Side effects:\n
                Create the file `${layer_name}.arch.yaml` in /result/`layer_name` dir
            Return:\n
                full path of the newly created arch.yaml
        '''
        # load the specification and change the number of second-level component to `top_level_pe_cnt`
        obj, top_arch_spec = self._get_top_arch_specification()
        top_level_name = obj["architecture"]["subtree"][0]["subtree"][0]["name"]
        new_top_level_name = re.sub(r"0..\d+", "0.."+str(top_level_pe_cnt-1), top_level_name)
        obj["architecture"]["subtree"][0]["subtree"][0]["name"] = new_top_level_name

        # write the modified file to the working directory
        new_top_arch_spec = os.path.join(self.working_dir, "modified_arch.yaml")
        yaml.dump(obj, open(new_top_arch_spec, "w"))
        self.arch_specs = self.arch_specs.replace(top_arch_spec, new_top_arch_spec)
        return self.arch_specs

    def _invoke_timeloop_mapper(self, timeout):
        '''
            Invoking timeloop-mapper for finding optimal dataflow,
            and calling timeloop-model for obtaining communcation status
        '''


        # invoke mapper for searching for the optimal dataflow, searching `timeout` times
        executable = "timeloop-mapper"
        command = " ".join([executable, self.arch_specs, self.mapper_specs,
                            self.constraint_specs, self.prob_specs])
        try:
            if gc.timeloop_verbose:
                mapper_sp = subprocess.Popen(command, cwd=self.working_dir, shell=True, preexec_fn=os.setpgrp)
            else:
                mapper_sp = subprocess.Popen(command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                                            cwd=self.working_dir, shell=True, preexec_fn=os.setpgrp)

            # Register sigint handler
            def sigint_handler(signum, frame):
                os.killpg(os.getpgid(mapper_sp.pid), signal.SIGINT)
                exit(-1)
            signal.signal(signal.SIGINT, sigint_handler)
            if timeout:
                begin_time = time.time()
                while (time.time() - begin_time < timeout and mapper_sp.poll is not None):
                    pass
                os.killpg(os.getpgid(mapper_sp.pid), signal.SIGINT)
            mapper_sp.wait()

            print("Info: Mapper timeout!!")
        except ProcessLookupError:
            print("Info: Mapper has found optimal solution")
        print("Info: Mapper search finished")

    def _dump_invoke_timeloop_model(self):
        # dump the result file for the used dataflow
        transformer = Loop2Map()
        transformer.transform(self.mapper_map_file, self.prob_specs, self.dump_mapping_file)
        
        self.dram_tile_spatial_size = transformer.getSpatialComponentSize(0)

        # invoke model for getting communication status, single process
        executable = "timeloop-model"
        command = " ".join([executable, self.arch_specs, self.dump_mapping_file, self.prob_specs])
        if gc.timeloop_verbose:
            model_sp = subprocess.Popen(command, shell=True, cwd=self.working_dir)
        else:
            model_sp = subprocess.Popen(command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        cwd=self.working_dir, shell=True)
        model_sp.wait()

        print("Info: Communication status extraction finished")

    def _locate_components(self, comm_graph):
        '''We just assign the global buffer (notated as -1) to 0 now, PE mapping is needed
        '''
        for graph_per_datatype in comm_graph:
            def locate(x): 
                # if x == -1: 
                #     return 0
                return x
                # return random.randint(0, 512)
            graph_per_datatype["graph"] = [(locate(src), locate(dst), vol) for src, dst, vol in graph_per_datatype["graph"]]
        return comm_graph

    def get_dram_tile_spatial_size(self):
        if not hasattr(self, "dram_tile_spatial_size"):
            transformer = Loop2Map()
            transformer.transform(self.mapper_map_file, self.prob_specs, self.dump_mapping_file)
            self.dram_tile_spatial_size = transformer.getSpatialComponentSize(0)
        return self.dram_tile_spatial_size

    def get_mac_number(self):
        obj = yaml.load(open(self.prob_specs, "r"), Loader=yaml.FullLoader)

        # mac_cnt = multiplicative of output dims and weight dims
        weight_obj = obj["problem"]["shape"]["data-spaces"][0]
        output_obj = obj["problem"]["shape"]["data-spaces"][2]

        if not output_obj["name"] == "Outputs" and weight_obj["name"] == "Weight":
            raise Exception("Datatypes in problem spec misordered!")

        def extract_dim_names(datatype): return [wrapper[0][0] for wrapper in datatype["projection"]]
        weight_dim_names, output_dim_names = extract_dim_names(weight_obj), extract_dim_names(output_obj)
        mac_related_dim_names = list(set(weight_dim_names + output_dim_names))

        mac_related_dim_sizes = [obj["problem"]["instance"][dim_name] for dim_name in mac_related_dim_names]
        mac_cnt = reduce(lambda x, y: x * y, mac_related_dim_sizes)
        return mac_cnt

    def get_cycles(self):
        # extract Cycles from *.stats.txt
        cycle_pattern = re.compile(r"Cycles: ([0-9]+)")
        with open(os.path.join(self.working_dir, self.model_stats_file), "r") as f:
            for line in f:
                match_obj = cycle_pattern.match(line)
                if match_obj:
                    return int(match_obj.group(1))
        raise Exception("cycle not found!!")

    def collect_comm_status(self):
        '''
            Parse communication.yaml and traverse data to a nested list indexed with: [tile][pv]\n
            Calculate injection rate based on cycles extracted from .stats.txt\n
            Return:\n
                comm_bank:
                    ele["rate"]: injection rate (packet per cycle)\n
                    ele["interval"]: interval cycles between two transmission packets\n
        '''

        if gc.dump_comm_status:
            self._dump_invoke_timeloop_model()

        # parse files and calculate some useful variables
        yaml_obj = yaml.load(open(self.comm_report_file, "r"), Loader=yaml.FullLoader)

        # traverse comm_bank, noted that comm_bank[tile][pv] denotes traffics of *tile* and *pv*
        comm_bank = [[[] for j in range(3)] for i in range(10)]
        for datatype in yaml_obj["communication"]:
            pv = datatype["pv"]
            for request in datatype["contents"]:
                if request["volume"] != 0:
                    comm_bank[request["tile"]][pv].append(request)

        cycles = self.get_cycles()

        # calculate some useful statistic values
        for tile in comm_bank:
            for pv in tile:
                for ele in pv:
                    ele["pkt_interval"] = cycles / ele["cnt"]
                    ele["pkt_rate"] = 1 / ele["pkt_interval"]
                    ele["bit_rate"] = 8 * ele["volume"] * ele["pkt_rate"]   # average bit rate
                    ele["bit_volume"] = 8 * ele["volume"]
                    ele["ele_volume"] = ele["volume"]
                    del ele["volume"]

        return comm_bank

    def search_optimal_dataflow(self, timeout):
        # search for optimal dataflows
        try:
            self._invoke_timeloop_mapper(timeout)
        except FileNotFoundError:
            print("No valid solution found, please check timeloop configuration")
            exit(-1)

    def run_with_gc(self, analyze):
        '''Similar with `run`, but use global_control to store variables
        '''
        # changing working dir to /result/layer_name directory to avoid file pollution

        old_cwd = os.getcwd()
        os.chdir(self.working_dir)
        print("Info:", "Working for", self.layer_name)
        
        if self.top_level_cnt:
            self._modify_arch_specification(self.top_level_cnt)
        if gc.search_dataflow:
            self.search_optimal_dataflow(gc.timeout)
        
        # comm_bank: full information of communication status
        comm_bank = self.collect_comm_status()
        # bind function `analyze` to myself
        bounded_analyze = types.MethodType(analyze, self)
        # perform analysis
        ret = bounded_analyze(comm_bank)

        # recover the working directory
        os.chdir(old_cwd)
        seperator = "====================== FINISH =========================\n\n"
        print(seperator)

        return ret


    def run(self, analyze, top_level_pe_cnt=None, search_dataflow=False, timeout=None):
        '''Find optimal dataflow and perform analysis specified by callback function `analysis`\n
        '''
        print("Warning: This function is deprecated, please use `run_with_gc`")
        # changing working dir to /result/layer_name directory to avoid file pollution
        old_cwd = os.getcwd()
        os.chdir(self.working_dir)
        print("Info:", "Working for", self.layer_name)

        if top_level_pe_cnt:
            self._modify_arch_specification(top_level_pe_cnt)
        # # perform dataflow searching
        if search_dataflow:
            self.search_optimal_dataflow(timeout)

        # comm_bank: full information of communication status
        comm_bank = self.collect_comm_status()
        # bind function `analyze` to myself
        bounded_analyze = types.MethodType(analyze, self)
        # perform analysis
        ret = bounded_analyze(comm_bank)

        # recover the working directory
        os.chdir(old_cwd)
        seperator = "====================== FINISH =========================\n\n"
        print(seperator)

        return ret


if __name__ == "__main__":
    layer = Layer("resnet50_layer54.yaml", model_dir="resnet50")
    # layer = Layer("vgg16_layer")
    layer.get_mac_number()
    # layer.run(lambda x, y: y, search_dataflow=False)
    layer.run_with_gc(lambda x, y: y)
