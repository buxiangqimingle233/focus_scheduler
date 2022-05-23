import os


class Variables:

    @staticmethod
    def gen_working_dir(root, task_name):
        dest_dir = os.path.join(root, "tasks", task_name)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        return dest_dir

    @staticmethod
    def get_routing_board_path(root, task_name) -> str:
        dest_dir = os.path.join(root, "tasks", task_name)
        assert os.path.exists(dest_dir)
        rb_path = os.path.join(dest_dir, "routing_board")
        return rb_path
    
    @staticmethod
    def get_trace_file_path_dict(root, task_name, array_size) -> dict:
        dest_dir = os.path.join(root, "tasks", task_name)
        assert os.path.exists(dest_dir)
        names = Variables.get_trace_file_names(array_size)
        trace_files = {i: os.path.join(dest_dir, names[i]) for i in range(array_size)}
        return trace_files
    
    @staticmethod
    def get_trace_file_names(array_size) -> list:
        return ["c{}.inst".format(i) for i in range(array_size)]
    
    @staticmethod
    def get_ref_spec_path(root) -> str:
        spec_path = os.path.join(root, "runfiles", "spatial_spec_ref")
        return spec_path
    
    @staticmethod
    def get_spec_path(root, task_name) -> str:
        dest_dir = os.path.join(root, "tasks", task_name)
        assert os.path.exists(dest_dir)
        spec_path = os.path.join(dest_dir, "spatial_spec")
        return spec_path
    
    @staticmethod
    def get_inst_latency_path(root) -> str:
        inst_latency_path = os.path.join(root, "runfiles", "instr_latency")
        assert os.path.exists(inst_latency_path)
        return inst_latency_path