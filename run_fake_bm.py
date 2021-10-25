import focus
from utils import global_control as gc

widths = range(512, 4097, 512)
search_dataflow = False
result_file = "fake_trace.csv"

def run():
    gc.trace_gen_backend = "generator"
    gc.search_dataflow = False
    gc.result_file = result_file

    for w in widths:
        gc.arch_config["w"] = w
        focus.run()


if __name__ == "__main__":
    run()