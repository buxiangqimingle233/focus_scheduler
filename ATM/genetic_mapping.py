import yaml
from functools import reduce
import genetic as genetic
import mapping_sim
import datetime

def display(candidate, startTime):
    timeDiff = datetime.datetime.now() - startTime
    print("{}\t{}\t{}".format(
        candidate.Genes, candidate.Fitness, timeDiff))

class genetic_mapping():
    def __init__(self, task_list, array_size):
        self.geneset = []
        obj = yaml.load(open(task_list, "r"), Loader=yaml.FullLoader)
        layer_names = []
        for model in obj.values():
            layer_names += reduce(lambda x, y: x + y, map(lambda x: list(x.keys()), model))
        layer_num = len(layer_names)
        for i in range(layer_num):
            self.geneset.append(i)
        self.target_len = array_size * array_size

    def run(self):
        start_time = datetime.datetime.now()

        def fnDisplay(candidate):
            display(candidate, start_time)

        optimalPerformance = 1000
        best_mapping = genetic.get_best(mapping_sim.get_performance, self.target_len, optimalPerformance,
                                        self.geneset, fnDisplay)

if __name__ == "__main__":
    task_list = "task_list.yaml"
    gm = genetic_mapping(task_list, 4)
    gm.run()
    