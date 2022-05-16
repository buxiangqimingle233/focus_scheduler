
def aging_sim():
    working_dir = "./simulator/tasks/bert"
    array_diameter = 6
    op_time = list(0 for _ in range(array_diameter ** 2))
    for core in range(array_diameter ** 2):
    # for core in range(1):
        f = open(f"{working_dir}/c{core}.inst", "r")
        lines = f.readlines()
        for line in lines:
            if "CPU.sleep" in line:
                op_time[core] += int(line.split(" ")[-1])
    print(op_time)

if __name__ == "__main__":
    aging_sim()