import argparse
import yaml
import re
import os

def getArgumentParser():
    example_text = '''example:

    python3 interface.py -bm benchmark/small_test.yaml -d 4 -fr 1024-1024-512 tesd --buffersize 2048 --bufferbw 512 --macnum 16 --nocbw 128
    '''

    parser = argparse.ArgumentParser(description="FOCUS Testing", 
                                     epilog=example_text, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-bm", "--benchmark", dest="bm", type=str, metavar="benchmark/test.yaml",
                        default="benchmark/small_test.yaml", help="Spec file of task to run")
    parser.add_argument("-d", "--array_diameter", dest="d", type=int, metavar="8",
                        default=8, help="Diameter of the PE array")
    parser.add_argument("-fr", "--flit_size_range", dest="fr", type=str, metavar="Fmin-Fmax-Step",
                        default="1024-1024-512", help="Flit size range from Fmin to Fmax, interleave with Step")
    parser.add_argument("-b", "--batch", dest="b", type=int, default=1, metavar="4")
    parser.add_argument("-debug", dest="debug", action="store_true")
    parser.add_argument("mode", type=str, metavar="tgesf", default="",
                        help="Running mode, t: invoke timeloop-mapper, g: use fake trace generator, \
                              e: invoke timeloop-model, s: simulate baseline, f: invoke focus scheduler \
                              d: ONLY dump the trace file, do nothing else")

    parser.add_argument("-bs", "--buffersize", dest="bs", type=int, default=2048)
    parser.add_argument("-bbw", "--bufferbw", dest="bbw", type=int, default=512)
    parser.add_argument("-mn", "--macnum", dest="mn", type=int, default=15)
    parser.add_argument("-nbw", "--nocbw", dest="nbw", type=int, default=128)

    return parser


if __name__ == '__main__':
    parser = getArgumentParser()
    args = parser.parse_args()


    f = open('./database/arch/simba_512gops_256core.yaml','r+')
    arch_yaml = yaml.safe_load(f)
    arch_yaml["architecture"]["subtree"][0]["subtree"][0]["local"][0]['attributes']['depth'] = args.bs
    arch_yaml["architecture"]["subtree"][0]["subtree"][0]["local"][0]['attributes']['width'] = args.bbw * 8
    arch_yaml["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]['name'] = f"PE[0..{args.mn - 1}]"
    # arch_yaml["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]['local'][4]['name'] = f'LMAC[0..{args.mn * args.mn-1}]'
    f.close()
    f = open('./database/arch/simba_512gops_256core.yaml','w+')
    yaml.dump(arch_yaml, f)
    f.close()


    # f = open('./database/constraints/simba_constraints.yaml','r+')
    # arch_yaml = yaml.safe_load(f)
    # arch_yaml["mapspace_constraints"]["targets"][3]["factors"] = f'C={args.mn}'
    # arch_yaml["mapspace_constraints"]["targets"][4]["factors"] = f'M={args.mn}'
    # f.close()
    # f = open('./database/constraints/simba_constraints.yaml','w+')
    # yaml.dump(arch_yaml, f)
    # f.close()


    f = open('./simulator/runfiles/spatial_spec_ref','r+')
    lines_list = f.readlines()
    lines_list[21] = f'channel_width = {args.nbw};    // Only work for power simulation\n'
    f.close()
    f = open('./simulator/runfiles/spatial_spec_ref','w+')
    f.writelines(lines_list)
    f.close()



    # os.system(f"python3 -W ignore focus.py -bm {args.bm} -d {args.d} -b {args.b} -fr {args.fr} teds")
    os.system(f"python3 -W ignore focus.py -bm {args.bm} -d {args.d} -b {args.b} -fr {args.fr} teds")