import os
from random import sample
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import re

# Generate single layer workload similar to Timeloop's output
class FakeWorkload():
    
    def encode_dataframe(self, args=None, layer=None):
        """Generate fake dataframe"""
        if layer == None:
            assert args != None
            layer = self.encode_string(args)
        if args == None:
            assert layer != None
            args = self.parse_string(layer)

        cnt_index = 0
        df = pd.DataFrame(columns=['index', 'counts', 'datatype', 'dst', 'flit',\
            'interval', 'layer', 'src', 'delay'])
        
        # add edges from src to worker
        for suffix in ['w', 'i']:
            tmp_dict = {
                'layer': layer,
                'src': -3 if suffix == 'w' else -1,
                'dst': None,
                'counts': args['cnt_'+suffix],
                'flit': args['flit_'+suffix],
                'interval': args['delay_'+suffix],
                'delay': np.nan,
                'datatype': "weight" if suffix == 'w' else "input",
                'index': None,
            }
            if args['broadcast_'+suffix]:
                tmp_dict['dst'] = list(range(args['worker']))
                tmp_dict['index'] = cnt_index
                cnt_index += 1
                df = df.append(tmp_dict, ignore_index=True)
            else:
                for d in range(args['worker']):
                    tmp_dict['dst'] = d
                    tmp_dict['index'] = cnt_index
                    cnt_index += 1
                    df = df.append(tmp_dict, ignore_index=True)

        # add edges from worker to sink
        for worker in range(args['worker']):
            tmp_dict = {
                'layer': layer,
                'src': worker ,
                'dst': -2,
                'counts': args['cnt_o'],
                'flit': args['flit_o'],
                'interval': args['delay_o'],
                'delay': args['delay_o'],
                'datatype': "output",
                'index': cnt_index,
            }
            cnt_index += 1
            df = df.append(tmp_dict, ignore_index=True)

        return df

    def encode_string(self, args):
        """Encode args into string"""
        s = f"cw{args['cnt_w']}_ci{args['cnt_i']}_co{args['cnt_o']}" +\
            f"_bw{args['broadcast_w']}_bi{args['broadcast_i']}" +\
            f"_fw{args['flit_w']}_fi{args['flit_i']}_fo{args['flit_o']}" +\
            f"_dw{args['delay_w']}_di{args['delay_i']}_do{args['delay_o']}" +\
            f"_n{args['worker']}"
        return s

    def parse_string(self, s):
        """Parse encoded string into args"""
        args = dict()
        short2full = {
            "cw": "cnt_w",
            "ci": "cnt_i",
            "co": "cnt_o",
            "bw": "broadcast_w",
            "bi": "broadcast_i",
            "fw": "flit_w",
            "fi": "flit_i",
            "fo": "flit_o",
            "dw": "delay_w",
            "di": "delay_i",
            "do": "delay_o",
            "n": "worker",
        }
        for t in s.split('_'):
            i = re.search('\d+', t).span()[0]
            key = short2full[t[:i]]
            val = int(t[i:])
            args[key] = val
        return args

if __name__ == "__main__":
    fake_agent = FakeWorkload()
    df = fake_agent.encode_dataframe(layer="cw36_ci1764_co1764_bw0_bi1_fw5_fi22_fo7_dw343_di7_do7_n4")
    print(df)