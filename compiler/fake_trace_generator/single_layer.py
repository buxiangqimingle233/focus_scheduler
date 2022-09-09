import os
from random import sample
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import re

# Generate single layer workload similar to Timeloop's output

class Workload():
    def __init__(self, args):
        self.args = args
    
    def __repr__(self) -> str:
        return self.encode_string()

    def encode_dataframe(self):
        """Encode args into dataframe"""
        cnt_index = 0
        df = pd.DataFrame(columns=['index', 'counts', 'datatype', 'dst', 'flit',\
            'interval', 'layer', 'src', 'delay'])
        
        # add edges from src to worker
        for suffix in ['w', 'i']:
            tmp_dict = {
                'layer': repr(self),
                'src': -3 if suffix == 'w' else -1,
                'dst': None,
                'counts': self.args['cnt_'+suffix],
                'flit': self.args['flit_'+suffix],
                'interval': self.args['delay_'+suffix],
                'delay': np.nan,
                'datatype': "weight" if suffix == 'w' else "input",
                'index': None,
            }
            if self.args['broadcast_'+suffix]:
                tmp_dict['dst'] = list(range(self.args['worker']))
                tmp_dict['index'] = cnt_index
                cnt_index += 1
                df = df.append(tmp_dict, ignore_index=True)
            else:
                for d in range(self.args['worker']):
                    tmp_dict['dst'] = d
                    tmp_dict['index'] = cnt_index
                    cnt_index += 1
                    df = df.append(tmp_dict, ignore_index=True)

        # add edges from worker to sink
        for worker in range(self.args['worker']):
            tmp_dict = {
                'layer': repr(self),
                'src': worker ,
                'dst': -2,
                'counts': self.args['cnt_o'],
                'flit': self.args['flit_o'],
                'interval': self.args['delay_o'],
                'delay': self.args['delay_o'],
                'datatype': "output",
                'index': cnt_index,
            }
            cnt_index += 1
            df = df.append(tmp_dict, ignore_index=True)

        return df

    def encode_string(self):
        """Encode args into string"""
        s = f"cw{self.args['cnt_w']}_ci{self.args['cnt_i']}_co{self.args['cnt_o']}" +\
            f"_bw{self.args['broadcast_w']}_bi{self.args['broadcast_i']}" +\
            f"_fw{self.args['flit_w']}_fi{self.args['flit_i']}_fo{self.args['flit_o']}" +\
            f"_dw{self.args['delay_w']}_di{self.args['delay_i']}_do{self.args['delay_o']}" +\
            f"_n{self.args['worker']}"
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