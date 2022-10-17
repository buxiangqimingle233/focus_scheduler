import os
from random import sample
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import re
import yaml

# Copied from gnn4noc
class LayerSample():
    """ An easy-to-use representation of a fake layer's computation info.
    """
    def __init__(self, args):
        self.params = None
        if isinstance(args, str):
            self.params = self.__parse_str(args)
        elif isinstance(args, dict):
            self.params = self.__parse_dict(args)
        else:
            raise NotImplementedError
        

    def __repr__(self):
        p = self.params
        s = f"cw{p['cnt_w']}_ci{p['cnt_i']}_co{p['cnt_o']}" +\
            f"_bw{p['broadcast_w']}_bi{p['broadcast_i']}" +\
            f"_fw{p['flit_w']}_fi{p['flit_i']}_fo{p['flit_o']}" +\
            f"_dw{p['delay_w']}_di{p['delay_i']}_do{p['delay_o']}" +\
            f"_n{p['worker']}"
        return s

    def dump(self, save_root, model_name=None):
        """Dump to model config for focus scheduler.
        """
        s = self.__repr__()
        if model_name == None:
            model_name = s
        savepath = os.path.join(save_root, f"{model_name}.yaml")
        data = {s: [{s: 2}]}
        with open(savepath, 'w') as f:
            yaml.dump(data, f)

    def __empty_params(self):
        empty_params = {
            'cnt_w': None,
            'cnt_i': None,
            'cnt_o': None,
            'flit_w': None,
            'flit_i': None,
            'flit_o': None,
            'delay_w': None,
            'delay_i': None,
            'delay_o': None,
            'broadcast_w': None,
            'broadcast_i': None,
            'worker': None,
        }
        return empty_params

    def __parse_str(self, s):
        params = self.__empty_params()
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
            if t[:i] not in short2full.keys():
                continue
            key = short2full[t[:i]]
            val = int(t[i:])
            params[key] = val
        assert None not in params.values()
        return params

    def __parse_dict(self, args):
        params = self.__empty_params()
        for k in params.keys():
            params[k] = args[k]
        return params

    def to_dataframe(self):
        cnt_index = 0
        args = self.params
        layer = self.__repr__()
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

if __name__ == "__main__":
    pass