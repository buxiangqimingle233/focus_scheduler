import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import re

# traffic = pd.DataFrame(columns=["layer", "src", "dst", "interval", "flit", "counts"])

class MicroOpGraph:

    flow_cnt = 0

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    @staticmethod
    def __hash_node(layer_, vpe_):
        return hash(repr(str(layer_) + str(vpe_)))

    def add_layer(self, streams: pd.DataFrame):
        op_graph = self.graph
        layer = streams["layer"][0]

        # Assign an unique id to every flow
        streams.loc[:, "fid"] = range(MicroOpGraph.flow_cnt, MicroOpGraph.flow_cnt + streams.shape[0])
        MicroOpGraph.flow_cnt += streams.shape[0]

        group = streams.groupby("datatype")
        worker_num = group.get_group("output").explode("src").shape[0]

        # some magic numbers ... 
        i_source_magic, w_source_magic, sink_magic = -1, -3, -2
        i_source, w_source, sink = \
            MicroOpGraph.__hash_node(layer, i_source_magic), MicroOpGraph.__hash_node(layer, w_source_magic), MicroOpGraph.__hash_node(layer, sink_magic)

        def get_prelayer_name(name):
            layer_number = re.findall(r"\d+", name)[-1]
            pre_layer_number = str(int(layer_number) - 1)
            prelayer_name = re.sub(layer_number[::-1], pre_layer_number[::-1], name[::-1])[::-1]
            return prelayer_name

        pre_layer = get_prelayer_name(layer)
        pre_layer_sinks = nx.subgraph_view(op_graph, \
            filter_node=lambda x: op_graph.nodes[x]["op_type"] == "sink" and op_graph.nodes[x]["layer"] == pre_layer)
        assert len(pre_layer_sinks.nodes) <= 1

        # Setup weight source
        w_cnt = group.get_group("weight")["counts"].iloc[0]
        w_delay = group.get_group("weight")["interval"].iloc[0]
        op_graph.add_node(w_source, layer=layer, op_type="wsrc", v_pe=w_source_magic, delay=w_delay, cnt=w_cnt)
        # Add Control signals: the weight source won't activate until its preceeding layer finishes
        for s in pre_layer_sinks:
            op_graph.add_edge(s, w_source, edge_type="control")

        # Setup input source
        i_cnt = group.get_group("input")["counts"].iloc[0]
        i_delay = group.get_group("input")["interval"].iloc[0]
        op_graph.add_node(i_source, layer=layer, op_type="insrc", v_pe=i_source_magic, delay=i_delay, cnt=i_cnt)
        # Add control signals: the input source should wait for preceeding layer to finish
        # TODO: We put hard syncronization bairrer between two adjacent layers. However, in some cases, e.g. oc-tiling to ic-tiling,
        # the suceeding layer does not have to wait for whole preceeding layer to finish, it can start once a channel is 
        # generated.

        # FIXME: What does the sink push to next layer's isource, a control signal, or the entire output data ?
        i_data_amount = group.get_group("input")["flit"].iloc[0] * group.get_group("input")["counts"].iloc[0]
        for s in pre_layer_sinks:
            op_graph.add_edge(s, i_source, edge_type="data", fid=MicroOpGraph.flow_cnt, size=i_data_amount)
            MicroOpGraph.flow_cnt += 1
            # op_graph.add_edge(s, i_source, edge_type="control")

        # Setup sink (merger)
        o_cnt = group.get_group("output")["counts"].iloc[0]
        o_delay = group.get_group("output")["interval"].iloc[0]
        op_graph.add_node(sink, layer=layer, op_type="sink", v_pe=sink_magic, delay=0, cnt=1)   # FIXME: need test

        print(streams)
        # Setup workers
        edges = {(r["src"], r["dst"]): (r["fid"], r["flit"]) for _, r in streams.explode("src").explode("dst").iterrows()}
        for w in range(worker_num):
            worker = MicroOpGraph.__hash_node(layer, w)
            op_graph.add_node(worker, layer=layer, op_type="worker", v_pe=w, delay=o_delay, cnt=o_cnt)

            # Connect weight source to the worker
            w_flow = edges[(w_source_magic, w)]
            op_graph.add_edge(w_source, worker, edge_type="data", fid=w_flow[0], size=w_flow[1])

            # Connect input source to the worker
            i_flow = edges[(i_source_magic, w)]
            op_graph.add_edge(i_source, worker, edge_type="data", fid=i_flow[0], size=i_flow[1])

            # Connect the worker to sink
            o_flow = edges[(w, sink_magic)]
            op_graph.add_edge(worker, sink, edge_type="data", fid=o_flow[0], size=o_flow[1])

    def set_physical_position(self, vir_to_phy_map):
        op_graph = self.graph
        for _, attr in op_graph.nodes(data=True):
            layer, v_pe = attr["layer"], attr["v_pe"]
            assert layer in vir_to_phy_map and v_pe in vir_to_phy_map[layer]
            attr["p_pe"] = vir_to_phy_map[layer][v_pe]


    def get_data(self):
        return self.graph

    def draw(self, fig_path):
        seed = 123467

        G = self.get_data()
        pos = nx.spring_layout(G, seed=seed)

        cmap = plt.cm.plasma
        plt.rcParams["lines.linewidth"] = 3
        plt.rcParams["lines.linewidth"] = 2.5
        plt.rcParams["lines.markersize"] = 10
        plt.rcParams["lines.markerfacecolor"] = "7f7f7f"
        plt.rcParams["lines.markeredgecolor"] = "ffffff"
        plt.rcParams["lines.markeredgewidth"] = 10

        nodes = nx.draw_networkx_nodes(G, pos, node_size=80, node_color="#000000")
        edges = nx.draw_networkx_edges(
            G,
            pos,
            node_size=80,
            arrowstyle="->",
            arrowsize=10,
            edge_cmap=cmap,
            width=2,
            edge_color="#7f7f00"
        )

        ax = plt.gca()
        ax.set_axis_off()
        plt.savefig(fig_path)
        plt.close()
