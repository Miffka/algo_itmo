import argparse
import os.path as osp
import pickle
import timeit
import tqdm

import networkx as nx
import numpy as np

from algo_utils.config import system_config


def dijkstra_time(graph):
    data = []
    for i, start_v in enumerate(tqdm.tqdm(list(graph.nodes)[:-1], desc="Generating Dijkstra")):
        for end_v in list(graph.nodes)[i:]:
            t = timeit.timeit(stmt=f"nx.dijkstra_path(graph, {start_v}, {end_v})", globals=globals(), number=10)
            path = nx.dijkstra_path(graph, start_v, end_v)
            data.append({"vertices": [start_v, end_v], "path": path, "path_len": len(path), "time": t})
    return data


def bf_time(graph):
    data = []
    for i, start_v in enumerate(tqdm.tqdm(list(graph.nodes)[:-1], desc="Generating Bellman-Ford")):
        for end_v in list(graph.nodes)[i:]:
            t = timeit.timeit(stmt=f"nx.bellman_ford_path(graph, {start_v}, {end_v})", globals=globals(), number=10)
            path = nx.bellman_ford_path(graph, start_v, end_v)
            data.append({"vertices": [start_v, end_v], "path": path, "path_len": len(path), "time": t})
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate data for task5")
    parser.add_argument("--random_state", type=int, default=24, help="Random state for generator")
    parser.add_argument("--output", default=osp.join(system_config.data_dir, "task6.pkl"), help="Output file")

    args = parser.parse_args()

    np.random.seed(args.random_state)

    graph = nx.generators.random_graphs.gnm_random_graph(n=100, m=500, seed=args.random_state)
    for (u, v) in graph.edges():
        graph.edges[u,v]['weight'] = np.random.randint(1, 1000)

    dijkstra_data = dijkstra_time(graph)
    bf_data = bf_time(graph)

    # grid = nx.generators.lattice.grid_2d_graph()

    with open(args.output, "wb") as fout:
        pickle.dump(
            {"g": graph, "dijkstra": dijkstra_data, "bf": bf_data},
            fout,
        )
    print(f"Data were saved to {args.output}")
