import causaldag as cd

import json
import random
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os

from collections import defaultdict
from p_tqdm import p_map
from timeit import default_timer as timer
from tqdm import tqdm

from subset_verify import *

# See https://networkx.org/documentation/stable/reference/randomness.html#randomness
random_seed = 314159
random.seed(random_seed)
np.random.seed(random_seed)

'''
Generate random tree and G(n,p) graphs. Combine edgesets, orient in acyclic fashion, and add arcs to remove v-structures.
Randomly pick a fraction of these arcs to be in T
'''
def generate_instances(dirname, num_experiments, n, p, target_probabilities):
    start = timer()
    instances = []
    for idx in tqdm(range(num_experiments), leave=False):
        fname = "{0}/n={1}-p={2}-{3}.json".format(dirname, n, p, idx)
        if os.path.exists(fname):
            # Read from file
            with open(fname, 'r') as f:
                dict_obj = json.load(f)
                nodes, arcs, all_T = dict_obj['nodes'], dict_obj['arcs'], dict_obj['all_T']
                DAG = nx.DiGraph()
                DAG.add_nodes_from(nodes)
                DAG.add_edges_from(arcs)
                instances.append((DAG, all_T))
        else:
            gnp = nx.gnp_random_graph(n, p)
            tree = nx.random_tree(n)

            # Form graph by taking both edgesets and orient in acyclic fashion
            nx_dag = nx.DiGraph()
            nx_dag.add_nodes_from(gnp.nodes)
            nx_dag.add_edges_from([(u,v) if u < v else (v,u) for (u,v) in gnp.edges])
            nx_dag.add_edges_from([(u,v) if u < v else (v,u) for (u,v) in tree.edges])

            # Remove v-structures by adding arcs. Iterate from the back.
            for w in range(n-1, -1, -1):
                for v in range(w-1, -1, -1):
                    for u in range(v-1, -1, -1):
                        if (u,w) in nx_dag.edges and (v,w) in nx_dag.edges and (u,v) not in nx_dag.edges:
                            nx_dag.add_edge(u,v)

            nodes = list(nx_dag.nodes)
            arcs = list(nx_dag.edges)
            T_sizes = [int(np.floor(len(arcs) * p)) for p in target_probabilities]
            all_T = [random.sample(arcs, T_size) for T_size in T_sizes]
            instances.append((nx_dag, all_T))

            # Write to file
            with open(fname, 'w') as f:
                json.dump(dict(nodes=nodes, arcs=arcs, all_T=all_T), f)
    end = timer()
    return instances

'''
Computes interventions needed to fully orient edges in T
T is a subset of edges of nx_dag
'''
def run_instance_subset(nx_dag, T):
    interventions, _ = subset_verification(nx_dag, T, k=1)
    assert verify_correctness(interventions, nx_dag, T)
    return len(interventions)

'''
Compute verification number of given nx_dag
'''
def run_instance_full(nx_dag):
    interventions = verification(nx_dag, k=1)
    assert verify_correctness(interventions, nx_dag)
    return len(interventions)
    
'''
Driver code
'''
def run_experiment():
    # Experimental parameters
    num_experiments = 100
    edge_probabilities = [0.001, 0.01, 0.03, 0.05, 0.1, 0.3]
    target_probabilities = [0.3, 0.5, 0.7, 1]
    graph_sizes = [20, 40, 60, 80, 100]
    plots_dirname = "figures"
    instances_dirname = "random_T_instances"
    results_dirname = "random_T_results"

    # Setup sub-directories
    os.makedirs(instances_dirname, exist_ok=True)
    os.makedirs(plots_dirname, exist_ok=True)
    os.makedirs(results_dirname, exist_ok=True)

    print("Number of CPU cores available: {0}".format(mp.cpu_count()))
    for p in tqdm(edge_probabilities, desc='p', leave=False):
        # Get experiment results
        results = defaultdict(list)
        for n in tqdm(graph_sizes, desc='n'):
            fname = "{0}/p={1}-n={2}-T={3}-numexp={4}.results".format(results_dirname, p, n, target_probabilities, num_experiments)
            if os.path.exists(fname):
                # Read from file
                print("Reading stored results from {0}".format(fname))
                with open(fname, 'r') as f:
                    results[n] = json.load(f)["results"][str(n)]
            else:
                # Get instances
                instances = generate_instances(instances_dirname, num_experiments, n, p, target_probabilities)

                # instances are of the form (DAG, all_T) where all_T = [target_probabilities[0] of arcs, target_probabilities[1] of arcs, ...]
                # Across all instances, we will form the following arrays for multiprocessing
                # DAGs = [DAG1, DAG2, ...],
                # Ts[0] = [target_probabilities[0] of arcs for DAG1, target_probabilities[0] of arcs for DAG2, ...],
                # Ts[1] = [target_probabilities[1] of arcs for DAG1, target_probabilities[1] of arcs for DAG2, ...],
                # etc...
                DAGs, all_T = zip(*instances)
                Ts = zip(*all_T)

                # Run subset verification
                stats_subset = []
                stats_ftr = []
                for target_arcs in tqdm(Ts, desc='|T|', leave=False):
                    output_subset = p_map(run_instance_subset, DAGs, target_arcs, leave=False)
                    stats_subset.append(dict(mean=np.mean(output_subset), std=np.std(output_subset)))
                
                # Run verification
                output_full = p_map(run_instance_full, DAGs, leave=False)
                stats_full = dict(mean=np.mean(output_full), std=np.std(output_full))
                    
                # Store results
                results[n] = (stats_subset, stats_full, stats_ftr)

                # Write to file
                with open(fname, 'w') as f:
                    json.dump(dict(results = results), f)


        # Plot results
        # Note: linestyle "loosely dashed": (0, (5, 10))
        plt.clf()
        X = graph_sizes
        for i in range(len(target_probabilities)):
            target_probability = target_probabilities[i]
            y_mean = [results[n][0][i]['mean'] for n in graph_sizes]
            y_std = [results[n][0][i]['std'] for n in graph_sizes]
            plt.errorbar(X, y_mean, yerr=y_std, label="Subset verification with |T| = {0} m".format(target_probability), capsize=5)
        y_mean = [results[n][1]['mean'] for n in graph_sizes]
        y_std = [results[n][1]['std'] for n in graph_sizes]
        plt.errorbar(X, y_mean, yerr=y_std, label="Verification", linestyle=(0, (5, 10)), linewidth=3, capsize=2.5)
        plt.xlabel("Size of graph (n) with p = {0}".format(p))
        plt.ylabel("Number of interventions / Atomic (subset) verifying number")
        plt.legend()
        plt.savefig("{0}/p={1}.png".format(plots_dirname, p))

    # Density plot
    density_plot = defaultdict(dict)
    for p in edge_probabilities:
        for n in graph_sizes:
            instances = generate_instances(instances_dirname, num_experiments, n, p, target_probabilities)
            DAGs, _ = zip(*instances)
            arr = np.array([len(DAG.edges) for DAG in DAGs])
            density_plot[(p,n)]['mean'] = np.mean(arr)
            density_plot[(p,n)]['std'] = np.std(arr)
    plt.clf()
    X = graph_sizes
    for p in edge_probabilities:
        y_mean = [density_plot[(p,n)]['mean'] for n in graph_sizes]
        y_std = [density_plot[(p,n)]['std'] for n in graph_sizes]
        plt.errorbar(X, y_mean, yerr=y_std, label="p = {0}".format(p), capsize=5)
    y_max = [n*(n-1)/2 for n in graph_sizes]
    plt.plot(X, y_max, label="Maximum number of edges")
    plt.xlabel("Size of graph (n)")
    plt.ylabel("Number of edges")
    plt.legend()
    plt.savefig("{0}/density.png".format(plots_dirname))

if __name__ == "__main__":
    run_experiment()

