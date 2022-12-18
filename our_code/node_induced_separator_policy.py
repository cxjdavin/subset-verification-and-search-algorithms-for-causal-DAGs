from causaldag import DAG

import random
import networkx as nx
import numpy as np

from collections import defaultdict
import math
import sys
sys.path.insert(0, './PADS')
import LexBFS

'''
Verify that the peo computed is valid
For any node v, all neighbors that appear AFTER v forms a clique (i.e. pairwise adjacent)
'''
def verify_peo(adj_list, actual_to_peo, peo_to_actual):
    assert len(adj_list) == len(actual_to_peo)
    assert len(adj_list) == len(peo_to_actual)
    try:
        n = len(adj_list)
        for i in range(n):
            v = peo_to_actual[i]
            later_neighbors = [u for u in adj_list[v] if actual_to_peo[u] > i]
            for u in later_neighbors:
                for w in later_neighbors:
                    assert u == w or u in adj_list[w]
    except Exception as err:
        print('verification error:', adj_list, actual_to_peo, peo_to_actual)
        assert False

'''
Compute perfect elimination ordering using PADS
Source: https://www.ics.uci.edu/~eppstein/PADS/ABOUT-PADS.txt
'''
def peo(adj_list, nodes):
    n = len(nodes)

    G = dict()
    for v in nodes:
        G[v] = adj_list[v]
    lexbfs_output = list(LexBFS.LexBFS(G))

    # Reverse computed ordering to get actual perfect elimination ordering
    output = lexbfs_output[::-1]
    
    actual_to_peo = dict()
    peo_to_actual = dict()
    for i in range(n):
        peo_to_actual[i] = output[i]
        actual_to_peo[output[i]] = i

    # Sanity check: verify PADS's peo output
    # Can comment out for computational speedup
    #verify_peo(adj_list, actual_to_peo, peo_to_actual)
    
    return actual_to_peo, peo_to_actual

'''
Given a connected chordal graph on n nodes, compute the 1/2-clique graph separator
FAST CHORDAL SEPARATOR algorithm of [GRE84]
Reference: [GRE84] A Separator Theorem for Chordal Graphs
'''
def compute_clique_graph_separator(adj_list, nodes, subgraph_nodes):
    n = len(nodes)

    # Compute perfect elimination ordering via lex bfs
    actual_to_peo, peo_to_actual = peo(adj_list, nodes)

    w = [0] * n
    for v in subgraph_nodes:
        w[actual_to_peo[v]] = n/len(subgraph_nodes)
    total_weight = sum(w)
    # There may be rounding issues, so check with np.isclose
    assert np.isclose(total_weight, n)

    # Compute separator
    peo_i = 0
    while w[peo_i] <= total_weight/2:
        # w[i] is the weight of the connected component of {v_0, ..., v_i} that contains v_i
        # v_k <- lowest numbered neighbor of v_i with k > i
        k = None
        for j in adj_list[peo_to_actual[peo_i]]:
            if actual_to_peo[j] > peo_i and (k is None or actual_to_peo[j] < actual_to_peo[k]):
                k = j
        if k is not None:
            w[actual_to_peo[k]] += w[peo_i]
        peo_i += 1

    # i is the minimum such that some component of {v_0, ..., v_i} weighs more than total+weight/2
    # C <- v_i plus all of v_{i+1}, ..., v_n that are adjacent to v_i
    C = [peo_to_actual[peo_i]]
    for j in adj_list[peo_to_actual[peo_i]]:
        if actual_to_peo[j] > peo_i:
            C.append(j)
    return C

'''
Adaptation of [CSB22] separator policy for node-induced subgraph search
Assumption on input: The given subset of target edges are all edges within the node-induced subgraph of interest
'''
def node_induced_separator_policy(dag: DAG, k: int, target_edges: set, verbose: bool = False) -> set:
    subgraph_nodes = set()
    for u,v in target_edges:
        subgraph_nodes.add(u)
        subgraph_nodes.add(v)

    intervened_nodes = set()
    current_cpdag = dag.cpdag()

    intervention_queue = []
    while len(target_edges.difference(current_cpdag.arcs)) > 0:
        if verbose: print(f"Remaining edges: {current_cpdag.num_edges}")
        node_to_intervene = None

        undirected_portions = current_cpdag.copy()
        undirected_portions.remove_all_arcs()
        
        # Cannot directly use G = undirected_portions.to_nx() because it does not first add the nodes
        # We need to first add nodes because we want to check if the clique nodes have incident edges
        # See https://causaldag.readthedocs.io/en/latest/_modules/causaldag/classes/pdag.html#PDAG 
        G = nx.Graph()
        G.add_nodes_from(undirected_portions.nodes)
        G.add_edges_from(undirected_portions.edges)

        intervention = None
        while len(intervention_queue) > 0 and intervention is None:
            intervention = intervention_queue.pop()
    
            # If all incident edges already oriented, skip this intervention
            if sum([G.degree[node] for node in intervention]) == 0:
                intervention = None

        if intervention is None:
            assert len(intervention_queue) == 0

            # Compute 1/2-clique separator for each connected component of size >= 2
            clique_separator_nodes = []
            for cc_nodes in nx.connected_components(G):
                if len(cc_nodes) == 1:
                    continue
                cc = G.subgraph(cc_nodes)
                
                # Map indices of subgraph into 0..n-1
                n = len(cc.nodes())
                map_indices = dict()
                unmap_indices = dict()
                for v in cc.nodes():
                    map_indices[v] = len(map_indices)
                    unmap_indices[map_indices[v]] = v

                # Extract adj_list and nodes of subgraph
                nodes = []
                adj_list = []
                for v, nbr_dict in cc.adjacency():
                    nodes.append(map_indices[v])
                    adj_list.append([map_indices[x] for x in list(nbr_dict.keys())])

                # Compute clique separator for this connected component then add to the list
                cc_subgraph_nodes = []
                for v in cc.nodes():
                    if v in subgraph_nodes:
                        cc_subgraph_nodes.append(map_indices[v])
                if len(cc_subgraph_nodes) > 0:
                    clique_separator_nodes += [unmap_indices[v] for v in compute_clique_graph_separator(adj_list, nodes, cc_subgraph_nodes)]

            assert len(clique_separator_nodes) > 0
            if k == 1 or len(clique_separator_nodes) == 1:
                intervention_queue = [set([v]) for v in clique_separator_nodes]
            else:
                # Setup parameters. Note that [SKDV15] use n and x+1 instead of h and L
                h = len(clique_separator_nodes)
                k_prime = min(k, h/2)
                a = math.ceil(h/k_prime)
                assert a >= 2
                L = math.ceil(math.log(h,a))
                assert pow(a,L-1) < h and h <= pow(a,L)

                # Execute labelling scheme
                S = defaultdict(set)
                for d in range(1, L+1):
                    a_d = pow(a,d)
                    r_d = h % a_d
                    p_d = h // a_d
                    a_dminus1 = pow(a,d-1)
                    r_dminus1 = h % a_dminus1 # Unused
                    p_dminus1 = h // a_dminus1
                    assert h == p_d * a_d + r_d
                    assert h == p_dminus1 * a_dminus1 + r_dminus1
                    for i in range(1, h+1):
                        node = clique_separator_nodes[i-1]
                        if i <= p_d * a_d:
                            val = (i % a_d) // a_dminus1
                        else:
                            val = (i - p_d * a_d) // math.ceil(r_d / a)
                        if i > a_dminus1 * p_dminus1:
                            val += 1
                        S[(d,val)].add(node)

                # Store output
                intervention_queue = list(S.values())
            assert len(intervention_queue) > 0    
            intervention = intervention_queue.pop()

        # Intervene on selected node(s) and update the CPDAG
        assert intervention is not None
        assert len(intervention) <= k
        intervention = frozenset(intervention)
        intervened_nodes.add(intervention)
        current_cpdag = current_cpdag.interventional_cpdag(dag, intervention)

    return intervened_nodes
