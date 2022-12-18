import causaldag as cd

import random
import os
import networkx as nx

from networkx.algorithms import bipartite
from collections import defaultdict
from functools import cmp_to_key
from tqdm import tqdm
from timeit import default_timer as timer

from verify import *

def verification(G, k, verbose=False):
    if k == 1:
        atomic_intervention_vertices = atomic_verification(G)
        interventions = [[v] for v in atomic_intervention_vertices]
        if verbose:
            print("interventions = {0}".format(interventions))
        return interventions
    else:
        assert False

'''
Given an undirected graph H (networkx graph object), output the minimum vertex cover.
Since H is a forest (and hence is bipartite), we can use Konig's theorem to compute the minimum vertex cover.
However, networkx requires us to process connected components one at a time.
Konig's theorem: In bipartite graph, size maximum matching = size of minimum vertex cover.
'''
def compute_minimum_vertex_cover(H):
    assert bipartite.is_bipartite(H)
    mvc = set()
    for V in nx.connected_components(H):
        cc = H.subgraph(V)
        assert bipartite.is_bipartite(cc)
        matching_for_cc = nx.bipartite.eppstein_matching(cc)
        mvc_for_cc = nx.bipartite.to_vertex_cover(cc, matching_for_cc)
        mvc.update(mvc_for_cc)
    assert is_vertex_cover(H, mvc)
    return mvc
'''
Get (directed) adjacency list from networkx graph
'''
def get_adj_list(G):
    adj_list = dict()
    for node, nbrdict in G.adjacency():
        adj_list[node] = [nbr for nbr in nbrdict.keys()]
    return adj_list

'''
Compute Euler tour data structure. Works even when G is undirected.
See: https://usaco.guide/gold/tree-euler?lang=cpp
'''
def compute_euler_tour(G, root):
    adj_list = get_adj_list(G)
    tau = []
    first = dict()
    last = dict()
    def dfs(node, parent):
        # First visit to node
        tau.append(node)
        first[node] = len(tau)

        # Visit subtrees
        for nbr in adj_list[node]:
            if nbr != parent:
                dfs(nbr, node)

            # Returned from subtree T_{nbr}
            tau.append(node)

        # Final visit to node
        last[node] = len(tau)
    dfs(root, None)
    return tau, first, last

'''
Compute R(G,v)
'''
def compute_R(G,v):
    dag = cd.DAG.from_nx(G)
    cpdag = dag.interventional_cpdag([{v}], cpdag=dag.cpdag())
    return cpdag.arcs

'''
Compute Hasse diagram of a DAG. Returns root and the Hasse tree
- For each starting vertex, compute longest paths and keep all length one paths
- To compute longest path, first compute topological ordering (can be reused)

n = |V|, m = |E|
Topological sort: O(n + m)
Longest path from each starting vertex: O(n * (n+m))
Total: O(n * (n+m))

See: https://en.wikipedia.org/wiki/Transitive_reduction#Computing_the_reduction_in_sparse_graphs
'''
def compute_Hasse_diagram_for_DAG(G):
    assert nx.is_directed_acyclic_graph(G)
    H = nx.DiGraph()

    # Topological sort
    topo_ordering = list(nx.topological_sort(G))

    # For each source vertex, compute longest path
    n = G.number_of_nodes()
    adj_list = get_adj_list(G)
    for source in range(n):
        length = [-float('inf')] * n
        length[source] = 0
        for i in range(n):
            v = topo_ordering[i]
            for nbr in adj_list[v]:
                length[nbr] = max(length[nbr], length[v] + 1)
                    
        # Keep all length one paths (i.e. direct child edges)
        for v in range(n):
            if length[v] == 1:
                H.add_edge(source, v)

    assert nx.is_tree(H)
    return topo_ordering[0], H

'''
Compute sets and indices

prec-ordering:
[a,b] < [c,d] <=> (first[a] < first[c]) or (first[a] = first[c] and last[b] > last[d])

J = sorted list of indices, with respect to prec-ordering
E[v] = { [a,b] \in J : b = v }
M[v] = { [a,b] \in J : v \in (a,b) }
S[v] = { [a,b] \in J : a = v }
W[v] = { [a,b] \in J : a,b \in V(T_v) \setminus {v} }
I[v] = E[v] union M[v] union S[v] union W[v]
B[v] = S[v] union W[v]
C[v] = E[v] union M[v] union S[v]

e_v[v] = max(E[v]) if non-empty else -INF
a_y[y] = min(B[y]) if non-empty else INF
b_vy[(v,y)] = min((C[v] intersect I[y]) union B[y]) if non-empty else INF
'''
def prepare_for_DP(n, adj_list, intervals, first, last, verbose=False):
    # Prune superset intervals
    to_prune = [(b, last[a], a, b) for a,b in intervals]
    to_prune.sort()
    intervals = []
    for _, _, a, b in to_prune:
        if len(intervals) == 0 or intervals[-1][1] != b:
            intervals.append((a,b))

    # Sort intervals according to prec-ordering
    # Can ignore "return 0" since intervals will never be equal ordering
    def prec_ordering(ab,cd):
        a,b = ab
        c,d = cd
        if first[a] < first[c] or (a == c and last[b] > last[d]):
            return -1
        else:
            return 1
    intervals.sort(key=cmp_to_key(prec_ordering))
    J = intervals

    # Using Euler tour mappings, determine if node is in subtree T_v in O(1)
    def node_is_in_Tv(node, v):
        return first[v] <= first[node] and last[node] <= last[v]

    E = defaultdict(set)
    M = defaultdict(set)
    S = defaultdict(set)
    W = defaultdict(set)
    for z in range(len(J)):
        idx, interval = z, J[z]
        a,b = interval
        for v in range(n):
            if b == v:
                E[v].add(idx)
            elif a == v:
                S[v].add(idx)
            elif node_is_in_Tv(b,v):
                if node_is_in_Tv(v,a):
                    M[v].add(idx)
                else:
                    W[v].add(idx)

    I = [E[v].union(M[v]).union(S[v]).union(W[v]) for v in range(n)]
    B = [S[v].union(W[v]) for v in range(n)]
    C = [E[v].union(M[v]).union(S[v]) for v in range(n)]

    e_v = [max(E[v]) if len(E[v]) > 0 else -float('inf') for v in range(n)]
    a_y = [min(B[v]) if len(B[v]) > 0 else float('inf') for v in range(n)]
    b_vy = dict()
    for v in range(n):
        for y in adj_list[v]:
            possibilities = C[v].intersection(I[y]).union(B[y])
            b_vy[(v,y)] = min(possibilities) if len(possibilities) > 0 else float('inf')

    if verbose:
        print("J = {0}".format(J))
        print("E = {0}".format(E))
        print("M = {0}".format(M))
        print("S = {0}".format(S))
        print("W = {0}".format(W))
        print("I = {0}".format(I))
        print("B = {0}".format(B))
        print("C = {0}".format(C))
        print("e_v = {0}".format(e_v))
        print("a_y = {0}".format(a_y))
        print("b_vy = {0}".format(b_vy))

    return e_v, a_y, b_vy

def minimum_interval_cover_on_rooted_tree(T, root, intervals, verbose):
    assert nx.is_tree(T) and nx.is_connected(T.to_undirected()) and nx.is_directed(T)
    adj_list = get_adj_list(T)

    # Compute Euler tour mappings
    tau, first, last = compute_euler_tour(T, root)

    # Pre-compute indices for dynamic programming
    e_v, a_y, b_vy = prepare_for_DP(T.number_of_nodes(), adj_list, intervals, first, last, verbose)

    # Dynamic programming to compute minimum sized interval cover
    memo = dict()
    def dp(v,idx):
        if idx == float('inf'): # No more intervals to handle
            return 0
        elif (v,idx) in memo:
            return memo[(v,idx)]
        else:
            # Compute alpha and beta
            alpha_v, beta_v = 1, 0
            for y in adj_list[v]:
                alpha_v += dp(y, max(a_y[y], idx))
                beta_v += dp(y, max(b_vy[(v,y)], idx))

            if e_v[v] >= idx:
                # There is some interval ending at v that has not been handled, take v
                memo[(v,idx)] = alpha_v
            else:
                # Choose best of taking or ignoring v
                memo[(v,idx)] = min(alpha_v, beta_v)
            return memo[(v,idx)]

    # Extract set of vertices that correspond to dynamic programming solution
    def extract_vertices_from_dp(v,idx):
        if idx == float('inf'): # No more intervals to handle
            return []
        else:
            # Compute alpha and beta
            alpha_v, beta_v = 1, 0
            for y in adj_list[v]:
                alpha_v += dp(y, max(a_y[y], idx))
                beta_v += dp(y, max(b_vy[(v,y)], idx))

            output = []
            if alpha_v <= beta_v:
                # Take v
                output.append(v)
                for y in adj_list[v]:
                    output += extract_vertices_from_dp(y, max(a_y[y], idx))
            else:
                # Ignore v
                for y in adj_list[v]:
                    output += extract_vertices_from_dp(y, max(b_vy[(v,y)], idx))
            return output
            
    min_size = dp(root, 0)
    extracted_vertices = extract_vertices_from_dp(root, 0)
    if verbose:
        print("min_size = {0}".format(min_size))
        print("extracted_vertices = {0}".format(extracted_vertices))
    return extracted_vertices

def subset_atomic_verification_for_single_component(cc, T_cc, verbose=False):
    assert nx.is_connected(cc.to_undirected()) and nx.is_directed(cc)

    # Compute Hasse tree
    root, H = compute_Hasse_diagram_for_DAG(cc)
    adj_list = get_adj_list(H)

    # Compute Euler tour mappings
    tau, first, last = compute_euler_tour(H, root)

    # Intervene on every vertex one by one to learn R(G,v)
    start = timer()
    R = dict()
    for v in cc.nodes:
        R[v] = compute_R(cc,v)
    end = timer()
    compute_R_runtime = end - start

    # Compute R^{-1}(G, u -> v) for each u -> v in T
    R_inverse = defaultdict(list)
    for u,v in T_cc:
        for w in cc.nodes:
            if (u,v) in R[w]:
                R_inverse[(u,v)].append(w)

    # Compute intervals for each target edge
    intervals = []
    for u,v in T_cc:
        # Sort w in R^{-1} using first[w] to extract interval endpoints
        sorted_w = sorted([(first[w], w) for w in R_inverse[(u,v)]])
        intervals.append((sorted_w[0][1], sorted_w[-1][1]))

    if verbose:
        print("Hasse edges = {0}".format(H.edges))
        print("intervals = {0}".format(T_cc))
        print("R = {0}".format(R))
        print("R_inverse = {0}".format(R_inverse))
    return minimum_interval_cover_on_rooted_tree(H, root, intervals, verbose), compute_R_runtime

def compute_v_structs(G):
    directed_adjlist = get_adj_list(G)
    adjlist = defaultdict(set)
    parents = defaultdict(set)
    for u in G.nodes:
        for v in directed_adjlist[u]:
            print(u,v)
            adjlist[u].add(v)
            adjlist[v].add(u)
            parents[v].add(u)
    v_structure_arcs = [(u,v) for u,v in G.edges if len(parents[v] - {u} - adjlist[u]) > 0]
    return v_structure_arcs

'''
Given a DAG and target edges T, run DP
Note that this is *not* a search algorithm but a verification algorithm.
The input to this algorithm is the ground truth dag instead of the essential graph dag.cpdag()
G is a networkx DiGraph
'''
def subset_atomic_verification(G, T, verbose=False):
    assert nx.is_directed(G)
    for u,v in T:
        assert G.has_edge(u,v)

    # Remove v-structures and oriented edges in observational graph
    # We know that G_unoriented will not have v-structures
    G_unoriented = nx.DiGraph()
    dag = cd.DAG.from_nx(G)
    cpdag = dag.interventional_cpdag([], cpdag=dag.cpdag())

    unoriented = set()
    for e in G.edges:
        if e not in cpdag.arcs:
            unoriented.add(e)
    G_unoriented.add_edges_from(unoriented)

    # Handle each connected component separately
    atomic_intervention_set = []
    compute_R_runtime = 0
    for V in nx.connected_components(G_unoriented.to_undirected()):
        # Extract subgraph and relevant target edges
        # Remember to map indices of subgraph into 0..|V|-1
        map_indices = dict()
        unmap_indices = dict()
        for v in V:
            map_indices[v] = len(map_indices)
            unmap_indices[map_indices[v]] = v
        cc = nx.DiGraph()
        cc.add_edges_from([(map_indices[u], map_indices[v]) for u,v in G_unoriented.subgraph(V).edges])
        T_cc = [(map_indices[u], map_indices[v]) for u,v in T if u in V and v in V]
        if len(T_cc) > 0:
            cc_atomic_interventions, cc_compute_R_runtime = subset_atomic_verification_for_single_component(cc, T_cc, verbose)
            atomic_intervention_set += [unmap_indices[v] for v in cc_atomic_interventions]
            compute_R_runtime += cc_compute_R_runtime
    return atomic_intervention_set, compute_R_runtime

def subset_verification(G, T, k, verbose=False):
    if k == 1:
        start = timer()
        atomic_intervention_vertices, compute_R_runtime = subset_atomic_verification(G, T, verbose)
        end = timer()
        subset_verification_time = end - start
        fraction_of_time_computing_R = compute_R_runtime / subset_verification_time
        interventions = [[v] for v in atomic_intervention_vertices]
        if verbose:
            print("interventions = {0}".format(interventions))
        return interventions, fraction_of_time_computing_R
    else:
        assert False

"""
==============================
Validating the implementation of subset verification on simple examples which we know the exact answer
==============================
"""
'''
Return whether performing interventions on DAG fully orients all edges in T
'''
def verify_correctness(interventions, DAG, T=None):
    dag = cd.DAG.from_nx(DAG)
    cpdag = dag.interventional_cpdag([set(intervention) for intervention in interventions], cpdag=dag.cpdag())
    if T is not None:
        return len(cpdag.arcs & set([(u,v) for u,v in T])) == len(T)
    else:
        return len(cpdag.arcs) == DAG.number_of_edges()

if __name__ == "__main__":
    # Adjacency list of graph example
    example_adj_list = [(0,1), (0,2), (0,3), (0,4), (0,6), (0,7), (0,9), (1,4), (1,5), (2,6), (3,7), (3,9), (7,8), (7,9)]    
    example_T = [(0,1), (0,4), (0,7), (2,6), (3,9), (7,8)]
    example_G = nx.DiGraph(example_adj_list)
    example_interventions = subset_verification(example_G, example_T, k=1, verbose=True)
    verify_correctness(example_interventions, example_G, example_T)

    # For trees, any subset of edges should only need 1 intervention (e.g. root is enough, but DP may also choose something else)
    print("Testing on subset of edges on randomly generated trees")
    for _ in tqdm(range(100)):
        n = 100
        tree = nx.random_tree(100)
        directed_tree = nx.bfs_tree(tree, 0)
        random_T = random.sample(list(directed_tree.edges), random.randint(1, len(directed_tree.edges)))
        tree_interventions = subset_verification(directed_tree, random_T, k=1)
        verify_correctness(tree_interventions, directed_tree, random_T)
        assert len(tree_interventions) == 1

    # For any graph, if T is a subset of the covered edges, then intervention should be minimum vertex cover of T
    # Also, the size of the subset verifying set if T = E should be the same as the minimum verification number
    print("For random G(n,p) graphs, test on subset of covered edges and on entire edgeset. Takes a few seconds per instance.")
    for _ in tqdm(range(100)):
        while True:
            n = 100
            p = random.random()
            gnp = nx.gnp_random_graph(n, p)
            arcs = [(u,v) if u < v else (v,u) for (u,v) in gnp.edges]
            G = nx.DiGraph()
            G.add_edges_from(arcs)
            covered_edges = compute_covered_edges(G)

            dag = cd.DAG.from_nx(G)
            cpdag = dag.cpdag()
            unoriented = set()
            for e in arcs:
                if e not in cpdag.arcs:
                    unoriented.add(e)

            assert len(covered_edges & unoriented) == len(covered_edges)
            if len(unoriented) > 0:
                # Check that gnp_interventions is a minimum vertex cover of random_T
                random_T = random.sample(list(covered_edges), random.randint(1, len(covered_edges)))
                gnp_interventions = subset_verification(G, random_T, k=1)
                verify_correctness(gnp_interventions, G, random_T)
                for u,v in random_T:
                    assert [u] in gnp_interventions or [v] in gnp_interventions
                H = nx.Graph()
                H.add_edges_from(random_T)
                mvc = compute_minimum_vertex_cover(H)
                assert len(mvc) == len(gnp_interventions)

                # Check that subset verification on all edges and verification returns same sized verifying sets
                subset_interventions = subset_verification(G, list(G.edges), k=1)
                verify_correctness(subset_interventions, G)
                full_interventions = verification(G, k=1)
                verify_correctness(full_interventions, G)
                assert len(subset_interventions) == len(full_interventions)
                break

