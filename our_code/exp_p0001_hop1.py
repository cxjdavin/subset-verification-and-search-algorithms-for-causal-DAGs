from dag_loader import DagSampler
from plot_results_vary_nnodes import plot_results_vary_nodes

algs = [
    'random',
    'dct',
    'coloring',
    'separator_k1',
    'node_induced_separator_k1'
]
nnodes_list = [100, 150, 200, 250, 300]
plot_results_vary_nodes(
    nnodes_list,
    100,
    DagSampler.GNP_TREE,
    dict(p=0.001, hops=1, figname="p0001_hop1"),
    algorithms=algs,
    overwrite=True
)

