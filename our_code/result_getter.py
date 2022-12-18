import itertools as itr
from dag_loader import DagLoader
from alg_runner import AlgRunner
import pandas as pd
import math

from p_tqdm import p_map

from verify import *
from subset_verify import *

class ResultGetter:
    def __init__(self, algs, nnodes_list, sampler, other_params_list, ngraphs=100, comparable_edges=True):
        self.algs = algs
        self.nnodes_list = nnodes_list
        self.other_params_list = other_params_list
        self.sampler = sampler
        self.ngraphs = ngraphs
        self.dag_loaders = [
            DagLoader(nnodes, self.ngraphs, self.sampler, other_params, comparable_edges=comparable_edges)
            for nnodes, other_params in itr.product(self.nnodes_list, self.other_params_list)
        ]

    def get_results(self, overwrite=False):
        results = []

        for alg in self.algs:
            for dl in self.dag_loaders:
                ar = AlgRunner(alg, dl)
                nnodes_list, times_list = ar.get_alg_results(overwrite=overwrite)

                for nnodes, time in zip(nnodes_list, times_list):
                    results.append(dict(
                        alg=alg,
                        nnodes=dl.nnodes,
                        **dl.other_params,
                        interventions=nnodes,
                        time=time
                    ))

        # Use different benchmarks depending on experiments
        print("COMPUTING BENCHMARK VALUES")
        benchmarks = []
        for dl in self.dag_loaders:
            dags, targets = dl.get_dags()
            dags_nx = [dag.to_nx() for dag in dags]
            k = [1] * len(dags)
            subset_verification_list, fraction_of_time_computing_R_list = list(zip(*[(len(x), t) for x, t in p_map(subset_verification, dags_nx, targets, k)]))
            full_verification_list = [len(x) for x in p_map(verification, dags_nx, k)]

            for ftr, sv, fv in zip(fraction_of_time_computing_R_list, subset_verification_list, full_verification_list):
                benchmarks.append(dict(
                    nnodes=dl.nnodes,
                    **dl.other_params,
                    fraction_of_time_computing_R=ftr,
                    subset_verification_number=sv,
                    full_verification_number=fv
                ))

        res_df = pd.DataFrame(results)
        res_df = res_df.set_index(list(set(res_df.columns) - {'interventions', 'time'}))

        benchmark_df = pd.DataFrame(benchmarks)
        benchmark_df = benchmark_df.set_index(list(set(benchmark_df.columns) - {'fraction_of_time_computing_R', 'subset_verification_number', 'full_verification_number'}))
        return res_df, benchmark_df

