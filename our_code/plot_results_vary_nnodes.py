from dag_loader import DagSampler
import matplotlib.pyplot as plt
import seaborn as sns
from config import FIGURE_FOLDER, POLICY2COLOR, POLICY2LABEL
import os
import random
import ipdb
from result_getter import ResultGetter
sns.set()

OVERWRITE_ALL = True


def plot_results_vary_nodes(
        nnodes_list: list,
        ngraphs: int,
        sampler: DagSampler,
        other_params: dict,
        algorithms: set,
        overwrite=False
):
    random.seed(98625472)
    os.makedirs('figures', exist_ok=True)

    rg = ResultGetter(
        algorithms,
        nnodes_list,
        sampler,
        other_params_list=[other_params],
        ngraphs=ngraphs,
    )
    res_df, benchmark_df = rg.get_results(overwrite=overwrite)

    average_times = res_df.groupby(level=['alg', 'nnodes'])['time'].mean()
    std_times = res_df.groupby(level=['alg', 'nnodes'])['time'].std()
    mean_interventions = res_df.groupby(level=['alg', 'nnodes'])['interventions'].mean()
    std_interventions = res_df.groupby(level=['alg', 'nnodes'])['interventions'].std()
    mean_subset = benchmark_df.groupby(level=['nnodes'])['subset_verification_number'].mean()
    std_subset = benchmark_df.groupby(level=['nnodes'])['subset_verification_number'].std()
    mean_full = benchmark_df.groupby(level=['nnodes'])['full_verification_number'].mean()
    std_full = benchmark_df.groupby(level=['nnodes'])['full_verification_number'].std()

    algorithms = sorted(algorithms)

    plt.clf()
    for alg in algorithms:
        #plt.plot(nnodes_list, average_times[average_times.index.get_level_values('alg') == alg], color=POLICY2COLOR[alg], label=POLICY2LABEL[alg])
        plt.errorbar(nnodes_list, average_times[average_times.index.get_level_values('alg') == alg], color=POLICY2COLOR[alg], label=POLICY2LABEL[alg], yerr=std_times[std_times.index.get_level_values('alg') == alg], capsize=5)

    plt.clf()
    for alg in algorithms:
        plt.errorbar(nnodes_list, mean_interventions[mean_interventions.index.get_level_values('alg') == alg], color=POLICY2COLOR[alg], label=POLICY2LABEL[alg], yerr=std_interventions[std_interventions.index.get_level_values('alg') == alg], capsize=5)
    plt.errorbar(nnodes_list, mean_subset, yerr=std_subset, label="Subset verification number", capsize=5, linestyle="dashed")
    plt.errorbar(nnodes_list, mean_full, yerr=std_full, label="Full verification number", capsize=5, linestyle="dashed")

    plt.xlabel('Number of Nodes')
    plt.ylabel('Average number of interventions used')
    plt.legend()
    plt.xticks(nnodes_list)
    other_params_str = ','.join((f"{k}={v}" for k, v in other_params.items()))
    plt.savefig(os.path.join(FIGURE_FOLDER, '{0}_interventioncount.png'.format(other_params['figname'])))

    # Plot fraction of time computing R
    mean_ftr = benchmark_df.groupby(level=['nnodes'])['fraction_of_time_computing_R'].mean()
    std_ftr = benchmark_df.groupby(level=['nnodes'])['fraction_of_time_computing_R'].std()
    plt.clf()
    plt.errorbar(nnodes_list, mean_ftr, yerr=std_ftr, label="Time spent computing R when computing subset verification number", capsize=5)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Fraction of time')
    plt.legend()
    plt.xticks(nnodes_list)
    plt.savefig(os.path.join(FIGURE_FOLDER, '{0}_ftr.png'.format(other_params['figname'])))

