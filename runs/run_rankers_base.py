
from pathlib import Path

import sys
import pandas as pd
import numpy as np

import covasim
import covasim.utils as cvu

import sciris as sc

from covasibyl.rankers import sib_rank
from covasibyl import utils
from covasibyl import ranktest as rktest

import base_sim_run as base_sim



MU_SIR = 0.0714
OUT_FOLD = "results"
values_vload = [1.53846, 0.76923]


def get_ranker(which, tau, delta, beta, mu):
    if which == "MF":
        from covasibyl.rankers import mean_field_rank
        return mean_field_rank.MeanFieldRanker(tau=tau,delta=delta,mu=mu,lamb=beta)
    elif which=="CT":
        from covasibyl.rankers import tracing_rank
        return tracing_rank.TracingRanker(tau=tau, lamb=beta)
    else:
        raise ValueError("Ranker not recognized")
    
if __name__ == "__main__":

    parser = base_sim.create_parser()
    parser.add_argument("--ranker", dest="ranker", type=str, help="Choose the ranker")

    args = parser.parse_args()
    print("Arguments:")
    print("\t",args)

    rk_name = args.ranker

    ranker_fn = lambda: get_ranker(rk_name, 10, 4, base_sim.BETA, MU_SIR)

    sim = base_sim.build_run_sim(ranker_fn, rk_name, args)

    base_sim.save_sim_results(sim, args, rk_name, OUT_FOLD)


