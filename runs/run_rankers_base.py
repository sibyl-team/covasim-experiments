
from pathlib import Path

import sys
import pandas as pd
import numpy as np


from covasibyl import base, utils
from covasibyl import ranktest as rktest

import base_sim_run as base_sim



MU_SIR = 0.0714
OUT_FOLD = "results"
values_vload = [1.53846, 0.76923]


def get_ranker(which, tau, delta, beta, mu):
    
    def_t = 10
    if which == "MF":
        from covasibyl.rankers import mean_field_rank
        return mean_field_rank.MeanFieldRanker(tau=tau if tau > def_t else def_t,
                delta=delta,mu=mu,lamb=beta)
    elif which=="CT":
        from covasibyl.rankers import tracing_rank
        return tracing_rank.TracingRanker(tau=tau if tau > def_t else def_t, lamb=beta)
    elif which=="greedy":
        from covasibyl.rankers import greedy_rank
        return greedy_rank.GreedyRanker(lamb=beta, tau=tau if tau > 5 else greedy_rank.TAU_INF)
    elif which=="greedy2n":
        from covasibyl.rankers import greedy_rank
        return greedy_rank.GreedyRanker(lamb=beta, tau=tau if tau > 5 else greedy_rank.TAU_INF, sec_neigh=True )
    else:
        raise ValueError("Ranker not recognized")
    
if __name__ == "__main__":

    parser = base_sim.create_parser()
    parser.add_argument("--ranker", dest="ranker", type=str, help="Choose the ranker")
    parser.add_argument("--tau", type=int, default=-1, help="Number of days to use before current")
    parser.add_argument("--delta",type=int, default=3, help="Delta for the MF ranker")

    args = parser.parse_args()
    print("Arguments:")
    print("\t",args)

    rk_name = args.ranker

    ranker = get_ranker(rk_name, tau=args.tau, delta=3, 
            beta=base_sim.BETA, mu=MU_SIR)

    interv = base_sim.make_interv_new(ranker, rk_name, args)

    if args.full_iso:
        interv.iso_cts_strength = 0.
        args.prefix+="fulliso_"
    #interv.mitigate = False
    #interv._check_epi_tests = True
    #args.prefix +="nomit_rndtest_"
    args.prefix+="newrk_"

    

    sim = base_sim.build_run_sim(interv, rk_name, args, OUT_FOLD)

    base_sim.save_sim_results(sim, args, rk_name, OUT_FOLD)


