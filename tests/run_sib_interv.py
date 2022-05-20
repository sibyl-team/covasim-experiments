
from importlib import reload

import argparse
import pandas as pd
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

import sib
import covasim
import covasim.utils as cvu

import covasibyl

from covasibyl.rankers import sib_rank
from covasibyl import utils
from covasibyl import ranktest as rktest

import base_sim_pars as base

class dummy_logger:
    def info(self,s):
        print(s)


MU_SIR = 0.0714


def get_sib_pars(prob_i, prob_r, p_seed, p_sus=0.5):
    pseed = p_seed / (2 - p_seed)
    psus = p_sus * (1 - pseed)
    pautoinf = 1e-6

    sibPars = sib.Params(
            prob_i=prob_i,#sib.Uniform(beta), #prob_i,
            prob_r=prob_r, #sib.Exponential(0.06), #prob_r,
            pseed=pseed,
            psus=psus,
            pautoinf=pautoinf)
    return sibPars

def compute_probs_i_r(beta,T):

    pars_nov0=(1.21533407, 3.87484739)
    pars_vlow=(1.04475749, 5.97047953)
    values_vload= [1.53846, 0.76923]

    u0 = expit(pars_nov0[0]*(range(T+1) - pars_nov0[1]*np.ones(T+1)))
    u1 = expit(pars_vlow[0]*(range(T+1) - pars_vlow[1]*np.ones(T+1)))

    vact = u0*values_vload[0]+(values_vload[1]-values_vload[0])*u1

    gamma_p=(11.40049704552924, 0.8143055827103249)

    rec_dist = utils.gamma_pdf_full(np.arange(0,T+1), *gamma_p)

    prob_i = sib.PiecewiseLinear(sib.RealParams(
        list(vact*beta)))
    prob_r = sib.PiecewiseLinear(sib.RealParams(
        list(1-rec_dist.cumsum()) ))
    
    return prob_i, prob_r

if __name__ == "__main":

    parser = base.create_parser()
    parser.add_argument("--n_cores", default=5, help="Set the number of cores for sib")

    args = parser.parse_args()
    print("Arguments:")
    print("\t",args)
    N = 20000
    T = 90
    seed = args.seed

    params = base.make_std_pars(N,T, seed=seed)
    popfile = base.get_people_file(seed, N)

    ### ranker parameter
    prob_seed = 1/N
    prob_sus = 0.5
    fp_rate = 0.0
    fn_rate = 0.0

    prob_i, prob_r = compute_probs_i_r(base.BETA, T)

    sibPars = get_sib_pars(prob_i, prob_r, p_seed=prob_seed, p_sus=prob_sus)


    sibRanker = sib_rank.SibRanker(
        params=sibPars,
        maxit0=15,
        maxit1=20,
        tol=1e-3,
        memory_decay=1e-5,
        window_length=21,
        tau=0,
        fnr=fn_rate,
        fpr=fp_rate

    )
    sibRkTest = rktest.RankTester(sibRanker, "sib ranker",
                                num_tests_algo=200,
                                num_tests_rand=100,
                                symp_test=80.,
                                start_day=10,
                                logger=dummy_logger(),
                                )
    sib.set_num_threads(args.n_cores)

    ct = covasim.contact_tracing(trace_probs=.4, trace_time=1, start_day=10)

    sim = covasim.Sim(pars=params, interventions=[sibRkTest, ct],
        popfile=popfile,
        label="sib rk interv",
    )

    sim.run()

    ###

    tt = sim.make_transtree()