import argparse
from pathlib import Path


import numpy as np
import pandas as pd
import sciris as sc
import covasim 
from covasibyl import ranktest

class dummy_logger:
    def info(self,s):
        print(s)

BETA= 0.0149375 ## from best fit of the parameters to the King County area -> Kerr et al
N_PEOPLE = 20000

def make_std_pars(N=N_PEOPLE, T=100,seed=1, dynamic_layers=["w","c"]):
    pars_sim={  'pop_size'      : N,
                'pop_scale'     : 1,
                'pop_type'      : 'synthpops',
                'pop_infected'  : int(300*N/225e3),
                'beta'          : BETA,
                "n_days" : T,
                'rescale'       : False,
                'rand_seed'     : seed,
                #'analyzers'     : covasim.daily_stats(days=[]),
                'beta_layer'    : dict(h=3.0, s=0.6, w=0.6, c=0.3, l=1.5),
                "use_waning": False,
                }

    pars_sim_test = dict(pars_sim)
    #pars_sim_test["n_days"] = 60
    pars_sim_test["dynam_layer"] = {k:1 for k in dynamic_layers}

    return pars_sim_test

def get_people_file(seed, n):
    p = Path("../runs/pops")
    f = p / f"kc_rnr_{int(n/1000)}k_seed{int(seed % 50)}.ppl"
    return f

def create_parser():
    parser = argparse.ArgumentParser(description='Run simulation with covasim and rankers')

    parser.add_argument("-s","--seed", type=int, default=1, dest="seed")
    parser.add_argument("-N", type=int, default=int(20e3), dest="N", help="Number of agents")
    parser.add_argument("-T",type=int, default=90, dest="T", help="Number of days to run" )
    parser.add_argument("--prefix", type=str, default="", help="Out file prefix")
    parser.add_argument("--nt_algo", type=int, default=200,
        help="Number of test per day for the ranking algorithm")
    parser.add_argument("--nt_rand", type=int, default=100, 
        help="Number of random tests per day to find symptomatics")

    parser.add_argument("--day_start", default=10, dest="start_day",
        help="day to start the intervention")

    return parser

def check_save_folder(fold, create=True):
    p = Path(fold)
    if not p.exists():
        print(f"PROBLEM: folder of {fold} does NOT exist ")
        if create:
            p.mkdir(parents=True)
    return p
    
def build_run_sim(ranker_fn, rk_name, args):
    ## construct the simulation and run it
    N = args.N
    T = args.T
    seed = args.seed
    params = make_std_pars(N,T, seed=seed)
    popfile = get_people_file(seed, N)

    ranker = ranker_fn()

    sibRkTest = ranktest.RankTester(ranker, f"{rk_name} ranker",
                                num_tests_algo=args.nt_algo,
                                num_tests_rand=args.nt_rand,
                                symp_test=80.,
                                start_day=args.start_day,
                                logger=dummy_logger(),
                                )

    ct = covasim.contact_tracing(trace_probs=.4, trace_time=1, start_day=10)

    sim = covasim.Sim(pars=params, interventions=[sibRkTest, ct],
        popfile=popfile,
        label=f"{rk_name} ranking interv",
    )

    sim.run()

    return sim

def save_sim_results(sim, args, rk_name, out_fold):
    seed = args.seed
    tt = sim.make_transtree()
    N = sim.pars["pop_size"]
    T= args.T
    assert T == sim.pars["n_days"]

    testranker = sim["interventions"][0]
    assert type(testranker) == ranktest.RankTester

    rank_stats = pd.DataFrame(testranker.hist).to_records(index=False)

    test_stats = np.concatenate(testranker.tester.tests_stats)

    pars_sim = dict(sim.pars)
    del pars_sim["interventions"]
    del pars_sim["analyzers"]

    save_dict = dict(tt=tt, rank_stats=rank_stats, 
         test_stats=test_stats,
         sim_res=sim.results,
         sim_pars=pars_sim) 

   
    out_fold = check_save_folder(out_fold)
    savefile_name = args.prefix +f"epi_kc_{int(N/1000)}k_T_{T}_s_{seed}_rk_{rk_name}"
    print("Saving results to: ", out_fold, savefile_name)
    sc.saveobj(out_fold / f"{savefile_name}_res.pkl", save_dict)

    args_d = vars(args)

    sc.savejson(out_fold / f"{savefile_name}_args.json", args_d)
