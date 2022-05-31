import argparse
from pathlib import Path
from turtle import update


import numpy as np
import pandas as pd
import sciris as sc
import covasim 
from covasibyl import ranktest, ranktestnew
from covasibyl import analyzers as analysis

import subprocess

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
         cwd=Path(__file__).resolve().parent
         ).decode('ascii').strip()

git_version = get_git_revision_hash

class dummy_logger:
    def info(self,s):
        print(s)

BETA= 0.0149375 ## from best fit of the parameters to the King County area -> Kerr et al
N_PEOPLE = 20000

LAYERS_KEYS=["h","s","c","w","l"]
PARS_no_quar_red = {
    "quar_factor":dict(zip(["h","s","c","w","l"],[1.]*5))
}
FULL_ISO_PARS={
    "iso_factor": {k: 0. for k in LAYERS_KEYS}
}

def make_std_pars(N=N_PEOPLE, T=100,seed=1, dynamic_layers=["w","c"], full_iso=False):
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

    pars_sim_test.update(PARS_no_quar_red)
    
    if full_iso:
        pars_sim_test.update(FULL_ISO_PARS)
    else:
        pars_sim_test["iso_factor"] = {k: 0.1 for k in LAYERS_KEYS}

    return pars_sim_test

def get_people_file(seed, n):
    pare = Path(__file__).parent
    print(pare)
    p = pare/Path("../runs/pops")
    f = p / f"kc_rnr_{int(n/1000)}k_seed{int(seed % 50)}.ppl"
    return f

def create_parser():
    parser = argparse.ArgumentParser(description='Run simulation with covasim and rankers')

    parser.add_argument("-s","--seed", type=int, default=1, dest="seed")
    parser.add_argument("-N", type=int, default=int(20e3), dest="N", help="Number of agents")
    parser.add_argument("-T",type=int, default=50, dest="T", help="Number of days to run" )
    parser.add_argument("--prefix", type=str, default="", help="Out file prefix")
    parser.add_argument("--nt_algo", type=int, default=200,
        help="Number of test per day for the ranking algorithm")
    parser.add_argument("--nt_rand", type=int, default=100, 
        help="Number of random tests per day to find symptomatics")

    parser.add_argument("--day_start", default=10, type=int, dest="start_day",
        help="day to start the intervention")
    parser.add_argument("--save_every", default=5, type=int, dest="n_days_save",
        help="Number of days to wait before saving results periodically")

    parser.add_argument("--fold_save", type=str, default="", help="Saving folder")

    parser.add_argument("--full_iso", action="store_true", help="Test with full isolation")

    return parser

def check_save_folder(fold, create=True):
    p = Path(fold)
    if not p.exists():
        print(f"PROBLEM: folder of {fold} does NOT exist ")
        if create:
            p.mkdir(parents=True)
    return p

def save_data(sim, period,  args, rk_name, out_fold, start_day):
    if sim.t > 1 and  (sim.t % period) == 0:
        save_sim_results(sim, args, rk_name, out_fold)

def make_save_data(period,  args, rk_name, out_fold):
    return lambda sim: save_data(sim, period,  args, rk_name, out_fold)

def make_interv(ranker, rk_name, args, **kwargs):
    rktest_int = ranktest.RankTester(ranker, f"{rk_name} ranker",
                                num_tests_algo=args.nt_algo,
                                num_tests_rand=args.nt_rand,
                                symp_test=80.,
                                start_day=args.start_day,
                                logger=dummy_logger(),
                                **kwargs
                                )
    return rktest_int

def make_interv_new(ranker, rk_name, args, **kwargs):
    rktest_int = ranktestnew.RankTester(ranker, f"{rk_name} ranker",
                                num_tests=args.nt_algo+args.nt_rand,
                                start_day=args.start_day,
                                logger=dummy_logger(),
                                symp_test_p=0.5,
                                **kwargs
                                )
    return rktest_int

def build_run_sim(rktest_int, rk_name, args, out_fold, run=True):
    ## construct the simulation and run it
    N = args.N
    T = args.T
    seed = args.seed
    args.git_version = {"runfiles": get_git_revision_hash()}
    params = make_std_pars(N,T, seed=seed, full_iso=args.full_iso)
    popfile = get_people_file(seed, N)
    period_save = args.n_days_save

    analyz = [analysis.store_seir(),
    lambda sim: save_data(sim, period_save,  args, rk_name, out_fold, args.start_day)]
    ct = covasim.contact_tracing(trace_probs=.4, trace_time=1, start_day=10)

    sim = covasim.Sim(pars=params, interventions=[rktest_int, ct],
        popfile=popfile,
        label=f"{rk_name} ranking interv",
        analyzers=analyz,
    )

    if run:
        sim.run()

    return sim

def save_sim_results(sim, args, rk_name, out_fold):
    seed = args.seed
    N = sim.pars["pop_size"]
    T= args.T
    if len(args.fold_save)>0:
        out_fold = args.fold_save
    
    assert T == sim.pars["n_days"]

    testranker = sim["interventions"][0]
    #assert type(testranker) == ranktest.RankTester
    
    counter = sim["analyzers"][0]
    assert isinstance(counter, analysis.store_seir)

    counts_arr = counter.out_save()

    #print(pd.DataFrame(testranker.hist[:15]) )
    print(testranker.hist[-1])
    rank_stats = pd.DataFrame(testranker.hist).to_records(index=False)
    
    ts = testranker.tester.tests_stats
    if len(ts) > 0:
        test_stats = np.concatenate(ts)
    else:
        test_stats = np.array([])

    pars_sim = dict(sim.pars)
    del pars_sim["interventions"]
    del pars_sim["analyzers"]

    def pars_log(x):
        if x["source"] is None:
            x["source"] = -1
        return x
    inf_log = pd.DataFrame(map(pars_log, 
            sim.people.infection_log)).to_records(index=False)
    rdata = dict(testranker.ranker_data)

    arrs_save = dict()
    
    if "logger" in rdata: del rdata["logger"]
    print("ranker_data: ", rdata.keys())
    for k in rdata.keys():
        if "margs_" in k:
            arrs_save[k] = rdata[k]
    for k in arrs_save:
        rdata.pop(k)

    if len(rdata.keys())==0:
        ranker_data = np.empty(0)
    else:
        ranker_data = pd.DataFrame(rdata).to_records(index=False)
    
    arrs_save["sim_counts"] = counts_arr
    arrs_save.update(dict(rank_stats=rank_stats, test_stats=test_stats,
            ranker_data=ranker_data,
            infect_log=inf_log))

    if sim.results_ready:
        tt = sim.make_transtree()
    else:
        tt = []
    
    save_dict = dict(tt=tt,  
         sim_res=sim.results,
         sim_pars=pars_sim,
         )
    
    save_dict.update(arrs_save)

   
    out_fold = check_save_folder(out_fold)
    savefile_name = args.prefix +f"epi_kc_{int(N/1000)}k_T_{T}_s_{seed}_rk_{rk_name}"
    print("Saving results to: ", out_fold, savefile_name)
    sc.saveobj(out_fold / f"{savefile_name}_res.pkl", save_dict)
    np.savez_compressed(out_fold / f"{savefile_name}_stats.npz", arrs_save)

    args_d = vars(args)

    sc.savejson(out_fold / f"{savefile_name}_args.json", args_d)
