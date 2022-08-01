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
REL_T_MED=0.41
N_PEOPLE = 20000

LAYERS_KEYS=["h","s","c","w","l"]
PARS_no_quar_red = {
    "quar_factor":dict(zip(["h","s","c","w","l"],[1.]*5))
}
FULL_ISO_PARS={
    "iso_factor": {k: 0. for k in LAYERS_KEYS}
}

def make_std_pars(N=N_PEOPLE, T=100,seed=1, dynamic_layers=["w","c"], full_iso=False, quar_fact=1.):
    N=int(N)
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

    #pars_sim_test.update(PARS_no_quar_red)
    pars_sim_test["quar_factor"] ={
        k: quar_fact for k in ["h","s","c","w","l"]
    }
    
    if full_iso:
        pars_sim_test.update(FULL_ISO_PARS)
    else:
        pars_sim_test["iso_factor"] = {k: 0.1 for k in LAYERS_KEYS}

    return pars_sim_test

def get_people_file(seed, n, verbose=True):
    pare = Path(__file__).parent
    if verbose: print(pare)
    p = pare/Path("../runs/pops")
    f = p / f"kc_rnr_{int(n/1000)}k_seed{int(seed)}.ppl"
    return f

def create_parser():
    parser = argparse.ArgumentParser(description='Run simulation with covasim and rankers')

    parser.add_argument("-s","--seed", type=int, default=1, dest="seed")
    parser.add_argument("-N", type=float, default=20e3, dest="N", help="Number of agents")
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

    parser.add_argument("--fnr", type=float, default=0., help="False negative test rate")
    parser.add_argument("--fpr", type=float, default=0., help="False positive rate of testing")
    parser.add_argument("--test_delay", type=int, default=0, help="Delay in delivering the tests (in days)")
    parser.add_argument("--p_loss", type=float, default=0., help="Probability of losing a test result")

    parser.add_argument("--save_rank", type=int, default=-10, help="Number of the ranking at each time step to save, set to a value >0")

    ##internal contact tracing
    parser.add_argument("--quar_factor", type=float, default=1., help="Effective infectivity reduction in quarantine, 1=> no reduction")
    parser.add_argument("--ct_trace_time", type=int, default=0, help="Time taken to quarantine people with internal contact tracing")

    parser.add_argument("--ct_trace_p", type=float, default=0.5, help="Probability of individual quarantined through contact tracing")

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

interv_args_def=lambda args: dict(
    sensitivity=1-args.fnr,
    specificity=1-args.fpr,
    start_day=args.start_day,
    loss_prob=args.p_loss,
    test_delay=args.test_delay
    )
def make_interv(ranker, rk_name, args, **kwargs):
    pars = interv_args_def(args)
    pars.update(kwargs)
    rktest_int = ranktest.RankTester(ranker, f"{rk_name} ranker",
                                num_tests_algo=args.nt_algo,
                                num_tests_rand=args.nt_rand,
                                symp_test=80.,
                                logger=dummy_logger(),
                                **pars
                                )
    return rktest_int

def make_interv_new(ranker, rk_name, args, **kwargs):
    pars = interv_args_def(args)
    pars.update(kwargs)
    rktest_int = ranktestnew.RankTester(ranker, f"{rk_name} ranker",
                                num_tests=args.nt_algo+args.nt_rand,
                                logger=dummy_logger(),
                                symp_test_p=0.5,
                                quar_factor=args.quar_factor,
                                **pars
                                )
    if args.save_rank > 0:
        rktest_int.set_extra_stats_fn(
            lambda sim,rank,ll: rank.sort_values(ascending=False)[:args.save_rank]
        )
    return rktest_int

def build_run_sim(rktest_int, rk_name, args, out_fold, run=True, args_analy=None):
    ## construct the simulation and run it
    if args_analy==None:
        args_analy = dict()
    args.N = int(args.N)
    N = args.N
    T = args.T
    seed = args.seed
    args.git_version = {"runfiles": get_git_revision_hash()}
    params = make_std_pars(N,T, seed=seed, full_iso=args.full_iso)
    popfile = get_people_file(seed, N)
    period_save = args.n_days_save

    analyz = [analysis.store_seir(**args_analy),
    lambda sim: save_data(sim, period_save,  args, rk_name, out_fold, args.start_day)]
    ct = covasim.contact_tracing(trace_probs=args.ct_trace_p, trace_time=args.ct_trace_time, start_day=args.start_day)

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

    ### ranker data
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
    
    ## Add the sim results to the arrays saved in the stats.npz
    if sim.results_ready:
        z = {k: np.array(v) for k,v in sim.results.items()}
        del z["variant"]
        sim_res = pd.DataFrame(z).to_records(index=False)
        arrs_save["sim_res"] = sim_res
        print("Saving sim results")
    
    if testranker.extra_stats:
        ex_st = testranker.extra_stats
        save_dict["rk_extra_stats"] = testranker.extra_stats

        if isinstance(next(iter(ex_st.values())), pd.Series):
            ### transform to array for npz
            out_arr ={}
            for key,v in ex_st.items():
                out_arr[key] = np.fromiter(zip(v.index,v.values), 
                        dtype=[("idx",v.index.dtype),("val",v.values.dtype)])
        else:
            out_arr = ex_st
        arrs_save["rk_extra_stats"] = out_arr
   
    out_fold = check_save_folder(out_fold)
    fnr_str = ""
    if args.fnr > 0:
        fnr_str+=f"_fnr_{round(args.fnr,3)}"
    if args.fpr > 0:
        fnr_str+=f"_fpr_{round(args.fpr,3)}"
    savefile_name = args.prefix +f"epi_kc_{int(N/1000)}k_T_{T}{fnr_str}_s_{seed}_rk_{rk_name}"
    print("Saving results to: ", out_fold, savefile_name)
    sc.saveobj(out_fold / f"{savefile_name}_res.pkl", save_dict)
    np.savez_compressed(out_fold / f"{savefile_name}_stats.npz", **arrs_save)

    args_d = vars(args)

    sc.savejson(out_fold / f"{savefile_name}_args.json", args_d)
