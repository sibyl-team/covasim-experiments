from pathlib import Path
import covasim 
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