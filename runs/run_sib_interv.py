
from pathlib import Path

import argparse
import warnings
import pandas as pd
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

import sib
import covasim.utils as cvu

import sciris as sc

from covasibyl.rankers import sib_rank
from covasibyl import utils

import base_sim_run as base

class dummy_logger:
    def info(self,s):
        print(s)


MU_SIR = 0.0714
OUT_FOLD = "results"

BETA_MEDIAN = base.BETA*base.REL_T_MED

def get_prec_exact(T:int):
    vals_json = """[2.325553437072944e-30, 1.9771667718346412e-22, 3.98436387721634e-15, 6.376081651963155e-10, 2.3360876347739566e-07, 1.845099815866748e-05, 0.00039696442684132994, 0.0033356695257003425, 0.014908984392460745, 0.04174110327051436, 
    0.08178324587620472, 0.12135948842102566, 0.14458445410883655, 0.1445588125118671, 0.12554378617011983, 0.09733313157321885, 0.06889650017289549, 0.045405433020872195, 0.028399358513704454, 0.017237203127853987, 0.010464029966138187,
     0.006623690334476752, 0.004581208656192367, 0.003563947861964797, 0.0030907123639256637, 0.002881031276784798, 0.0027805606105493915, 0.0027095649281498597, 0.002630566467036701, 0.002529149057968519, 0.002402942578075652, 0.0022554224076080597,
      0.0020924667961153298, 0.0019204823949358245, 0.0017454265119129068, 0.0015723471074820215, 0.0014052224437768462, 0.0012469710655352638, 0.001099553190825605, 0.0009641145785930401, 0.0008411427141654509, 0.0007306173770154723, 0.0006321457583792639, 0.0005450776388216133, 
      0.0004685995567386912, 0.0004018089683064163, 0.00034377052996925465, 0.0002935571319476555, 0.00025027840114674315, 0.00021309923738753746, 0.00018125066212167226, 0.00015403492017559563, 0.00013082643106461874, 0.0001110698654099229, 9.427633854118646e-05, 
      8.001847294198311e-05, 6.792488363657283e-05, 5.767448257435156e-05, 4.8990874638114644e-05, 4.163702367845997e-05, 3.541029668412201e-05, 3.0137943038222435e-05, 2.5673029617755168e-05, 2.1890827774716904e-05, 1.868563214025311e-05, 1.5967981442395654e-05, 
      1.3662246342667665e-05, 1.170454730928417e-05, 1.0040965722359887e-05, 8.626012983730681e-06, 7.421324823893908e-06, 6.394550864746977e-06, 5.518412534240906e-06, 4.7699054533241324e-06, 4.129625308490403e-06, 3.581198914443948e-06, 3.1108046241104787e-06, 
      2.7067684438391886e-06, 2.3592241618995516e-06, 2.059827509741633e-06, 1.8015158651442932e-06, 1.578306294419164e-06, 1.385125838373289e-06, 1.2176688946121954e-06, 1.0722773567978325e-06, 9.458398579905874e-07, 8.357070468241995e-07, 7.396203168272676e-07, 
      6.556518238090003e-07, 5.821539753147179e-07, 5.177168696573201e-07, 4.611324085011709e-07, 4.113640137487954e-07, 3.675210528116718e-07, 3.288372215496826e-07, 2.946522557498632e-07, 2.643964437932976e-07, 2.3757749833688106e-07, 2.137694161128089e-07, 
      1.9260301459388513e-07, 1.737578841935922e-07, 1.56955536458436e-07, 1.4195356369632967e-07, 1.2854065478772616e-07, 1.1653233648316227e-07, 1.0576733007783468e-07, 9.610443062032149e-08, 8.74198303046656e-08, 7.960481986520266e-08, 7.25638120201797e-08]"""
    import json
    rec_dist=np.array(json.loads(vals_json))[:T+1]
    return sib.PiecewiseLinear(sib.RealParams(
        list(1-rec_dist.cumsum()) ))

def get_vload_real(T:int):
    vals_json = """
    [3.019362537770576e-11, 0.0009782360948130835, 0.07633861827893693, 0.4156928589223723, 0.8594888462305331, 1.1366568819030092, 1.173214488765383, 1.0683914075752121, 0.9441065739274009, 0.8564495046788241, 
    0.8085247246272426, 0.7859035727331752, 0.7760843741470433, 0.7720107983382031, 0.7703569964105829, 0.7696902603401734, 0.7694208901403633, 0.7693112051903307, 0.7692660316539217, 0.7692471736931447, 
    0.7692391841951942, 0.7692357467193761, 0.7692342443920941, 0.7692335774481694, 0.7692332767480884, 0.769233139100848, 0.7692330751523832, 0.7692330450125202, 0.7692330306076164, 0.7692330236294361, 
    0.7692330202045904, 0.7692330185023794, 0.7692330176459926, 0.7692330172100527, 0.7692330169856099, 0.7692330168687848, 0.7692330168073306, 0.769233016774672, 0.7692330167571447, 0.769233016747648,
     0.7692330167424548, 0.7692330167395898, 0.7692330167379954, 0.7692330167371009,
     0.7692330167365948, 0.7692330167363062, 0.7692330167361403, 0.7692330167360445, 0.7692330167359887, 0.7692330167359559]
    """
    import json
    vload = json.loads(vals_json)
    assert T >= len(vload)
    x = np.zeros(T)
    x[:len(vload)] = vload
    x[len(vload):] = x[len(vload)-1]
    return x

def get_sib_pars(prob_i, prob_r, p_seed, p_sus=0.5, p_autoinf=1e-6):
    pseed = p_seed / (2 - p_seed)
    psus = p_sus * (1 - pseed)

    sibPars = sib.Params(
            prob_i=prob_i,#sib.Uniform(beta), #prob_i,
            prob_r=prob_r, #sib.Exponential(0.06), #prob_r,
            pseed=pseed,
            psus=psus,
            pautoinf=p_autoinf)
    return sibPars

def get_sib_markov_p(beta,p_seed, p_sus, p_autoinf=1e-10):
    pseed = p_seed / (2 - p_seed)
    psus = p_sus * (1 - pseed)
    mu_rate = -np.log(1-MU_SIR)

    sibPars = sib.Params(
            prob_i=sib.Uniform(beta), #prob_i,
            prob_r=sib.Exponential(mu_rate), #sib.Exponential(0.06), #prob_r,
            pseed=pseed,
            psus=psus,
            pautoinf=p_autoinf)
    return sibPars

def compute_probs_i_r(beta,T, tcut_inf):

    #pars_nov0=(1.21533407, 3.87484739)
    #pars_vlow=(1.04475749, 5.97047953)
    pars_nov0 =(1.21533406, 3.87484739)
    pars_vlow=(1.04399808, 6.97266543)
    values_vload= [1.53846, 0.76923]

    u0 = expit(pars_nov0[0]*(range(T+1) - pars_nov0[1]*np.ones(T+1)))
    u1 = expit(pars_vlow[0]*(range(T+1) - pars_vlow[1]*np.ones(T+1)))

    cutoff = -expit(1.6*(range(T+1) - tcut_inf*np.ones(T+1)))

    #vact = get_vload_real(T+1) + cutoff*values_vload[1]
    vact = u0*values_vload[0]+(values_vload[1]-values_vload[0])*u1 + cutoff*values_vload[1]

    gamma_p=(11.40049704552924, 0.8143055827103249)

    rec_dist = utils.gamma_pdf_full(np.arange(0,T+1), *gamma_p)

    prob_i = sib.PiecewiseLinear(sib.RealParams(
        list(vact*beta)))
    prob_r = sib.PiecewiseLinear(sib.RealParams(
        list(1-rec_dist.cumsum()) ))
    
    return prob_i, prob_r


if __name__ == "__main__":

    parser = base.create_parser()
    parser.add_argument("--n_cores", default=5, type=int, help="Set the number of cores for sib")
    parser.add_argument("--sib_tol", default=1e-3, type=float, help="Tolerance for sib convergence")
    parser.add_argument("--markov", action="store_true", help="Use Markov SIR params")
    parser.add_argument("--n_src_sib", default=1, type=int, help="Number of seeds in the epidemic cascade")
    parser.add_argument("--p_autoinf", default=1e-6, type=float, help="Prob of autoinfection for sib")
    parser.add_argument("--win_length", default=21, type=int, help="Length of BP window")
    parser.add_argument("--debug_c", action="store_true", help="Debug convergence time")
    parser.add_argument("--prec_exact", action="store_true")

    parser.add_argument("--vload_cut", type=int, default=65, help="Day of cutoff of vload")
    parser.add_argument("--no-bmed",action="store_false", dest="bmed", help="Use median value for beta")
    parser.add_argument("--fast_ct", action="store_true", dest="sib_fast_ctadd")
    parser.add_argument("--sib_tau", default=-1,type=int, help="Sib tau value")

    args = parser.parse_args()
    base.check_args(args)
    if args.give_trel:
        print("Giving T rel, not using beta med")
        args.bmed = False
    
    print("Arguments:")
    print("\t",args)
    seed = args.seed

    T=args.T
    N=args.N
    
    if(args.n_sources > 0):
        args.n_src_sib = args.n_sources

    ### ranker parameter
    prob_seed = args.n_src_sib/N
    prob_sus = 0.5
    fp_rate = args.fpr if args.fpr > 0 else 1e-6
    fn_rate = args.fnr if args.fnr > 0 else 1e-6
    if args.bmed:
        warnings.warn("Using median value for Relative transmission")
        args.prefix+="bmed_"
        mbeta = BETA_MEDIAN
    else:
        mbeta = base.BETA

    if args.markov:
        print("Using markov SIR dynamics")
        sibPars = get_sib_markov_p(mbeta, prob_seed, prob_sus)
    else:
        prob_i, prob_r = compute_probs_i_r(mbeta, T, tcut_inf=args.vload_cut)
        if args.prec_exact:
            prob_r = get_prec_exact(T)
        
        sibPars = get_sib_pars(prob_i, prob_r, p_seed=prob_seed,
             p_sus=prob_sus, p_autoinf=args.p_autoinf)

    print("sib pars:", sibPars)
    rk_name = "sib"
    if args.markov:
        rk_name+="_mk"
    #tau_sib = args.sib_tau if args.sib_tau > 0 else None
    tau_sib=None
    if args.sib_tau > 0:
        tau_sib = args.sib_tau
        rk_name+=f"_t{tau_sib}"

    ranker = sib_rank.SibRanker(
        params=sibPars,
        maxit0=20,
        maxit1=20,
        tol=args.sib_tol,
        memory_decay=1e-5,
        window_length=args.win_length,
        tau=tau_sib,
        fnr=fn_rate,
        fpr=fp_rate,
        debug_times=True,
        faster_ctadd=args.sib_fast_ctadd,

    )
    interv = base.make_interv_new(ranker, rk_name, args)


    #interv.mitigate = False
    #interv._check_epi_tests = True
    #args.prefix +="nomit_rndtest_"
    try:
        args.prefix += (interv._comp_flag()+"_")
    except AttributeError:
        args.prefix +="newrk_"
    #interv.mitigate = False
    #interv._check_epi_tests = True
    #args.prefix +="nomit_rndtest_"
    if args.full_iso:
        interv.iso_cts_strength = 0.
    
    sib.set_num_threads(args.n_cores)
    argsv=vars(args)
    argsv["sib_version"]=sib.version()

    sim = base.build_run_sim(interv, rk_name, args, OUT_FOLD, args_analy={"printout":True})

    base.save_sim_results(sim, args, rk_name, OUT_FOLD)
