import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from epidemic import get_state
from forw_back import prep_ranking

def filter_idcs_infected(inf_log, idcs):
    """
    Filter indices selecting only those resulting as infected in the log
    """
    sups_infected = set(inf_log["target"]).intersection(idcs)
    return  list(sups_infected)

def adjust_rank_day(rks, t_lim):
    rk = rks[t_lim]
    return pd.Series(data=rk["val"], index=rk["idx"])
def set_rank_day_np(rks, t_lim):
    rk = rks[t_lim]
    return pd.Series(data=rk)

def filter_onlyI(idcs, state):
    st=state[idcs]
    return idcs[(st<7)&(st>0)]


def correct_idcs_find(res_sel, idcs_find, fin_T:int, N:int, filter_infect=True):
    infect_log = pd.DataFrame(res_sel["infect_log"])

    test_stats=pd.DataFrame(res_sel["test_stats"])

    infect_log = infect_log[infect_log["date"]<=fin_T]
    test_stats = test_stats[test_stats["date_res"]<fin_T]

    #ranks = res_sel["rk_extra_stats"].item()

    obs_inf = test_stats[test_stats["res_state"]==1]

    if filter_infect:
        idcs_find = filter_idcs_infected(infect_log, idcs_find)

    idcs_unobs = np.setdiff1d(idcs_find, obs_inf["i"])

    #ss_idcs=filter_idcs_inf(infect_log, idcs_find)

    state = get_state(N,res_sel["people_dates"],fin_T)

    isS=state==0
    isR=(state>=7)
    idcs_SR= np.where(isS | isR)[0]

    unobs_inf=filter_onlyI(idcs_unobs, state)

    return unobs_inf, idcs_SR, state

def calc_auc_idcs(dat, idcs_find, fin_T:int, N:int, ranks=None, filter_infect=True):

    unobs_inf, idcs_SR, state = correct_idcs_find(dat, idcs_find, fin_T, N, filter_infect=filter_infect)

    if ranks is None:
        ranks = dat["rk_extra_stats"].item()
    try:
        rank_day = adjust_rank_day(ranks, fin_T)
    except IndexError:
        rank_day = set_rank_day_np(ranks, fin_T)

    idcs_all=np.union1d(unobs_inf, idcs_SR)

    auc=roc_auc_score(*prep_ranking(unobs_inf,idcs_all,rank_day,N))
    return auc, len(unobs_inf)

def calc_auc_idcs_sims(dat, idcs_find, fin_T:int, N:int,):

    unobs_inf, idcs_SR, state = correct_idcs_find(dat, idcs_find, fin_T, N, filter_infect=True)

    ranks = dat["rk_extra_stats"].item()

    rank_day = adjust_rank_day(ranks, fin_T)

    idcs_all=np.union1d(unobs_inf, idcs_SR)

    auc=roc_auc_score(*prep_ranking(unobs_inf,idcs_all,rank_day,N))
    return auc, len(unobs_inf)

