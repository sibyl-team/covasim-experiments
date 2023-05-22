from collections import defaultdict
import warnings
import numpy as np
from pandas import Series, DataFrame
from sklearn.metrics import roc_auc_score

#from epidemic import get_state

def find_back_rk(inf_log, idc_obsi, idc_ranking, state):
    ### backward
    df_back=inf_log[np.isin(inf_log["target"],idc_obsi)]
    sources_back = df_back[df_back["source"]>=0]["source"]
    ## remove idc already observed
    sources_back = sources_back[~np.isin(sources_back, idc_obsi)]
    # check those in the ranking
    found_backw = sources_back[np.isin(sources_back, idc_ranking)]
    isR = (state[found_backw]>=7)

    backw_I = found_backw[~isR]
    backw_R = found_backw[isR]
    return backw_I, backw_R

def find_forw_rk(inf_log, idc_obsi, idc_ranking, state):

    ff_idc = inf_log[inf_log.source.isin(idc_obsi)].target
    #ff_idc=inf_log[np.isin(inf_log["source"],idc_obsi)]["target"]
    # not in observed
    ff_idc = ff_idc[~ff_idc.isin(idc_obsi)]
    # is in ranking
    inf_for = ff_idc[ff_idc.isin(idc_ranking)]
    is_R = state[inf_for]>=7
    forw_I = inf_for[~is_R]
    forw_R = inf_for[is_R]
    return forw_I, forw_R

def _count_sideward_contacts(inf_log, contacts, idc_obs, idc_I, debug=False):
    ctc_sideward = Series(data=[0]*len(idc_I), index=idc_I, dtype=np.int)
    for i in idc_I:
        src = inf_log[inf_log.target == i].source.iloc[0]
        d = [src]
        while (src >=0):
            src = int(inf_log[inf_log.target == src].source.iloc[0])
            d.append(src)
        if debug: print(i,d)
        ctc_count = 0
        if len(d)>2:
            for k in range(1, len(d)-1):
                src = d[k]
                c=np.isin(contacts["i"], src) & np.isin(contacts["j"],i)
                cts = contacts[c]
                if debug:
                    print(f"\tSrc {src} obs {src in idc_obs}",f"Contacts {src}->{i}: {len(cts)>0}",)
                if src in idc_obs and len(cts)>0:
                    if debug: print("Found!")
                    ctc_count = k
                    break
                #print(cts[:20])
        ctc_sideward[i] = ctc_count
    if debug: print(ctc_sideward)
    return ctc_sideward


def find_forw_back_side(test_res, inf_log_f, t_a, ranks_day, state, contacts, nranks):
    obs_i = test_res[(test_res["res_state"]==1)&(test_res["date_res"]<t_a)]
    idc_obsi=np.unique(obs_i["i"])

    # filter rank
    r=ranks_day[~np.isin(ranks_day["idx"], idc_obsi)]
    ii= r["val"].argsort()[::-1][:nranks]
    ranks_an = r[ii]
    idc_ranking = ranks_an["idx"]

    inf_log = inf_log_f[inf_log_f["date"]<=t_a]
    ### backward
    backw_I, backw_R = find_back_rk(inf_log, idc_obsi, idc_ranking, state)
    ### forward
    forw_I, forw_R = find_forw_rk(inf_log, idc_obsi, idc_ranking, state)
    ## sideward
    # remove found
    idc_tbd = np.setdiff1d(idc_ranking, np.unique(np.concatenate((backw_I, backw_R, forw_I, forw_R))))
    contacts = contacts[contacts["day"]<=t_a]
    idc_EI = idc_tbd[(state[idc_tbd]>0) & (state[idc_tbd]<7)]

    mask=np.isin(contacts["i"], idc_EI) | np.isin(contacts["j"],idc_EI)
    conts_filt = contacts[mask]

    count_side = _count_sideward_contacts(inf_log, conts_filt, idc_obsi, idc_EI,debug=False)

    return (backw_I,backw_R),  (forw_I,forw_R), (count_side[count_side>0].index,)

def make_for_back_side(dat, day, n_pos_rank, N):
    inf_log_f = pd.DataFrame(dat["infect_log"])
    state=epian.get_state(N, dat["people_dates"],day)
    contacts=dat["contacts"]
    ranks_all=dat["rk_extra_stats"].item()
    ranks_day=ranks_all[day]
    test_res=dat["test_stats"]

    u = find_forw_back_side(test_res, inf_log_f, day, ranks_day, state, contacts, n_pos_rank)
    return [len(v) for v in u]

def find_idc_back(inf_log_f, idc_obsi, t_rk):
    inf_log = inf_log_f[inf_log_f["date"]<t_rk]
    ### backward
    df_back=inf_log[np.isin(inf_log["target"],idc_obsi)]
    sources_back = df_back[df_back["source"]>=0]["source"]
    ## remove idc already observed
    sources_back = sources_back[~np.isin(sources_back, idc_obsi)]

    return sources_back

def find_idc_forw(inf_log_f, idc_obsi, t_rk):

    inf_log = inf_log_f[inf_log_f["date"]<t_rk]

    ff_idc = inf_log[inf_log.source.isin(idc_obsi)].target
    ## remove idc already observed
    ff_idc = ff_idc[~np.isin(ff_idc, idc_obsi)]

    return ff_idc

def select_idcs_forw_back(dat, t_lim, N, state):
    """
    Give indices and stats
    """
    inf_log = DataFrame(dat["infect_log"])
    tests_st = DataFrame(dat["test_stats"])
    
    if "rk_extra_stats" in dat:
        ranks = dat["rk_extra_stats"].item()
        rank_ser = DataFrame(ranks[t_lim]).set_index("idx")["val"]
    else:
        rank_ser = None
    #state = get_state(N, dat["people_dates"],t_lim)
    ## find tested infected
    df_obs_i=tests_st[(tests_st.res_state==1) & (tests_st.date_res<t_lim)]

    counts=df_obs_i["i"].value_counts()
    infected = (state > 0) & (state < 7)
    inf_idcs = np.where(infected)[0]

    idcs_obs = np.unique( df_obs_i["i"])
    # backward nodes, who have infected the observed, but have not been observ.
    idc_back=np.unique(find_idc_back(inf_log,idcs_obs, t_lim))
    # forward nodes, have been infected by the observed (directly), not observ
    idc_forw=np.unique(find_idc_forw(inf_log,idcs_obs, t_lim))
    #backward of backward
    idcs_back_back = np.unique(find_idc_back(inf_log, idc_back, t_lim))
    
    
    frac_back_i = (infected[idc_back]).mean()
    idc_back_i = idc_back[infected[idc_back]]
    
    frac_forw_i = (infected[idc_forw]).mean()
    idc_forw_i = idc_forw[infected[idc_forw]]

    frac_b2_i = (infected[idcs_back_back]).mean()
    idc_b2_i = idcs_back_back[infected[idcs_back_back]]
    
    ### sideward
    inf_log_cut= inf_log[inf_log["date"]<t_lim]
    #poss_side,_,_ = forback.find_possible_side(inf_log_cut, idcs_obs)
    #contacts = dat["contacts"]
    #idcs_side, _, _ = filter_infect_contacts(tests_st, contacts, poss_side,t_lim)
    ## other forwards
    other_inf_forw = find_possible_side(inf_log_cut, idcs_obs)[0]
    #inf_distance = forw_idcs_distance(other_inf_forw)
    #forw_sec = np.array(list(set(inf_distance[2])))
    forw_sec = np.array(
        forw_idcs_min_d(other_inf_forw, d_min=2, d_max=3)
    )
    frac_forw_sec_i = (infected[forw_sec]).mean()
    idc_forw_sec_i = forw_sec[infected[forw_sec]]
    
    #idcs_side = np.unique(list(idcs_side))
    ## filter side by status
    #idcs_side_inf = idcs_side[infected[idcs_side]] if len(idcs_side) > 0 else []
    
    idcs_inf_else = set(np.where(infected)[0]).difference(idcs_obs).difference(idc_forw_i).\
        difference(idc_back_i).difference(idc_forw_sec_i).difference(idc_b2_i)
    #.difference(idcs_side_inf)
    indices_find = {"back": idc_back_i, "forw": idc_forw_i,
         "forw_2": forw_sec, "back_2": idc_b2_i, "frac_b2_i": frac_b2_i, "else": idcs_inf_else}
    return idcs_obs,indices_find,rank_ser, \
        (frac_back_i, frac_forw_i, frac_forw_sec_i, (counts).mean(), (counts>1).mean())

def auc_roc_ranking(idc_find, idc_exclude, ranks, N):
    idc_sel=sorted(set(range(N)).difference(idc_exclude))
    idc_find = list(set(idc_find).intersection(idc_sel))
    vals_forw = Series(np.zeros(len(idc_sel)),index=idc_sel)
    vals_forw.loc[idc_find] = 1

    return roc_auc_score(vals_forw.sort_index(),ranks.loc[idc_sel].sort_index())

def auc_roc_ranking_idc(idc_find, idc_ranking, ranks):
    idc_sel=list(idc_ranking) #sorted(set(range(N)).difference(idc_exclude))
    
    idc_find = list(set(idc_find).intersection(idc_sel))
    if len(idc_find) == 0:
        print(f"No indices to find, before intersect where {len(idc_sel)}")
    vals_forw = Series(np.zeros(len(idc_sel)),index=idc_sel)
    vals_forw.loc[idc_find] = 1

    return roc_auc_score(vals_forw.sort_index(),ranks.loc[idc_sel].sort_index())

def prep_ranking(idc_find, idcs, ranks, N, exclude_idcs=False):
    """
    Prepare ranking for sklearn methods

    input: indices to put as 1s, all idcs to include in the metric, rank in pandas series, N
    """
    if exclude_idcs:
        idc_sel=sorted(set(range(N)).difference(idcs))
    else:
        idc_sel = list(idcs)
    #idc_find = list(set(idc_find).intersection(idc_sel))
    vals_forw = Series(np.zeros(len(idc_sel)),index=idc_sel)
    vals_forw.loc[idc_find] = 1

    return vals_forw.sort_index(),ranks.loc[idc_sel].sort_index()

def find_possible_side(inf_log, iobs):
    log_noobs=inf_log[~inf_log.target.isin(iobs)]
    whoinf=dict(zip(log_noobs["target"],log_noobs["source"]))
    inf_path={}
    c=0
    for l in range(len(log_noobs)):
        src = log_noobs["source"].iloc[l]
        trg = log_noobs["target"].iloc[l]
        #if trg in inf_path:
        if src <0:
            continue
        inf_path[trg] = [src]
        c+=1
        if src in whoinf:
            src = whoinf[src]
            #if src in iobs:
            #    #print("Stop")
            inf_path[trg].insert(0,src)
            c+=1
            while src >=0:
                if src in whoinf:
                    src = whoinf[src]
                    inf_path[trg].insert(0,src)
                else:
                    src = -2

                c+=1
        #print(f"{l:4d}/{len(log_noobs):4d}",end="\r")
    #print("\n",c)
    
    deleted=0
    fow_p_side=defaultdict(set)
    for k in sorted(inf_path.keys()):
        #print(k, len(inf_path[k]), sum(np.isin(inf_path[k], iobs)))
        if len(inf_path[k])>1 and inf_path[k][0] in iobs:
            #print(k, inf_path[k])
            assert np.any(np.isin(inf_path[k][1:], iobs))==False
            lu = inf_path[k]
            for i in range(2,len(lu)):
                fow_p_side[lu[0]].add((lu[i],i))
            fow_p_side[lu[0]].add((k,len(lu)))
        else:
            del inf_path[k]
            deleted+=1
    
    return fow_p_side, deleted, inf_path

def forw_idcs_distance(fow_poss_side):
    oth_inf_f= defaultdict(list)
    for k, d in fow_poss_side.items():
        for l in d:
            oth_inf_f[l[1]].append(l[0])
    return oth_inf_f

def forw_idcs_min_d(fow_poss_side, d_min=2, d_max=1000):
    oth_inf_f= list() #defaultdict(list)
    for k, d in fow_poss_side.items():
        for l in d:
            if l[1] >= d_min and l[1] <= d_max:
                oth_inf_f.append(l[0])
            #oth_inf_f[l[1]].append(l[0])
    return oth_inf_f