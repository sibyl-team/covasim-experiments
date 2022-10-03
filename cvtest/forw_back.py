import numpy as np
from pandas import Series

def find_idc_backward(inf_log, idc_obsi, idc_ranking, state):
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

def find_idc_forward(inf_log, idc_obsi, idc_ranking, state):

    
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
    backw_I, backw_R = find_idc_backward(inf_log, idc_obsi, idc_ranking, state)
    ### forward
    forw_I, forw_R = find_idc_forward(inf_log, idc_obsi, idc_ranking, state)
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