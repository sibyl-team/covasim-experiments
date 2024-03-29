from collections import defaultdict
from warnings import warn
import warnings
import numpy as np
from pandas import Series, DataFrame
import forw_back as forback
import epidemic as epian

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def compute_aucs_forback(mres, t_rk,seeds,N, random_ranks=False,other_ranks=None):
    """
    Compute AUCs for the forward backward.

    Returns counts, aucs (backward, forward, forward2+), average prec and count infected, and the ROCs
    """
    c=[]
    aucs=[]
    aver_prec = []
    counts_infect = []
    rocs={"forw":[], "back":[], "idc_back":[]}
    for i in sorted(seeds):
        state = epian.get_state(N, mres[i]["people_dates"],t_rk)
        iobs, idc_find, rank, l=forback.select_idcs_forw_back(mres[i], t_rk, N, state)
        if rank is None:
            if other_ranks is None:
                print(f"No ranking found for seed {i}",end=" ")
                raise ValueError("You have to give a ranking if it's missing from the res dict")
            rank = other_ranks[i][t_rk]
        
        iback = idc_find["back"]
        ifor = idc_find["forw"]
        iforw_sec = idc_find["forw_2"]
        iback2 = idc_find["back_2"]

        inf_rest = idc_find["else"]
        #print(i,l[-1])
        c.append(list(l))
        ib_only=set(iback).difference(ifor).difference(iforw_sec)
        if_only = set(ifor).difference(iback).difference(iforw_sec)
        if2_only = set(iforw_sec).difference(ifor).difference(iback)
        ibb2 = set(iback2).difference(ifor).difference(iforw_sec)
        assert len(set(iobs).intersection(ifor))==0
        assert len(set(iobs).intersection(iforw_sec))==0
        assert len(set(iobs).intersection(inf_rest)) == 0
        counts_infect.append([len(iobs), len(set(iback)), len(set(ifor)), len(iforw_sec),
                                len(iback2),
                              len(inf_rest), len(ib_only), len(if_only), len(if2_only),
                              len(ibb2)])
        if random_ranks:
            rank= Series(np.random.rand(len(rank)))
        rank = rank/rank.max()
        assert isinstance(rank, Series)
        ##assert len(set(iback).intersection(iside)) == 0
        ##assert len(set(ifor).intersection(iside)) == 0
        #print(i, cs, len(set(iback).intersection(ifor)))
        #ib_find=list(set(iback).difference(ifor)) #np.setdiff1d(iback,ifor)
        #if_find=list(set(ifor).difference(iback))
        #iside_find = list(set(iside).difference(iback))
        
        all_notinf = np.where((state <1) | (state >=7))[0] #set(range(N)).difference(all_inf)
        rocs["idc_back"].append(iback)
        if len(iback)>0:
            idc_choose=set(iback).union(all_notinf)
            r = forback.prep_ranking(iback,idc_choose ,rank, N, exclude_idcs=False)
            auc_back= roc_auc_score(*r)
            ap_back = average_precision_score(*r)
            rocs["back"].append(roc_curve(*r))
        else:
            print(f"Seed {i} has no backward")
            auc_back = np.nan
            ap_back = np.nan

        sn_forw =  set(ifor).union(all_notinf)
        rf = forback.prep_ranking(ifor,sn_forw ,rank, N, exclude_idcs=False)
        forw2_dat = forback.prep_ranking(iforw_sec, set(iforw_sec).union(all_notinf), rank, N, exclude_idcs=False)
        auc_b2, ap_b2, roc_b2 = forback.get_rocs_aucs(iback2, rank, N, all_notinf, print_msg=f"Seed {i} has no back2", roc=False)
        aucs.append((auc_back,
                    roc_auc_score(*rf),
                     ##dat_side[0],
                     roc_auc_score(*forw2_dat),
                     auc_b2
                   ))
        aver_prec.append((ap_back,
                          average_precision_score(*rf), 
                          #dat_side[1],
                          average_precision_score(*forw2_dat),
                          ap_b2
                         ))
        rocs["forw"].append(roc_curve(*rf))
        print(f"{i:4d}/{len(seeds):4d}", end="\r")
    print("")
    #print(c[0])
    c=np.stack(c)
    aucs=np.stack(aucs)
    aver_prec = np.stack(aver_prec)
    return c, aucs, aver_prec, np.stack(counts_infect), rocs

def find_infection_counts(tt):
    newhist = defaultdict(list)
    counts = defaultdict(lambda: 0)
    detailed = tt.detailed.to_dict('records')
    for i,entry in enumerate(detailed):
        if ~np.isnan(entry['target']):
            idx_tg = int(entry["target"])
            if idx_tg in newhist:
                raise ValueError()
            newhist[idx_tg] = [idx_tg]
            source = entry['source']
            lastsrc=-1
            while ~np.isnan(source) and source > 0:
                lastsrc = int(source)
                newhist[i].insert(0, lastsrc)
                source = detailed[lastsrc]['source']
            if lastsrc > 0:
                counts[lastsrc]+=1
            else:
                counts[idx_tg]+=1

    return newhist, counts

def get_sim_res_all(allres, key, seeds):
    vmean = []
    for s in seeds:
        x = allres[s]["sim_res"][key]
        if hasattr(x,"values"):
            x = x.values
        vmean.append(x)
    vmean = np.stack(vmean)
    return vmean

def _find_supersp_by_inf(infect_log, ninf_super: int):
    infectors, n_infected=np.unique(infect_log.source[infect_log.source>=0], return_counts=True)
    superspread=infectors[n_infected>= ninf_super]
    return superspread

def _check_cut_infect_log(inf_log, t_max:int=None):
    if t_max is not None:
        inf_log=inf_log[inf_log["date"]<=t_max]
    return inf_log

def filter_idcs_inf(inf_log, idcs):
    """
    Filter indices selecting only those resulting as infected in the log
    """
    sups_infected = set(inf_log["target"]).intersection(idcs)
    return  list(sups_infected)

def count_superspread(infect_log,ranks_day, tests_stats, ninf_super=11, 
        n_rank_pos=100, t_max=None, supersp_idcs=None, ss_notobs=False, debug=False, filter_sups=False):
    """
    Count the number of superspreaders found
    """
    ## cut infection log to before the time of 
    if t_max is not None:
        infect_log=infect_log[infect_log["date"]<=t_max]
        tests_stats = tests_stats[tests_stats["date_res"]<t_max]
    
    if supersp_idcs is None:
        
        superspread = _find_supersp_by_inf(infect_log, ninf_super)
    else:
        if debug: print(f"n ss before: {len(supersp_idcs)}", end=" ")
        superspread = filter_idcs_inf(infect_log, supersp_idcs) if filter_sups else supersp_idcs
        if debug: print(f"after: {len(superspread)}")
    
    inf_obs = tests_stats[tests_stats["res_state"]==1]

    if ss_notobs:
        idcs_iobs = np.unique(inf_obs["i"])
        #print(f"num ss not obs: {len(set(superspread).difference(idcs_iobs))*100 / len(superspread)}")
        superspread = np.setdiff1d(superspread, idcs_iobs)
    
    v=infect_log[np.isin(infect_log["target"], superspread)]
    ## they are infected
    sups_date = Series(index=v["target"],data=v["date"])

    day_ss_found=Series(np.ones(len(superspread),dtype=int)*-1000,index=superspread, dtype=np.int)
    day_perfect= sups_date+1
    day_perfect[day_perfect<min(ranks_day.keys())] = min(ranks_day.keys())
    # find min obs time for supersp

    for d, rk in ranks_day.items():
        if t_max is not None and d>t_max:
            break
        # exclude all observed indivi
        inf_obs_day = np.unique(inf_obs[inf_obs["date_res"]<d]["i"])
        # and sups found at this day
        sups_found_day = day_ss_found[(day_ss_found>=0)&(day_ss_found<d)].index
        idcs_remove = list(set(inf_obs_day).union(sups_found_day))
        try:
            it = rk.sort_values(ascending=False)
            it = it.iloc[~np.isin(it.index, idcs_remove)]
            indc_val = it.index.values[:n_rank_pos]
            #.index.values
        except AttributeError:
            ### we have a numpy array
            ## sort by val, reverse
            ii = np.argsort(rk["val"])[::-1]
            ii_choose = ii[~np.isin(ii, idcs_remove)][:n_rank_pos]
            # apply order to idx 
            indc_val = rk["idx"][ii_choose]

        #supersp to find
        su_find = day_ss_found[day_ss_found<0].index
        # that have not been observed yet
        #su_find = np.setdiff1d(su_find, date_obs_ss[date_obs_ss<d].index)
        # find the supersp that are in the ranking
        idc_found = list(set(su_find).intersection(indc_val))
        ## set the day of the ranking
        day_ss_found[idc_found] = d

        if debug: print(f"day : {d}, n found: {(day_ss_found>=0).sum()}")
    
    return day_ss_found, day_perfect

def pars_infect_log(x):
        if x["source"] is None:
            x["source"] = -1
        return x

def find_supersp_sim_tests(sim,  ninf_super=8, test_stats=None, test_idcs=None,
    t_max=None, supersp_idcs=None, ss_notobs=False,debug=False, filter_sups=False):
    inf_log = DataFrame(map(pars_infect_log, 
            sim.people.infection_log)).to_records(index=False)

    idc_d=sim.people.diagnosed.nonzero()[0]

    date_d=sim.people.date_diagnosed[idc_d]

    #test_data =np.fromiter(zip(idc_d, date_d, np.ones(len(idc_d),dtype=int)), 
    #           dtype=np.dtype([("i",int),("date_res",int), ("res_state",int)]))
    if test_stats is None:
        test_data = np.rec.fromarrays((idc_d, date_d.astype(int), np.ones(len(idc_d),dtype=int)),
                      names=["i","date_res","res_state"])
    else:
        test_data = test_stats[test_stats["res_state"]==1]
    
    if t_max is not None:
        ##
        test_data = test_data[test_data["date_res"]< t_max]

    if test_idcs is None:
        tested=sim["interventions"][0].tested_idcs_rnd
    else:
        tested = test_idcs
    inf_log = _check_cut_infect_log(inf_log, t_max)
    if supersp_idcs is None:
        
        superspread = _find_supersp_by_inf(inf_log, ninf_super)
    else:
        #sups_infected = set(inf_log["target"]).intersection(supersp_idcs)
        #superspread = list(sups_infected)
        pot_f = len(supersp_idcs)
        superspread = filter_idcs_inf(inf_log, supersp_idcs) if filter_sups else supersp_idcs
        if debug: print(f"N supersp before: {pot_f}, now: ", len(superspread))
        #inf_log = _check_cut_infect_log(inf_log, t_max)
    if ss_notobs:
        idc_obsi = np.unique(test_data["i"])
        #print(f"num ss not obs: {len(set(superspread).difference(idc_obsi)) / len(superspread)}")
        superspread = np.setdiff1d(superspread, idc_obsi)
    
    day_rank=Series(np.full(len(superspread),-900),index=superspread, dtype=np.int_)

    for d in sorted(tested.keys()):
        if t_max is not None and d>t_max:
            break
        tt = tested[d]

        iobs_day=test_data[test_data["date_res"]<d]
        ss_notfound = day_rank[day_rank<0].index
        ss_tofind = np.intersect1d(superspread, ss_notfound)
        ## choose only the superspreaders
        ss_find = np.intersect1d(tt,ss_tofind)
        ## remove those already found infected
        for i in np.setdiff1d(ss_find, iobs_day[iobs_day["res_state"]==1]["i"]):
            #print(i)
            if debug and day_rank[i]>=0:
                print(f"ERROR, i {i} already found")
            assert day_rank[i]<0
            day_rank[i] = d #if day_rank[i] == -200 else max(d,day_rank[i])
        
        #if debug: print("Day rank: ", day_rank)
    return (day_rank,)

    ### get the observed superspreaders
    """supsp_obs = inf_obs[np.isin(inf_obs["i"], superspread)]
    ii = np.unique(supsp_obs["i"])
    date_obs_ss = Series(data=np.full(len(ii),-2,dtype=int), index=ii)
    for i,t in zip(supsp_obs["i"], supsp_obs["date_res"]):
        if date_obs_ss[i]<0:
            date_obs_ss[i] = t
        elif date_obs_ss[i]>t:
            date_obs_ss[i] = t
    assert np.all(date_obs_ss>=0)
    """