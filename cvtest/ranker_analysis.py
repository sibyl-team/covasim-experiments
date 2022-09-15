from collections import defaultdict
from warnings import warn
import numpy as np
from pandas import Series

def get_rank_forw_back(ranks_t, test_stats, inf_log, n_ranking=100):
    """
    Find the number of infections a ranker is able to 
    find, either backward or forward in the transmission tree,
    from the observations of infected inviduals

    ranks_t: dict of all ranks for time
    test_stats: all the test statistics from a run
    inf_log: DataFrame of the log of the infections from covasim, 
             !!assuming that seed infections have negative source!!
    """
    counts_back = {}
    counts_for = {}
    discovered = set()
    pos_tests = test_stats[test_stats["res_state"]==1]
    ## another version
    for d, rk in ranks_t.items():
        if np.abs(rk.sum()-rk.mean()) <1e-12:
            continue
        #print(rk.sum(), rk.mean())#-rk.mean()))
        obs_ = pos_tests[pos_tests["date_res"] == (d-1)]["i"]

        rk_cut= rk.sort_values(ascending=False)[:n_ranking]
        infectors = inf_log[inf_log.target.isin(obs_) & (inf_log.date < d) ]
        infectors = infectors[infectors.source >= 0]
        #print(infectors)
        #infect_rel[(infect_rel.date_obs==(d-1)) & (infect_rel.date < d)]
        src_tofind = (infectors.source.values)
        inf_found = rk_cut.index[
            rk_cut.index.isin(src_tofind) ]#.sum()
        new_f = set(inf_found).difference(discovered)
        c = len(new_f)
        #print(d,set(inf_found),"new: ", new_f)
        counts_back[d] = c
        discovered.update(inf_found)
        
        ## infected (forward)
        assert np.all(infectors.target.isin(obs_))
        toinf = inf_log[inf_log.source.isin(obs_) & (inf_log.date <= d)]
        inf_tofind = (toinf.target.values)
        inf_found = rk_cut.index[
            rk_cut.index.isin(inf_tofind) ]
        ## save new
        new_f = set(inf_found).difference(discovered)
        #print(set(inf_found),"new: ", new_f)
        counts_for[d] = len(new_f)
        discovered.update(inf_found)

    return dict(forward=counts_for, backward=counts_back)
    #counts_for, counts_back

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

def count_superspread(infect_log,ranks_day, tests_stats, ninf_super=11, n_ranks=100):
    infectors, n_infected=np.unique(infect_log.source[infect_log.source>=0], return_counts=True)

    superspread=infectors[n_infected>= ninf_super] 
    #pd.Series(data=n_infected, index=infectors)

    inf_obs = tests_stats[tests_stats["res_state"]==1]
    supsp_obs=inf_obs[np.isin(inf_obs["i"], superspread)]
    
    v=infect_log[np.isin(infect_log["target"], superspread)]
    sups_date = Series(index=v["target"],data=v["date"])

    
    day_rank=Series(np.ones(len(superspread),dtype=int)*-200,index=superspread)
    day_perfect= sups_date+1
    day_perfect[day_perfect<min(ranks_day.keys())] = min(ranks_day.keys())
    # find min obs time for supersp
    day_o={}
    for r in supsp_obs:
        i,t=(r.i, r.date_res)
        if i not in day_o:
            day_o[i] =t 
        else:
            day_o[i] = min(t,day_o[i])
    ### convert to series for easier access
    day_obs = Series(day_o)

    for d, rk in ranks_day.items():
        try:
            it = rk.sort_values(ascending=False)[:n_ranks].index
        except AttributeError:
            ### we have a numpy array
            ## sort by val, reverse
            ii = np.argsort(rk["val"])[::-1]
            # apply order to idx and take first 
            it = rk["idx"][ii][:n_ranks]

        #supersp to find
        su_find = day_rank[day_rank<0].index
        #supersp not obs yet
        i_notf = day_obs[day_obs>=d].index
        su_find = np.setdiff1d(su_find,i_notf)
        # find the supersp that are in the ranking
        found_b = np.isin(su_find, it)
        idc_found=su_find[found_b]
        ## set the day of the ranking
        day_rank[idc_found] = d

    #print("Found:",(day_rank>=0).sum())
    
    return day_rank, day_perfect