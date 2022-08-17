from collections import defaultdict
from warnings import warn
import numpy as np

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

    found = {}
    ids = set()
    for d, rk in ranks_day.items():
        try:
            it = rk[:n_ranks].sort_values(descending=False).index
        except AttributeError:
            ### we have a numpy array
            ii = np.argsort(rk["val"])[::-1]
            it = rk["idx"][ii][:n_ranks]

        u = np.intersect1d(it,superspread)
        newfound = set(u).difference(ids)
        #print(u, newfound, ids)
        #print(d,len(u))
        found[d] = newfound#np.array(newfound)
        ids.update(u)
        #print(d, len(ids))
        
        ### remove obs superspreaders
        irem= supsp_obs[ supsp_obs["date_res"] == d]["i"]
        if len(irem) > 0: 
            #print(f"day {d}, remove {irem} from supsp",superspread)
            superspread = np.setdiff1d(superspread, irem)
            #print(superspread)
    
    return found, ids