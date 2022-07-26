import numpy as np

def get_rank_forw_back(ranks_t, test_stats, inf_log):
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
    n_ranking = 100
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