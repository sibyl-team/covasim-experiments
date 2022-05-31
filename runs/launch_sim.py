import covasim as cv
import base_sim_run as bs
from importlib import reload

import numpy as np

N = int(20e3)
T = 100
seed = 2
params = bs.make_std_pars(N,T, seed=seed)
popfile = bs.get_people_file(seed, N)
print(popfile)


from covasibyl.test_num import TestNumQuar
from covasibyl import analyzers as analy
reload(analy)

#ct = cv.contact_tracing(trace_probs=1., trace_time=0, start_day=10)

sim_ct = cv.Sim(pars=params, interventions=[],#[TestNumQuar(400,start_day=10), ct],
    popfile=popfile,
    label="Covasim TTQ",
    analyzers=analy.store_seir()
)
sim_ct.run()

anl = sim_ct["analyzers"][0]

np.savez_compressed(f"counts_sim_s{seed}", counts=anl.out_save())