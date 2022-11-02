#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from pathlib import Path


#import sys
#sys.path.append("../../")
#from base_sim_run import make_filename


def make_filename(args, N:int,T:int,seed:int, rk_name:str):
    fnr_str = ""
    if args.adopt_fraction < 1:
        fnr_str+=f"_AF_{round(args.adopt_fraction,2)}"
    if args.fnr > 0:
        fnr_str+=f"_fnr_{round(args.fnr,3)}"
    if args.fpr > 0:
        fnr_str+=f"_fpr_{round(args.fpr,3)}"   

    savefile_name = args.prefix +f"epi_kc_{int(N/1000)}k_T_{T}{fnr_str}_s_{seed}_rk_{rk_name}"

    return savefile_name
# In[ ]:


parser = argparse.ArgumentParser(description="Check run")

parser.add_argument("folder", type=str, help="Saving folder")
parser.add_argument("rank", type=str, help="Ranker name")
parser.add_argument("-T", type=int, default=100)
parser.add_argument("-N",type=float,default=50e3)
parser.add_argument("-pf","--prefix", type=str, default="", help="Out file prefix")


parser.add_argument("--fnr", type=float, default=0., help="False negative test rate")
parser.add_argument("--fpr", type=float, default=0., help="False positive rate of testing")
parser.add_argument("-AF","--adopt_fraction", type=float, default=1., help="Fraction of people who report the contacts")

parser.add_argument("--seeds",nargs=2, default=[0,1],type=int)


# In[ ]:


args = parser.parse_args() # "ntest500/MF_good MFex -pf nt500_t2_d11_EI_bmed_newrk_ -AF 1 --fnr 0. --seeds 1 20".split(" "))

print(args)

# In[ ]:


p = Path(args.folder)
if not p.resolve().exists():
    raise ValueError(f"Folder {p} doesn't exist.")

c=0
e=0
rseeds = range(args.seeds[0], args.seeds[1]+1)
for seed in rseeds:
    nn=int(args.N)
    fname = make_filename(args, nn,args.T,seed, args.rank)
    try:
        with np.load(p/f"{fname}_stats.npz") as d:
            last = d["sim_counts"][-1]["t"]
            c+=1
            if last < args.T:
                print(f"seed: {seed}, last time save: {last}")
                print(f"\t{p/fname} is not finished")
                e+=1
    except FileNotFoundError:
        print(f"{p/fname} does not exist")
        e+=1
        #print(f"seed: {seed}, last time save: {last}")
lu = list(rseeds)
print(f"\nChecked {c} files,  {e} are not finished")
print(f"DONE,\nChecked seeds range: {lu[0]} - {lu[-1]}")

