'''
Pre-generate the population. Takes about 70 s per population (which
is parallelized by default). 

Requires SynthPops, which must be installed from the repository:

https://github.com/institutefordiseasemodeling/synthpops
'''
from pathlib import Path
import psutil
import sciris  as sc
import covasim as cv
import synthpops as sp
sp.config.set_nbrackets(20) # Essential for getting the age distribution right
sp.logger.setLevel('DEBUG') # Show additional information during population creation

# Settings
pop_size = 20e3 # 10% of the King County population
inputs_f   = Path("pops")
if not inputs_f.exists():
    inputs_f.mkdir(parents=True)
popfile_stem = f'kc_rnr_{int(pop_size/1000)}k_seed' # Stands for "King County revised new run random seed"

def cache_populations(seed=0, popfile=None):
    ''' Pre-generate the synthpops population '''

    pars = sc.objdict(
        pop_size = pop_size,
        pop_type = 'synthpops',
        rand_seed = seed,
    )

    if popfile is None:
        popfile = inputs_f/f'{popfile_stem}{pars.rand_seed}.ppl'

    T = sc.tic()
    print(f'Making "{popfile}"...')
    sim = cv.Sim(pars)
    people = cv.make_people(sim, with_facilities=True,  layer_mapping={'LTCF':'l'}) #generate=True,
    filepath = sc.makefilepath(filename=popfile.as_posix())
    cv.save(filepath, people)
    sc.toc(T)

    print('Done')
    return


if __name__ == '__main__':

    seeds = list(range(50)) #[0,1,2,3,4] # NB, each one takes 1 GB of RAM!
    ram = psutil.virtual_memory().available/1e9
    required = pop_size/225e3*len(seeds)
    if required < ram:
        print(f'You have {ram} GB of RAM, and this is estimated to require {required} GB: you should be fine')
    else:
        raise ValueError(f'You have {ram:0.2f} GB of RAM, but this is estimated to require {required} GB')
    sc.parallelize(cache_populations, iterarg=seeds) # Run them in parallel