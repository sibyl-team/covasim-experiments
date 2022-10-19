import numpy as np
import numba as nb

#@nb.njit()
def histo_renorm(y, x=None, nbins=100, return_points=True):
    if x is None:
        x = np.arange(len(y))
    xc = x / x.max()
    bins = np.linspace(0,1, nbins+1)
    vals = np.zeros(nbins)
    for u in range(nbins):
        up = bins[u+1]
        low = bins[u]
        d = (xc >= low) & (xc < up)
        yv = y[d]
        if len(yv)>= 1:
            vals[u] = np.mean(yv)
    if return_points:
        st = (bins[1]-bins[0])/2
        bins = bins[1:]-st
    return bins, vals

@nb.njit()
def histo_renorm_faster(y, x=None, nbins=100, return_points=True):
    if x is None:
        x = np.arange(len(y))
    xc = x / x.max()
    bins = np.linspace(0,1, nbins+1)
    vals = np.zeros(nbins)
    counts = np.zeros(nbins)
    ordr = np.argsort(xc)
    ibin = 0
    for i in ordr:
        xi = xc[i]
        yi = y[i]
        #print(ibin, f"{bins[ibin]:.6f} -- {xi:.6f} -- {bins[ibin+1]:.6f}", )
        
        if ibin+2 < len(bins):
            while xi < bins[ibin] or xi >= bins[ibin+1]:
                #if xi >= bins[ibin]:
                #    break
                ibin+=1
                if ibin == len(bins)-2:
                    # we're at the end
                    break
        #print(f"{bins[ibin]:.6f} -- {xi:.6f} -- {bins[ibin+1]:.6f}", )

        #print(xi, bins[ibin], ibin)
        counts[ibin] +=1
        vals[ibin]+=yi

    #vals = np.divide(#vals/counts
    if return_points:
        st = (bins[1]-bins[0])/2
        bins = bins[1:]-st
    return bins, vals/counts