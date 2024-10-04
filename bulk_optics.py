#Andrew Geiss, 31 Jan 2023

import numpy as np
from scipy.special import erfinv, erf
from TAMie import sphere, coreshell

#define constants
nrad = 1024             #resolution of the numerical integration
intfrac = 0.999         #how much of the distribution to capture

#trapezoidal integration
def integrate(x,y):
    dx = x[1:] - x[:-1]
    my = y[1:] + y[:-1]
    return np.sum(0.5*dx*my)

#determines bounds of integration
def integration_bounds(mug,sigmag):
    t = np.array([(1-intfrac)/2,intfrac+(1-intfrac)/2])
    return np.exp(erfinv(2*t-1)*np.log(sigmag)*np.sqrt(2))*mug

#calculates the bulk extinction and scattering efficiencies (times rho, see companion manuscript)
def mass_efficiency(wavelength,mug,sigmag,refr,refi,refrc=None,refic=None,f=None):
    
    #determine integration bounds and values of r
    rmin, rmax = integration_bounds(mug,sigmag)
    r = np.exp(np.linspace(np.log(rmin),np.log(rmax),nrad))
    
    #calculate the particle optical properties:
    Qe, Qs, g = np.zeros(r.shape), np.zeros(r.shape), np.zeros(r.shape)
    for i in range(len(r)):
        xs = 2*np.pi*r[i]/wavelength
        if refrc is None:
            Qe[i], Qs[i], g[i] = sphere(refr+1j*refi, xs)
        else:
            Qe[i], Qs[i], g[i] = coreshell(refrc+1j*refic, refr+1j*refi, f*xs, xs)
    
    #calculate the bulk efficiencies:
    epdf = np.exp(-(np.log(r/mug)**2)/(2*np.log(sigmag)**2))  #the exponential term from the log-normal pdf
    #integrate in log-space so the missing 1/r in epdf cancels with the r from rdln(r)
    ke_rho = 0.75*integrate(np.log(r),Qe*epdf*r**2)/integrate(np.log(r),epdf*r**3)
    ks_rho = 0.75*integrate(np.log(r),Qs*epdf*r**2)/integrate(np.log(r),epdf*r**3)
    g = integrate(np.log(r),g*Qs*epdf*r**2)/integrate(np.log(r),Qs*epdf*r**2)
    
    return ke_rho, ks_rho, g