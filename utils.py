#Andrew Geiss, 31 Jan 2023
#
#This script contains information about the range of plausible refractive indices 
#and wavelengths that might occur in EAM / MAM and contains functions for generating
#random training samples.

import numpy as np
#the range of potential refractive indices:
refr_range = {'sw': [1.1, 2.2],  'lw': [1.1, 3.0], 'all': [1.1, 3.0]}
refi_range = {'sw': [1E-8,0.8],  'lw': [1E-8,1.0], 'all': [1E-8,1.0]}
              
#the range of potential parameters defining aerosol size distributions:
sigma_range = [1.2,2.8]
mu_range = [5E-9, 5E-5]     #(m)

#the optical band centers used in E3SM by wavelength in meters:
swb_bounds = 0.01/np.array([[2600,3250,4000,4650,5150,6150,7700,8050,12850,16000,22650,29000,38000,820],
            [3250,4000,4650,5150,6150,7700,8050,12850,16000,22650,29000,38000,50000,2600]],dtype='double').T
lwb_bounds = 0.01/np.array([[10,350,500,630,700,820,980,1080,1180,1390,1480,1800,2080,2250,2390,2600],
            [350,500,630,700,820,980,1080,1180,1390,1480,1800,2080,2250,2390,2600,3250]],dtype='double').T
wvl_bands = {'sw': np.mean(swb_bounds,axis=1),
             'lw': np.mean(lwb_bounds,axis=1)}
wvl_bands['all'] = np.sort(np.concatenate([wvl_bands['sw'],wvl_bands['lw']]))
wvl_range = {'sw': [np.min(swb_bounds),np.max(swb_bounds)],
             'lw': [np.min(lwb_bounds),np.max(lwb_bounds)]}
wvl_range['all'] = [np.min(swb_bounds),np.max(lwb_bounds)]

def log_uniform(xmin,xmax,size):
    #draws from a log-uniform distribution
    return np.exp(np.random.uniform(np.log(xmin),np.log(xmax),size=size))

def random_params(wvl_regime,N=None):
    #generates randomly selected aerosol distributions (for creating training/testing data)
    wavelength = log_uniform(*wvl_range[wvl_regime],N)
    refr = np.random.uniform(*refr_range[wvl_regime],size=N)
    refi = log_uniform(*refi_range[wvl_regime],N)
    sigma = np.random.uniform(*sigma_range,size=N)
    mu = log_uniform(*mu_range,N)
    return wavelength, mu, sigma, refr, refi

def random_params_sc(wvl_regime,N=None):
    #generates randomly selected aerosol distributions (for creating training/testing data)
    wavelength, mu, sigma, refrs, refis = random_params(wvl_regime,N)
    refrc = np.random.uniform(*refr_range[wvl_regime],size=N)
    refic = log_uniform(*refi_range[wvl_regime],N)
    f = np.random.uniform(0,0.98,size=N)
    return wavelength,mu,sigma,refrs,refis,refrc,refic,f