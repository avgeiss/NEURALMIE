#Andrew Geiss, Dec 19th, 2023
#
#This script generates training data for the neural networks

#the total number of random samples to generate. The published neural networks used 100_000_000.
ntot_samples = 1_000

#the outputs files are inputs.npy, sphere_targets.py, and coreshell_targets.py
#both networks use the same input file with the 'sphere' network ignoring the last three input columns
#that define the properties of the particle core.

import numpy as np
from utils import random_params_sc
from bulk_optics import mass_efficiency
from multiprocess import Pool
from tqdm import tqdm

def make_inputs(N):
    inputs = np.zeros((N,8),dtype='double')
    rparams = random_params_sc('all',N)
    for i in range(8):
        inputs[:,i] = rparams[i]
    np.save('./data/inputs.npy',inputs)
    
def optics_wrapper(inputs):
    try:
        return mass_efficiency(*inputs)
    except:
        return np.nan, np.nan, np.nan

def make_targets(coreshell=True):
    inputs = np.load('./data/inputs.npy')
    if coreshell:
        fname = './data/coreshell_targets.npy'
    else:
        fname = './data/sphere_targets.npy'
        inputs = inputs[:,:-3]
    input_list = [inputs[i,:] for i in range(inputs.shape[0])]
    p = Pool(24)
    targets = p.map(optics_wrapper,input_list)
    p.close()
    targets = np.array(targets)
    np.save(fname,targets)

#make the training dataset
make_inputs(ntot_samples)
make_targets(coreshell=False)
make_targets(coreshell=True)
