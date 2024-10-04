#functions defining neural networks, loss functions and custom metrics, and pre- and post-processing functions

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

#Functions for building neural networks:
def serial_ann(nlayers, nneurons, ninputs):
    xin = Input((ninputs,))
    x = xin
    for i in range(nlayers):
        x = Dense(nneurons,activation='swish')(x)
    x = Dense(3)(x)
    return Model(xin,x)
    
def random_serial_ann(nparams, ninputs):
    
    LAYER_COUNT_RANGE = (2,5)
    nlayers = np.random.randint(*LAYER_COUNT_RANGE)
    NOUT = 3
    
    def param_count(neurons):
        return neurons*(ninputs+NOUT+(neurons+1)*(nlayers-1)+1)+NOUT
    
    neurons = 1
    while param_count(neurons) < nparams:
        neurons += 1
    
    print('Random Serial ANN Summary:')
    print('Weight + Bias Count: ' + str(param_count(neurons)) )
    print('Number of Layers: ' + str(nlayers))
    print('Layer Sizes: ' + str(neurons) + '\n')
    
    return serial_ann(nlayers,neurons,ninputs)

#Metrics:
def tanh_perc(y_true,y_pred):
    ke_true = tf.math.exp(tf.clip_by_value(y_true[:,0]-2,-12,3.5))
    ke_pred = tf.math.exp(tf.clip_by_value(y_pred[:,0]-2,-12,3.5))
    return 100*tf.reduce_mean(tf.math.tanh(tf.math.abs(1-ke_pred/ke_true)))

def g_mae(y_true,y_pred):
    g_pred = tf.math.sigmoid(y_pred[:,2])
    return tf.reduce_mean(tf.math.abs(y_true[:,2] - g_pred))

def ssa_mae(y_true,y_pred):
    ssa_pred = tf.math.sigmoid(y_pred[:,1])
    return tf.reduce_mean(tf.math.abs(y_true[:,1] - ssa_pred))

def combined_loss(y_true, y_pred):
    g_ssa_pred = tf.math.sigmoid(y_pred[:,1:])
    g_ssa_true = y_true[:,1:]
    g_ssa_mae = tf.reduce_mean(tf.math.abs(g_ssa_true-g_ssa_pred))
    ln_ke_true = y_true[:,0]-2
    ln_ke_pred = y_pred[:,0]-2
    rmsle = tf.math.sqrt(tf.reduce_mean((ln_ke_pred - ln_ke_true)**2))
    return rmsle+g_ssa_mae

#data loading, preprocessing, and postprocessing functions:
def load_dataset(ind_range,scat_mode):
    
    #load the data:
    inputs = np.load('./data/inputs.npy')[ind_range[0]:ind_range[1],:]
    upper_bounds = np.load('./data/upper_x.npy')[ind_range[0]:ind_range[1]]
    targets = np.load('./data/' + scat_mode + '_targets.npy')[ind_range[0]:ind_range[1],:]

    #remove Rayleigh scattering cases:
    keep = upper_bounds>0.1
    targets = targets[keep,:]
    inputs = inputs[keep,:]
    
    #apply corrections to Ke and Ks and convert to SSA:
    wavelength = inputs[:,0]
    ke = targets[:,0]
    ke[ke<0] = 0
    ks = targets[:,1]
    ks[ks>ke] = ke[ks>ke]
    ssa = ks/ke
    ssa[ks==ke] = 1
    ln_ke_lambda_rho = np.log(ke*wavelength)
    targets[:,0] = ln_ke_lambda_rho
    targets[:,1] = ssa

    #express inputs in terms of mu_x and scale them:
    inputs[:,1] = 2*np.pi*inputs[:,1]/wavelength
    inputs = inputs[:,1:]
    inputs[:,0] = (np.log(inputs[:,0])+1.5)/3.6
    inputs[:,[1,2,4]] = 2*inputs[:,[1,2,4]]-4
    inputs[:,[3,5]] = (np.log(inputs[:,[3,5]])+9)/5
    inputs[:,6] = inputs[:,6]*4-2
    if scat_mode == 'sphere':
        inputs = inputs[:,:4]
    
    return inputs, targets, wavelength

def sigmoid(x):
    return 1/(1+np.exp(-x))

def scale_outputs(outputs,wavelength):
    ke_rho = np.exp(outputs[:,0])/wavelength
    ssa = sigmoid(outputs[:,1])
    g = sigmoid(outputs[:,2])
    return ke_rho, ssa, g