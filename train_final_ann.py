#Andrew Geiss, 20 Dec 2023

import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from uuid import uuid4
from neural_networks import serial_ann, tanh_perc, ssa_mae, g_mae, combined_loss, load_dataset

def lr_schedule(epoch, lr):
    if (epoch+1) % 10 == 0:
        lr*=0.1
    return lr

SCAT_MODE = 'sphere' #either 'sphere' or 'coreshell'
fname = './data/models/' + SCAT_MODE + '.h5'
ninputs = {'sphere': 4, 'coreshell': 7}
nneurons = {'sphere': 69, 'coreshell': 112}
nlayers = {'sphere': 4, 'coreshell': 4}
model = serial_ann(nlayers[SCAT_MODE],nneurons[SCAT_MODE],ninputs[SCAT_MODE])
inputs, targets, _ = load_dataset([0,90_000_000],SCAT_MODE)
model.compile(optimizer=Adam(learning_rate = 0.001), loss = combined_loss, metrics=[tanh_perc,ssa_mae,g_mae])
model.fit(inputs,targets,batch_size=64,epochs=33,callbacks=LearningRateScheduler(lr_schedule,verbose=0),verbose=2)
model.save(fname)