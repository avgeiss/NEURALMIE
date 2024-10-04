#Andrew Geiss, 20 Dec 2023

import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from uuid import uuid4
from neural_networks import random_serial_ann, tanh_perc, ssa_mae, g_mae, combined_loss, load_dataset

def lr_schedule(epoch, lr):
    if (epoch+1) % 30 == 0:
        lr*=0.1
    return lr

SCAT_MODE = 'sphere' # 'sphere' or 'coreshell'
fname = './data/models/random_' + SCAT_MODE + '/' + uuid4().hex + '.h5'
n_inputs = {'sphere': 4, 'coreshell': 7}
inputs, targets, _ = load_dataset([0,80_000_000],SCAT_MODE)
n_params = np.random.randint(500,100_000)
model = random_serial_ann(n_params,n_inputs[SCAT_MODE])
model.compile(optimizer=Adam(learning_rate = 0.001), loss = combined_loss, metrics=[tanh_perc,ssa_mae,g_mae])
model.fit(inputs,targets,batch_size=2048,epochs=100,callbacks=LearningRateScheduler(lr_schedule,verbose=0),verbose=2)
model.save(fname)