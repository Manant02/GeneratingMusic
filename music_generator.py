# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 19:15:35 2019

@author: manue
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf #tensorflow==2.0.0
import sys
import os
from datetime import datetime
import pretty_midi
import argparse

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

tf.keras.backend.set_floatx('float64')

tf.keras.backend.clear_session()

#Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--learning-rate', 
                    type=float, 
                    default=0.01, 
                    help='Set the learning rate, default=0.01')
parser.add_argument('--data-dir', 
                    type=str, 
                    default="./data/", 
                    help='Firectory where data is located, default="./data/"')
parser.add_argument('--test-fraction', 
                    type=float, 
                    default=0.1, 
                    help='Fraction which gets used for varifying the model, 0<x<1, default=0.1')
parser.add_argument('--run-id', 
                    type=str, 
                    default="default", 
                    help='Id for identification of run , default="default"')
parser.add_argument('--n-epochs', 
                    type=int, 
                    default=1000, 
                    help='Numer of epochs to train, default=1000')
parser.add_argument('--batch_size', 
                    type=int, 
                    default=1, 
                    help='Batch size, default=1')
parser.add_argument('--n-music', 
                    type=int, 
                    default=3, 
                    help='Number of samples to generate after each epoch, default=3')
parser.add_argument('--gen-freq',
                    default=1,
                    type=int,
                    help='generate samples every x epoch, default=1')
parser.add_argument('--gen-threshold',
                    default=0,
                    type=float,
                    help='minimal activation of pixel in generated sample to be played in midi file, default=0')
parser.add_argument('--shuffle',
                    default=True,
                    type=bool,
                    help='wether to shuffle the dataset between epochs, default=True')
parser.add_argument('--n-samples', 
                    type=int, 
                    default=1400, 
                    help='Number of samples in data_dir to use, set 0 for all samples, default=1400')
parser.add_argument('--stdout-to-file', 
                    type=bool, 
                    default=False, 
                    help='Wether to write stdout to file (True) of to terminal (False), default=False')
parser.add_argument('--run-info',
                    default="default run info",
                    type=str,
                    help='string outputted into run info file')

#parser.add_argument('--', 
#                    type=, 
#                    default=, 
#                    help=', default=')

args, _ = parser.parse_known_args()

#Helper functions
def separate(x):
    y = tf.expand_dims(x, axis=0)
    y = tf.reshape(y, (-1, n_measures, n_hidden6))
    return y

def sample(x):
    noise = tf.keras.backend.random_normal(shape=(n_hidden4,))
    return tf.add(x[0], tf.multiply(tf.exp(0.5 * x[1]), noise))

def get_notelist(midi_array, threshold=0, sampling_frequency=10):
    
    """returns n*4 array containing list of all notes with following info: velocity, pitch, start, end. 
    sampling_frequency must be same as used when creating midi array"""
    
    #returns n*4 array containing list of all notes with following info: velocity, pitch, start, end
    
    toneon = False
    tone_arr = np.zeros((1,4))
    for i in range(midi_array.shape[0]): 
        for j in range(midi_array.shape[1]): 
            
            #start of new tone
            if midi_array[i,j]>threshold and toneon==False and j!=midi_array.shape[1]-1:
                toneon = True
                velocity = midi_array[i,j]*100
                pitch = i + ((128-midi_array.shape[0])//2) #pretty_midi pitch range is 128, we assume midi_array pitch range is centered
                start = j/sampling_frequency
                
                tone_arr = np.append(tone_arr, np.array([[velocity, pitch, start, start]]), axis=0 )         
            
            #end of tone
            elif midi_array[i,j]<=threshold and toneon==True:
                toneon = False
                tone_arr[tone_arr.shape[0]-1, 3] = j/sampling_frequency
                pass
            
            #lasting tone on final tick
            elif midi_array[i,j]>threshold and toneon==True and j==midi_array.shape[1]-1:
                toneon = False
                tone_arr[tone_arr.shape[0]-1, 3] = (j+1)/sampling_frequency
                pass
            
            #tone on final tick
            elif midi_array[i,j]>threshold and toneon==False and j==midi_array.shape[1]-1:
                toneon = False
                velocity = midi_array[i,j]*100
                pitch = i + ((128-midi_array.shape[0])//2) #pretty_midi pitch range is 128, we assume midi_array pitch range is centered
                start = j/sampling_frequency
                
                tone_arr = np.append(tone_arr, np.array([[velocity, pitch, start, start]]), axis=0 )         
                
                tone_arr[tone_arr.shape[0]-1, 3] = (j+1)/sampling_frequency
                pass
            
            pass
        pass
    tone_arr = np.delete(tone_arr, 0, axis=0)
    
    return tone_arr

def get_midifile(input_array, threshold=0, sampling_frequency=10):
    
    """Returns a pretty_midi.PrettyMIDI() object. Input can be midi array or note list.
    Use midi_file.write('filename.mid') to create midi file.
    Note: note velocity hardcoded to 100
    Note: note duration hardcoded to 1/fs seconds"""
    
    #check if input_array is midi array, if so, get note list
    if input_array[0,1] <= 1:
        tone_arr = get_notelist(input_array, threshold=threshold, sampling_frequency=sampling_frequency)
        pass
    else:
        tone_arr = input_array
        pass
        
    #create pretty_midi.PrettyMidi() object based on list of tones
    midi_file = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    
    for i in range(tone_arr.shape[0]):
        note = pretty_midi.Note(int(tone_arr[i,0]) if int(tone_arr[i,0])<=127 else 127, int(tone_arr[i,1]), float(tone_arr[i,2]), float(tone_arr[i,3])) #(velocity, pitch, start, end)
        piano.notes.append(note)
        pass
    midi_file.instruments.append(piano)
    
    return midi_file

def normalize_onoff(midi_array, threshold=0):
    
    """Returns a numpy array with each value in midi_array normalized to 0 or 1, depending on if value is above threshold"""
    
    def on(tone):
        if tone > threshold:
            tone=1
            pass
        else:
            tone=0
        return tone
    
    normalizer = np.vectorize(on)  
    return normalizer(midi_array)

def generate_song():
    global song
    global gendir
    
    gen_img = gendir + "img/"
    gen_npy = gendir + "npy/"
    gen_midi = gendir + "midi/"
    epoch = "_pt"
    
    sample = decoder(np.random.normal(size=[1, 120]))
    outputs_rsp = np.reshape(sample, (16, 96, 96), order='C')
    outputs_rsp = np.concatenate(outputs_rsp[:], axis=1)
    outputs_rsp = outputs_rsp / np.amax(outputs_rsp)
    plt.imsave("{0}epoch{1}_{2}.png".format(gen_img, epoch, song), outputs_rsp, cmap='binary')
    np.save("{0}epoch{1}_{2}.npy".format(gen_npy, epoch, song), outputs_rsp)
    midi_file = get_midifile(outputs_rsp, threshold=gen_threshold, sampling_frequency=48) #fs=48 because pretty_midi creates measures 2s long -> 96/2
    midi_file.write("{0}epoch{1}_{2}.mid".format(gen_midi, epoch, song))
    outputs_rsp_norm = normalize_onoff(outputs_rsp)
    plt.imsave("{0}epoch{1}_norm_{2}.png".format(gen_img, epoch, song), outputs_rsp_norm, cmap='binary')
    np.save("{0}epoch{1}_norm_{2}.npy".format(gen_npy, epoch, song), outputs_rsp_norm)
    midi_file = get_midifile(outputs_rsp_norm, threshold=gen_threshold, sampling_frequency=48) #fs=48 because pretty_midi creates measures 2s long -> 96/2
    midi_file.write("{0}epoch{1}_norm_{2}.mid".format(gen_midi, epoch, song))
    song += 1


#Build Keras model

#Model Parameters
    
n_measures = 16
n_inputs = 96*96
n_hidden1 = 2000
n_hidden2 = 200
n_hidden3 = 1600
n_hidden4 = 120
n_hidden5 = n_hidden3
n_hidden6 = n_hidden2
n_hidden7 = n_hidden1
n_outputs = n_inputs

lr = args.learning_rate

#build encoder Model
inputs = tf.keras.Input(shape=(n_measures,n_inputs), name="inputs")
hidden1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_hidden1, activation='relu'), name="hidden1")(inputs)
hidden2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_hidden2, activation='relu'), name="hidden2")(hidden1)
hidden2_flattened = tf.keras.layers.Flatten(name="hidden2_flattened")(hidden2)
hidden3 = tf.keras.layers.Dense(n_hidden3, activation='relu', name="hidden3")(hidden2_flattened)
hidden4_mean = tf.keras.layers.Dense(n_hidden4, activation='relu', name="hidden4_mean")(hidden3)
hidden4_gamma = tf.keras.layers.Dense(n_hidden4, activation='relu', name="hidden4_gamma")(hidden3)
hidden4 = tf.keras.layers.Lambda(sample, name="hidden4")([hidden4_mean, hidden4_gamma])

#instatiate encoder model
encoder = tf.keras.Model(inputs, [hidden4_mean, hidden4_gamma, hidden4], name="encoder")

#build decoder model
latent_inputs = tf.keras.Input(shape=(n_hidden4,), name="latent_inputs")
hidden5 = tf.keras.layers.Dense(n_hidden5, activation='relu', name="hidden5")(latent_inputs)
hidden6 = tf.keras.layers.Dense(n_measures*n_hidden6, activation='relu', name="hidden6")(hidden5)
hidden6_separated = tf.keras.layers.Lambda(separate, name="hidden6_separated")(hidden6)
hidden7 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_hidden7, activation='relu'), name="hidden7")(hidden6_separated)
outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_outputs, activation='relu'), name="outputs")(hidden7)

#instatiate decoder model
decoder = tf.keras.Model(latent_inputs, outputs, name="decoder")

#instatialte VAE model
outputs = decoder(encoder(inputs)[2])
VAE = tf.keras.Model(inputs, outputs, name="VAE")

reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
reconstruction_loss *= n_inputs
#latent_loss = tf.keras.losses.KLD(hidden4, noise)
latent_loss = 0.5 * tf.keras.backend.sum(tf.keras.backend.exp(hidden4_gamma) + tf.keras.backend.square(hidden4_mean) - 1 - hidden4_gamma, axis=None)
#loss = tf.keras.backend.mean(reconstruction_loss + latent_loss)
#loss = reconstruction_loss - latent_loss
loss = reconstruction_loss + latent_loss
VAE.add_loss(loss)
VAE.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))


#Build data generator

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_dir, batch_size=32, dim=(16,9216), shuffle=True, n_music=3, gen_dir="./generated/", gen_freq=1, gen_threshold=0, init_epoch=0):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.n_music = n_music
        self.gen_dir = gen_dir
        self.epoch = init_epoch
        self.gen_freq = gen_freq
        self.gen_threshold = gen_threshold
        
        self.gen_img = gen_dir + "img/"
        self.gen_npy = gen_dir + "npy/"
        self.gen_midi = gen_dir + "midi/"
        
        os.makedirs(self.gen_img, exist_ok=True)
        os.makedirs(self.gen_npy, exist_ok=True)
        os.makedirs(self.gen_midi, exist_ok=True)
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

#     def on_batch_end(self):
#         print("end of batch")
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            pass
        
        'generate samples'
        if self.epoch % self.gen_freq == 0:
            for song in range(self.n_music):
                sample = decoder(np.random.normal(size=[1, 120]))
                outputs_rsp = np.reshape(sample, (16, 96, 96), order='C')
                outputs_rsp = np.concatenate(outputs_rsp[:], axis=1)
                outputs_rsp = outputs_rsp / np.amax(outputs_rsp)
                plt.imsave("{0}epoch{1}_{2}.png".format(self.gen_img, self.epoch, song), outputs_rsp, cmap='binary')
                np.save("{0}epoch{1}_{2}.npy".format(self.gen_npy, self.epoch, song), outputs_rsp)
                midi_file = get_midifile(outputs_rsp, threshold=self.gen_threshold, sampling_frequency=48) #fs=48 because pretty_midi creates measures 2s long -> 96/2
                midi_file.write("{0}epoch{1}_{2}.mid".format(self.gen_midi, self.epoch, song))
        print("Songs generated at epoch {0}".format(self.epoch))
        self.epoch += 1
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        #y = np.empty((self.batch_size, *self.dim))
        #X = None
        y = None

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = np.load(self.data_dir + ID + '.npy')
            
            #Store class (not actually needed)
            #y[i] = np.load(self.data_dir + ID + '.npy')

        return X, y


#Train the model

#Set paramters
data_dir = args.data_dir
train_fraction = 1 - args.test_fraction
run_id = args.run_id
n_epochs = args.n_epochs
batch_size = args.batch_size
n_music = args.n_music
gen_freq = args.gen_freq
gen_threshold = args.gen_threshold
shuffle = args.shuffle
n_samples_in = args.n_samples
stdout_to_file = args.stdout_to_file
run_info = args.run_info
song = 0

#Build directory
now = datetime.now().strftime("%Y%m%d%H%M%S")
rundir = "./runs/{0}_{1}/".format(now, run_id)
logdir = "./runs/{0}_{1}/graphs/".format(now, run_id)
gendir = "./runs/{0}_{1}/generated/".format(now, run_id)
checkpointdir = "./runs/{0}_{1}/checkpoints/".format(now, run_id)
os.makedirs(rundir, exist_ok=True)
os.makedirs("{0}/train/plugins/profile/".format(logdir), exist_ok=True)
os.makedirs(gendir, exist_ok=True)
os.makedirs(checkpointdir, exist_ok=True)

#Write .txt files
if stdout_to_file:
    stdout = open(rundir + "/{0}_{1}_stdout.txt".format(now, run_id), 'w')
    sys.stdout = stdout
    print('Python Console Output:')

with open(rundir + "/{0}_{1}_info.txt".format(now, run_id), 'w') as file:
    file.write("""{17}
Network_Keras_partseparated.ipynb
train with .fit_generator
    
run_id= \t{0}
data_dir= \t{1}
test_fraction= \t{2}
n_epochs= \t{3}
batch_size= \t{4}
n_music= \t{5}
gen_freq= \t{6}
gen_threshold= \t{15}
shuffle= \t{7}
n_samples_in = \t{18}
stdout_to_file = \t{19}
run_info= \t{8}

n_measures= \t{9}
n_inputs= \t{10}
n_hidden1= \t{11}
n_hidden2= \t{12}
n_hidden3= \t{13}
n_hidden4= \t{14}
lr= \t\t{16}
""".format(run_id, data_dir, train_fraction, n_epochs, batch_size, n_music, gen_freq, shuffle, run_info, 
                           n_measures, n_inputs, n_hidden1, n_hidden2, n_hidden3, n_hidden4, gen_threshold, lr, now, n_samples_in, stdout_to_file))

#build list and set parameters for DataGenerator
id_list = []
n_samples = len(os.listdir(data_dir)) if n_samples_in == 0 else n_samples_in
for k in range(n_samples):
    id_list.append(str(k))
train_id_list = id_list[:round(train_fraction*len(id_list))]
test_id_list = id_list[round(train_fraction*len(id_list))+1:]

params = {'dim': (n_measures,9216),
          'batch_size': batch_size,
          'shuffle': shuffle,
         'n_music': n_music,
         'gen_dir': gendir,
         'gen_freq': gen_freq,
         'gen_threshold': gen_threshold}

#Build callbacks and datagenerators
checkpoint_cbk = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointdir, save_weights_only=False, save_freq=1)
tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_images=False, update_freq='batch')
csv_cbk = tf.keras.callbacks.CSVLogger(rundir + "{0}_{1}_csv.log".format(now, run_id), separator=";")
training_generator = DataGenerator(train_id_list, data_dir=data_dir, **params)
testing_generator = DataGenerator(test_id_list, data_dir=data_dir, **params) if len(test_id_list) != 0 else None

#Train
try:
    train_history = VAE.fit_generator(generator=training_generator, epochs=n_epochs, callbacks=[tensorboard_cbk, csv_cbk], validation_data=testing_generator, use_multiprocessing=False)
except Exception as e:
    print("\n", e)
    generate_song()

decoder.save('{0}/decoder_pt.h5'.format(checkpointdir))
encoder.save('{0}/encoder_pt.h5'.format(checkpointdir))
print("\nsaved Models")
if stdout_to_file:
    stdout.close()