# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:48:51 2020

@author: manue
"""

import numpy as np
import os
import pretty_midi
import argparse


#get arguments
parser = argparse.ArgumentParser()

parser.add_argument('--n-measures', 
                    type=int, 
                    default=16, 
                    help='default=16')
parser.add_argument('--data-dir', 
                    type=str, 
                    default="./data/", 
                    help='default="./data/"')
parser.add_argument('--dest-dir', 
                    type=str, 
                    default="./dest/", 
                    help='default="./dest/')
parser.add_argument('--files-at-once', 
                    type=int, 
                    default=100, 
                    help='default=100')
parser.add_argument('--noduration', 
                    type=bool, 
                    default=True, 
                    help='default=True')

args, _ = parser.parse_known_args()


#helper functions

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


#load data

os.makedirs(args.dest_dir, exist_ok=True)
file_list = os.listdir(args.data_dir)
n_samples = 0

while len(file_list) != 0:
    
    files = file_list[:args.files_at_once]
    directory = args.data_dir
    img_size = 96
    warnings = True
    array_list_full = []
    counter = 0
    
    for file in files:
        try:
            data = pretty_midi.PrettyMIDI(directory + file)
            time_signature_list = data.time_signature_changes
            if len(time_signature_list) == 1:
                time_signature = [time_signature_list[0].numerator, time_signature_list[0].denominator]
                t_measure = data.get_beats()[time_signature[0]]
                
                if args.noduration:    
                    for instrument in data.instruments:
                        if not instrument.is_drum:
                            for note in instrument.notes:
                                note.end = note.start + t_measure/96
                
                array = data.get_piano_roll(fs=img_size/t_measure)
                array_shortened = np.delete(array, np.arange(array.shape[1]-(array.shape[1]%img_size), array.shape[1]), axis=1)
                array_list = np.array_split(array_shortened, array_shortened.shape[1]/img_size, axis=1)
                array_list_full += array_list
                counter += 1
                print("loaded {0}/{1} files".format(counter, len(files)), end="\r")
                pass
            
            else:
                if warnings:
                    print("WARNING: '{0}' was ignored (time signature changes)".format(file))
                    print("loaded {0}/{1} files".format(counter, len(files)), end="\r")
                pass
            pass
        except: 
            if warnings:
                print("WARNING: '{0}' could not be loaded (pretty_midi error)".format(file))
                print("loaded {0}/{1} files".format(counter, len(files)), end="\r")
    
    print("loaded {0}/{1} files".format(counter, len(files)))
    print("Stacking...                          ")        
    array_full = np.stack(array_list_full, axis=0)
    print("Shortening...")
    array_full_cut = np.delete(np.delete(array_full, range(16), axis=1), range(96,112), axis=1) #TODO: cut according to img_size
    print("Done")
    data = array_full_cut
    
    data = normalize_onoff(data)
    input_data = np.expand_dims(data, 0)
    if input_data.shape[1] % args.n_measures != 0:
        input_data = np.delete(input_data, range(input_data.shape[1]-input_data.shape[1]%args.n_measures, input_data.shape[1]), axis=1)
    input_data = np.reshape(input_data, (-1, args.n_measures, 96*96))
    counter = 0
    for m in range(input_data.shape[0]):
        np.save("./{0}/{1}.npy".format(args.dest_dir, n_samples+m), input_data[m])
        counter += 1
        print("saved {0}/{1} files".format(counter, input_data.shape[0]), end="\r")
        pass
    n_samples += input_data.shape[0]
    del file_list[:args.files_at_once]