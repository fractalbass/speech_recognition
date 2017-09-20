import os
import argparse

import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc
from HMMTrainer import HMMTrainer



# Function to parse input arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the HMM classifier')
    parser.add_argument("--input-folder", dest="input_folder", required=True,
            help="Input folder containing the audio files in subfolders")
    return parser

if __name__=='__main__':
    #args = build_arg_parser().parse_args()
    #input_folder = args.input_folder
    input_folder = "./data"
    hmm_models = []
    # Parse the input directory
    for dirname in os.listdir(input_folder):
        # Get the name of the subfolder
        subfolder = os.path.join(input_folder, dirname)

        if not os.path.isdir(subfolder):
            continue

        # Extract the label
        label = subfolder[subfolder.rfind('/') + 1:]
        # Initialize variables
        X = np.array([])
        y_words = []
        # Iterate through the audio files (leaving 1 file for testing in each class)
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            # Read the input file
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = wavfile.read(filepath)
            # Extract MFCC features
            mfcc_features = mfcc(audio, sampling_freq, nfft=1200)
            # Append to the variable X
            if len(X) == 0:
                X = mfcc_features
            else:
                X = np.append(X, mfcc_features, axis=0)
                # Append the label
                y_words.append(label)
                # Train and save HMM model
                hmm_trainer = HMMTrainer()
                hmm_trainer.train(X)
                hmm_trainer = None

    # Test files
    input_files = [
        'data/pineapple/pineapple15.wav',
        'data/orange/orange15.wav',
        'data/apple/apple15.wav',
        'data/kiwi/kiwi15.wav'
    ]    # Classify input data
    for input_file in input_files:
        # Read input file
        sampling_freq, audio = wavfile.read(input_file)
        # Extract MFCC features
        mfcc_features = mfcc(audio, sampling_freq)
        # Define variables
        max_score = None
        output_label = None

        # Iterate through all HMM models and pick
        # the one with the highest score
        for item in hmm_models:
            hmm_model, label = item
            score = hmm_model.get_score(mfcc_features)
            if max_score is None or score > max_score:
                max_score = score
                output_label = label

        # Print the output
        print("\nTrue:", input_file[input_file.find('/') + 1:input_file.rfind('/')])
        print("Predicted:", output_label)