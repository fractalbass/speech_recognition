import os
import argparse

import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc
from HMMTrainer import HMMTrainer
import warnings



class digit_recognizer:

    digit_directory = '/Users/milesporter/data-science/data-sets/free-spoken-digit-dataset/recordings'
    test_directory = '/Users/milesporter/data-science/speech_recognition/data/digit_test_data'
    nfft = 1203 # Number of FFTs

    def run(self):

        processed_files = list()
        mfcc_features = self.process_directory()
        for feature in mfcc_features:
            processed_files.append({'label': feature["label"], 'feature': feature["mfcc"]})

        # Train HMM for each MFCC and add to training set
        for processed_file in processed_files:
            X = processed_file['feature']
            hmm_trainer = HMMTrainer()
            hmm_trainer.train(X)
            processed_file['hmm_trainer'] = hmm_trainer

        # Run through test data and find matching label
        for filename in [x for x in os.listdir(self.test_directory) if x.endswith('.wav')]:

            # Read the input file
            filepath = os.path.join(self.test_directory, filename)
            sampling_freq, audio = wavfile.read(filepath)
            test_features = mfcc(audio, sampling_freq, nfft = self.nfft)
            max_score = None

            for item in processed_files:
                hmm_model = item['hmm_trainer']

                score = hmm_model.get_score(test_features)
                if max_score is None or score > max_score:
                    max_score = score
                    label = item['label']

            print("Filename: {0},  Digit: {1}".format(filename, label))

        print("Done.")

    def process_directory(self):
        mfcc_features = list()
        for filename in [x for x in os.listdir(self.digit_directory) if x.endswith('.wav')]:

            # Read the input file
            filepath = os.path.join(self.digit_directory, filename)
            sampling_freq, audio = wavfile.read(filepath)
            label = self.get_label(filename)
            # Extract MFCC features and append to list
            mfcc_features.append({"label": label, "mfcc":mfcc(audio, sampling_freq, nfft=self.nfft)})
        return mfcc_features


    def get_label(self, filename):
        #Files are assumed to be in the format {digitLabel}_{speakerName}_{index}.wav
        return filename.split("_")[0]


if __name__ == "__main__":
    dr = digit_recognizer()
    dr.run()


