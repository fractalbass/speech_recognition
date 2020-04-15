import os
import argparse
from glob import glob
import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
from python_speech_features import mfcc
from HMMTrainer import HMMTrainer
import warnings



class digit_recognizer:

    digit_directory = './data/digits'
    test_directory = './data/digit_test_data'
    nfft = 1203 # Number of FFTs

    def run(self):

        training_files = list()
        mfcc_features = self.process_training_directory()
        for feature in mfcc_features:
            training_files.append({'label': feature["label"], 'feature': feature["mfcc"]})

        # Train HMM for each MFCC and add to training set
        for training_file in training_files:
            X = training_file['feature']
            hmm_trainer = HMMTrainer()
            hmm_trainer.train(X)
            training_file['hmm_trainer'] = hmm_trainer

        # Run through test data and find matching label
        #for filename in [x for x in os.listdir(self.test_directory) if x.endswith('.wav')]:

        for testfile in glob("{0}/*.wav".format(self.test_directory)):
            # Read the input file
            sampling_freq, audio = wavfile.read(testfile)
            test_features = mfcc(audio, sampling_freq, nfft = self.nfft)
            max_score = None

            for item in training_files:
                hmm_model = item['hmm_trainer']

                score = hmm_model.get_score(test_features)
                if max_score is None or score > max_score:
                    max_score = score
                    label = item['label']

            print("Test file: {0},  Recognized digit: {1}".format(testfile, label))

        print("Done.")

    def process_training_directory(self):
        mfcc_features = list()
        for filename in glob("{0}/**/*.wav".format(self.digit_directory)):

            # Read the input file
            sampling_freq, audio = wavfile.read(filename)
            label = self.get_training_label(filename)
            # Extract MFCC features and append to list
            mfcc_features.append({"label": label, "mfcc":mfcc(audio, sampling_freq, nfft=self.nfft)})
        return mfcc_features


    def get_training_label(self, filename):
        #Files are assumed to be in the format {digitLabel}_{speakerName}_{index}.wav
        basename = os.path.basename(filename)
        return basename.split("-")[0]

    def get_test_label(self, filename):
        return filename.split("_")[0]

if __name__ == "__main__":
    dr = digit_recognizer()
    dr.run()


